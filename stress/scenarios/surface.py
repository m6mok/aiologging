"""
Feature-surface scenarios: subsystems the other categories never
touch, each stressed on the dimension where it can actually break.

- content integrity: the bytes that arrive must equal the bytes sent
  (formatters, unicode payloads, ``exc_info`` tracebacks) — every
  other category only counts records;
- stream handlers: two handlers sharing one buffer under concurrent
  producers must not interleave or lose lines;
- terminal failure: a permanently failing sink must exhaust its
  retries, fire ``error_handler`` exactly once per record and leave
  the pipeline alive;
- secret redaction: a failing Telegram sink must never leak the bot
  token into error text or ``repr``;
- ConfigManager (subprocess): a configured pipeline must survive a
  concurrent-construction race and deliver everything to the file;
- ``captureStdlib`` (subprocess): the global install path — cold
  records before any loop, thread producers, loop churn, reinstall
  idempotency and clean removal.

The subprocess scenarios mutate global state (the stdlib root, the
global manager, the global config registry) by design — that is the
code path under test — so they run in a child interpreter, mirroring
``delivery.atexit_drain``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import subprocess
import sys
from logging import LogRecord
from pathlib import Path
from typing import Dict, List, Tuple

from ..metrics import wait_until
from ..runner import Context, Skip, scenario
from ..sinks import CollectorHandler


# ----------------------------------------------------------------------
# Content integrity
# ----------------------------------------------------------------------


@scenario("surface.payload_integrity")
async def payload_integrity(ctx: Context) -> None:
    """Delivered text equals sent text; exc_info tracebacks intact."""
    manager = ctx.new_manager()
    sink = CollectorHandler(
        track=True,
        store=True,
        formatter=logging.Formatter("%(levelname)s %(message)s"),
    )
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    producers = 3
    count = ctx.n(1_500, 250)

    def payload(worker: int, seq: int) -> str:
        return f"p{worker}-s{seq}-" + "xyz" * (seq % 40) + "-π✓"

    async def produce_one(worker: int) -> None:
        for seq in range(count):
            await logger.info(
                "%s",
                payload(worker, seq),
                extra={"seq": seq, "producer": worker},
            )

    await asyncio.gather(
        *(produce_one(worker) for worker in range(producers))
    )

    exc_count = ctx.n(200, 40)
    for seq in range(exc_count):
        try:
            raise ValueError(f"boom-{seq}")
        except ValueError:
            await logger.error(
                "err-%d",
                seq,
                exc_info=True,
                extra={"seq": seq, "producer": 900},
            )
    await manager.flush()

    expected = {
        f"INFO {payload(worker, seq)}"
        for worker in range(producers)
        for seq in range(count)
    }
    got_info = [m for m in sink.stored if m.startswith("INFO ")]
    got_errors = [m for m in sink.stored if m.startswith("ERROR ")]

    corrupted = [m for m in got_info if m not in expected]
    missing = expected - set(got_info)
    bad_tracebacks = []
    for message in got_errors:
        match = re.match(r"ERROR err-(\d+)", message)
        seq = int(match.group(1)) if match else -1
        if (
            match is None
            or "Traceback (most recent call last)" not in message
            or f"ValueError: boom-{seq}" not in message
        ):
            bad_tracebacks.append(message[:120])

    ctx.metrics.update(
        {
            "records_sent": producers * count + exc_count,
            "records_delivered": sink.received,
            "payload_records": len(got_info),
            "exc_records": len(got_errors),
        }
    )
    ctx.check(
        "every payload arrived byte-identical",
        not corrupted and not missing,
        f"corrupted={len(corrupted)} missing={len(missing)}",
    )
    ctx.check(
        "no payload duplicated",
        len(got_info) == producers * count,
        f"got={len(got_info)} sent={producers * count}",
    )
    ctx.check(
        "every exc_info record carries its own traceback",
        len(got_errors) == exc_count and not bad_tracebacks,
        f"errors={len(got_errors)}/{exc_count} "
        f"bad={bad_tracebacks[:1]}",
    )


# ----------------------------------------------------------------------
# Stream handlers under concurrency
# ----------------------------------------------------------------------


@scenario("surface.stream_contention")
async def stream_contention(ctx: Context) -> None:
    """Two stream handlers on one buffer: every line lands intact."""
    from aiologging.handlers.stream import AsyncStreamHandler

    manager = ctx.new_manager()
    stream = io.StringIO()
    logger = ctx.new_logger(manager)
    for _ in range(2):
        logger.addHandler(AsyncStreamHandler(stream=stream))

    producers = 4
    count = ctx.n(1_200, 200)

    async def produce_one(worker: int) -> None:
        for seq in range(count):
            await logger.info("line-%d-%d", worker, seq)

    await asyncio.gather(
        *(produce_one(worker) for worker in range(producers))
    )
    await manager.flush()

    lines = stream.getvalue().splitlines()
    malformed = [
        line for line in lines if not re.fullmatch(r"line-\d+-\d+", line)
    ]
    counts: Dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    # each of the two handlers writes every record once
    uneven = {k: v for k, v in counts.items() if v != 2}
    expected_lines = producers * count * 2

    ctx.metrics.update(
        {
            "records_sent": producers * count,
            "lines_written": len(lines),
            "handlers": 2,
        }
    )
    ctx.check(
        "no line interleaved or torn",
        not malformed,
        f"malformed={malformed[:2]}",
    )
    ctx.check(
        "every record written exactly once per handler",
        len(lines) == expected_lines and not uneven,
        f"lines={len(lines)}/{expected_lines} uneven={len(uneven)}",
    )


# ----------------------------------------------------------------------
# Terminal failure: retries exhausted -> error_handler
# ----------------------------------------------------------------------


@scenario("surface.retry_exhaustion")
async def retry_exhaustion(ctx: Context) -> None:
    """Give-up branch: error_handler once per record, pipeline lives."""
    manager = ctx.new_manager()
    failures: List[int] = []

    async def on_error(record: LogRecord, error: Exception) -> None:
        failures.append(record.__dict__.get("seq", -1))

    sink = CollectorHandler(
        fail_every=1,  # every attempt fails: retries must exhaust
        track=True,
        retry_attempts=2,
        error_handler=on_error,
    )
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    count = ctx.n(400, 80)
    for seq in range(count):
        await logger.info(
            "dead %d", seq, extra={"seq": seq, "producer": 0}
        )
    await manager.flush()

    attempts_dead = sink.attempts
    delivered_dead = sink.received
    errored = list(failures)

    # the sink recovers: the pipeline must still deliver
    sink.fail_every = 0
    for seq in range(count):
        await logger.info(
            "alive %d", seq, extra={"seq": seq, "producer": 1}
        )
    await manager.flush()
    alive = sink.by_producer.get(1, [])

    ctx.metrics.update(
        {
            "records_sent": 2 * count,
            "dead_attempts": attempts_dead,
            "errors_reported": len(errored),
            "records_delivered": sink.received,
        }
    )
    ctx.check(
        "error_handler fired exactly once per dead record",
        len(errored) == count and len(set(errored)) == count,
        f"reported={len(errored)} unique={len(set(errored))} "
        f"sent={count}",
    )
    ctx.check(
        "every dead record exhausted its retries",
        attempts_dead == count * 3,
        f"attempts={attempts_dead} expected={count * 3}",
    )
    ctx.check(
        "nothing was delivered while the sink was dead",
        delivered_dead == 0,
        f"delivered={delivered_dead}",
    )
    ctx.check(
        "give-ups are not counted as queue drops",
        int(manager.get_metrics()["records_dropped"]) == 0,
    )
    ctx.check(
        "the pipeline recovered after the give-ups",
        len(alive) == count and sink.ordered_per_producer(),
        f"alive={len(alive)}/{count}",
    )


# ----------------------------------------------------------------------
# Secret redaction on the error path
# ----------------------------------------------------------------------


@scenario("surface.secret_redaction")
async def secret_redaction(ctx: Context) -> None:
    """A failing Telegram sink never leaks the bot token."""
    try:
        import httpx
    except ImportError:
        raise Skip("httpx not installed")
    from aiologging.handlers.telegram import AsyncTelegramHandler
    from aiologging.types import BatchConfig

    token = "987654:SECRET-TOKEN-DO-NOT-LEAK"
    errors: List[str] = []

    async def on_error(record: LogRecord, error: Exception) -> None:
        errors.append(f"{error!r} {error}")

    def respond(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(500)

    handler = AsyncTelegramHandler(
        token=token,
        chat_id="42",
        backend="httpx",
        batch_config=BatchConfig(
            batch_size=5,
            flush_interval=0.05,
            max_retries=1,
            retry_delay=0.001,
        ),
        error_handler=on_error,
    )
    handler._create_session = (  # type: ignore[method-assign]
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    manager = ctx.new_manager()
    logger = ctx.new_logger(manager)
    logger.addHandler(handler)

    count = ctx.n(120, 30)
    for seq in range(count):
        await logger.info("pager %d", seq)
    await manager.flush()
    await wait_until(lambda: len(errors) > 0, timeout_s=10.0)

    text = "\n".join(errors) + repr(handler)
    ctx.metrics.update(
        {
            "records_sent": count,
            "errors_captured": len(errors),
        }
    )
    ctx.check("the sink actually failed", len(errors) > 0)
    ctx.check(
        "the bot token never leaked",
        token not in text and f"bot{token}" not in text,
    )
    ctx.check(
        "errors carry the redacted URL instead",
        "***" in text,
        text[:160],
    )


# ----------------------------------------------------------------------
# ConfigManager pipeline (subprocess: the config registry and the
# configured loggers are global by design)
# ----------------------------------------------------------------------


_CONFIG_CHILD = """\
import asyncio
import sys
import threading

import aiologging
from aiologging.config import configure_from_dict, get_configured_logger

log_path = sys.argv[1]
count = int(sys.argv[2])
producers = 3

configure_from_dict(
    {
        "version": 1,
        "handlers": {
            "logfile": {
                "class": "file",
                "level": "INFO",
                "filename": log_path,
            },
        },
        "loggers": {
            "app": {
                "level": "INFO",
                "handlers": ["logfile"],
                "propagate": False,
            },
        },
    }
)

# Concurrent construction: racing threads must converge on a single
# logger instance (two instances would attach two file handlers and
# duplicate every line).
racers = 6
barrier = threading.Barrier(racers)
seen = []


def race() -> None:
    barrier.wait()
    seen.append(id(get_configured_logger("app")))


threads = [threading.Thread(target=race) for _ in range(racers)]
for t in threads:
    t.start()
for t in threads:
    t.join()


async def main() -> None:
    logger = get_configured_logger("app")

    async def produce(worker: int) -> None:
        for seq in range(count):
            await logger.info("cfg %d %d", worker, seq)

    await asyncio.gather(*(produce(w) for w in range(producers)))
    await aiologging.shutdown()


asyncio.run(main())
sys.stdout.write("unique_loggers=%d\\n" % len(set(seen)))
"""


@scenario("surface.config_pipeline", timeout=120.0)
def config_pipeline(ctx: Context) -> None:
    """ConfigManager: constructed pipeline delivers, no ctor races."""
    try:
        import aiofiles  # noqa: F401
    except ImportError:
        raise Skip("aiofiles not installed")

    count = ctx.n(2_000, 400)
    producers = 3
    proc, out_path = _run_child(
        ctx, _CONFIG_CHILD, "config_child.py", "config.log", [str(count)]
    )
    unique_loggers = _child_int(proc.stdout, "unique_loggers")
    text = (
        out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    )
    pairs = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"cfg (\d+) (\d+)", text)
    ]
    expected = {
        (worker, seq)
        for worker in range(producers)
        for seq in range(count)
    }
    missing = expected - set(pairs)
    duplicates = len(pairs) - len(set(pairs))

    ctx.metrics.update(
        {
            "records_sent": producers * count,
            "lines_in_file": len(pairs),
            "unique_loggers": unique_loggers,
        }
    )
    ctx.check(
        "child exited cleanly",
        proc.returncode == 0,
        f"rc={proc.returncode} stderr={proc.stderr[-300:]!r}",
    )
    ctx.check(
        "racing threads got one logger instance",
        unique_loggers == 1,
        f"unique_loggers={unique_loggers}",
    )
    ctx.check(
        "every configured record reached the file",
        not missing,
        f"missing={len(missing)}",
    )
    ctx.check(
        "no record duplicated",
        duplicates == 0,
        f"duplicates={duplicates}",
    )


# ----------------------------------------------------------------------
# captureStdlib global install path (subprocess: it mutates the
# stdlib root logger and the global manager by design)
# ----------------------------------------------------------------------


_CAPTURE_CHILD = """\
import asyncio
import logging
import sys

import aiologging

log_path = sys.argv[1]
count = int(sys.argv[2])
threads_n = int(sys.argv[3])
pre = int(sys.argv[4])

root = aiologging.getLogger()
root.setLevel("INFO")
root.addHandler(aiologging.create_file_handler(log_path))

aiologging.captureStdlib(True)
lib = logging.getLogger("bridged.lib")
lib.setLevel(logging.INFO)

# cold records: no event loop exists anywhere yet
for seq in range(pre):
    lib.info("cold %d", seq)


async def phase1() -> None:
    await aiologging.getLogger("warmup").info("warmup")

    def worker(worker_id: int) -> None:
        for seq in range(count):
            lib.info("t%d %d", worker_id, seq)

    await asyncio.gather(
        *(asyncio.to_thread(worker, i) for i in range(threads_n))
    )
    await aiologging.flush()


asyncio.run(phase1())

# the loop is gone: these wait in the cold buffer for the next one
for seq in range(pre):
    lib.info("gap %d", seq)

aiologging.captureStdlib(True)  # reinstall must replace, not stack
stdlib_root = logging.getLogger()
bridges = sum(
    1
    for h in stdlib_root.handlers
    if type(h).__name__ == "StdlibBridgeHandler"
)


async def phase2() -> None:
    await aiologging.flush()
    await aiologging.shutdown()


asyncio.run(phase2())

aiologging.captureStdlib(False)
left = sum(
    1
    for h in stdlib_root.handlers
    if type(h).__name__ == "StdlibBridgeHandler"
)
sys.stdout.write("bridges=%d\\n" % bridges)
sys.stdout.write("left=%d\\n" % left)
"""


@scenario("surface.capture_stdlib_global", timeout=120.0)
def capture_stdlib_global(ctx: Context) -> None:
    """captureStdlib: global install survives churn, loses nothing."""
    try:
        import aiofiles  # noqa: F401
    except ImportError:
        raise Skip("aiofiles not installed")

    count = ctx.n(1_500, 300)
    threads_n = 4
    pre = ctx.n(200, 40)
    proc, out_path = _run_child(
        ctx,
        _CAPTURE_CHILD,
        "capture_child.py",
        "capture.log",
        [str(count), str(threads_n), str(pre)],
    )
    bridges = _child_int(proc.stdout, "bridges")
    left = _child_int(proc.stdout, "left")
    text = (
        out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    )
    cold = re.findall(r"cold (\d+)", text)
    gap = re.findall(r"gap (\d+)", text)
    threaded = re.findall(r"t(\d+) (\d+)", text)
    expected_threaded = {
        (str(worker), str(seq))
        for worker in range(threads_n)
        for seq in range(count)
    }
    missing_threaded = expected_threaded - set(threaded)

    ctx.metrics.update(
        {
            "records_sent": 2 * pre + threads_n * count + 1,
            "cold_delivered": len(cold),
            "gap_delivered": len(gap),
            "threaded_delivered": len(threaded),
        }
    )
    ctx.check(
        "child exited cleanly",
        proc.returncode == 0,
        f"rc={proc.returncode} stderr={proc.stderr[-300:]!r}",
    )
    ctx.check(
        "records logged before any loop were delivered",
        sorted(int(s) for s in cold) == list(range(pre)),
        f"cold={len(cold)}/{pre}",
    )
    ctx.check(
        "records logged between loops were delivered",
        sorted(int(s) for s in gap) == list(range(pre)),
        f"gap={len(gap)}/{pre}",
    )
    ctx.check(
        "every thread record came through the global bridge",
        not missing_threaded
        and len(threaded) == len(expected_threaded),
        f"missing={len(missing_threaded)} "
        f"extra={len(threaded) - len(expected_threaded)}",
    )
    ctx.check(
        "reinstall replaced the bridge instead of stacking",
        bridges == 1,
        f"bridges={bridges}",
    )
    ctx.check(
        "captureStdlib(False) removed the bridge",
        left == 0,
        f"left={left}",
    )


# ----------------------------------------------------------------------
# Shared subprocess plumbing
# ----------------------------------------------------------------------


def _run_child(
    ctx: Context,
    source: str,
    script_name: str,
    output_name: str,
    args: List[str],
) -> Tuple["subprocess.CompletedProcess[str]", Path]:
    """Run a child interpreter script against a tmpdir output file."""
    import aiologging

    script = ctx.tmpdir / script_name
    script.write_text(source, encoding="utf-8")
    out_path = ctx.tmpdir / output_name
    env = dict(os.environ)
    env["PYTHONPATH"] = str(
        Path(aiologging.__file__).resolve().parents[1]
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(out_path), *args],
        capture_output=True,
        text=True,
        timeout=90.0,
        env=env,
        cwd=str(ctx.tmpdir),
    )
    return proc, out_path


def _child_int(stdout: str, key: str) -> int:
    """Parse ``key=N`` from a child's stdout (-1 when absent)."""
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return int(line.split("=", 1)[1])
    return -1
