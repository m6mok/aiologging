"""
Delivery guarantees at process death: flush_sync and bounded shutdown.

Records are enqueued at the call site and delivered in the background,
so the ones that matter most — "Service crashed" right before a
``raise``, "Received SIGTERM" during a container swap — are emitted
exactly when the event loop is about to die. This example shows the
three tools that close that window:

- ``shutdown(timeout=...)`` — drain bounded by a grace period while
  the loop is still alive (fits Docker's ~10 s SIGTERM budget);
- ``flush_sync(timeout=...)`` — synchronous drain on a private event
  loop when no loop is running anymore (``finally`` after
  ``asyncio.run``, plain sync code);
- the automatic atexit drain — on by default with a 2 s budget, so
  even a forgotten shutdown delivers what is still queued; configure
  it with ``basicConfig(atexit_flush=...)`` or ``set_atexit_flush``.

Note: atexit does not run when the process dies from an unhandled
signal — handle SIGTERM (e.g. call ``sys.exit``) for the drain to
cover container shutdowns.

Run:
    python examples/emergency_flush.py
"""

import asyncio

import aiologging


async def main() -> None:
    aiologging.basicConfig(
        level=aiologging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        atexit_flush=2.0,  # the default; 0 disables the safety net
    )
    logger = aiologging.getLogger("app")

    await logger.info("Service started")

    try:
        raise RuntimeError("simulated crash")
    except RuntimeError:
        # The record is only enqueued here; delivery happens later
        await logger.exception("Service crashed")
        # Bounded drain while the loop is alive: fit it into the
        # container's termination grace period
        await aiologging.shutdown(timeout=5.0)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        # The loop is gone; anything still undelivered (e.g. records
        # logged by exception handlers after shutdown) can be drained
        # synchronously on a private loop
        delivered = aiologging.flush_sync(timeout=2.0)
        print(f"emergency drain complete, fully delivered: {delivered}")

    # Even without any of the above, the atexit hook would spend up
    # to 2 seconds delivering leftover records before the interpreter
    # exits, and warn on stderr about anything it could not deliver.
