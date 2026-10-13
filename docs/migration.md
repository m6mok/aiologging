# Migration

## Standard logging applications

Replace the application import with `import aiologging as logging`. Add await
to logger/module debug/info/warning/error/critical/exception/log and direct
handle/callHandlers operations. Configure with normal basicConfig, dictConfig
or fileConfig. Dependency imports of stdlib logging need no bridge: they share
the actual logger hierarchy and synchronously wait for durable acceptance.

Logger subclasses and Handler remain synchronous stdlib extension bases.
Direct Logger/RootLogger construction produces an async view. Set custom logger
classes with setLoggerClass and retrieve their async view with getLogger.
Directly using a stdlib LoggerAdapter and
custom convenience methods delegated from a synchronous class remain
synchronous. Use aiologging.LoggerAdapter for awaitable adapter methods.
A custom class that fully replaces callHandlers without calling super also
replaces journal dispatch; this is an explicit custom-dispatch boundary.
Mutating a record inside custom emit/handle cannot affect another recipient's
already captured snapshot: asynchronous recipients are independent.

Global mutable flags, factories, logger levels/disable/propagate, actual
handler identity, formatters, callable filters, 3.12+ replacement filters,
configuration hooks and incremental configurations use stdlib state.
getLogger/parent/getChild cache async views; the underlying stdlib manager
retains synchronous actual loggers. The two views share configuration but are
not the same object. Documented names introduced by a particular Python
version are forwarded from that running version; getChildren is backported
to 3.11 by using the real hierarchy.

## Durable restrictions

Record snapshots preserve ready message text, standard fields, exception
text, stack text and JSON-valued extra, not original objects/traceback frames.
Set extra_encoder explicitly for custom objects; unsupported values reject
acceptance. Recovery restores a standard LogRecord, not arbitrary factory
subclasses with executable behaviour. Exceptions/stacklevel are captured at
the call site, before background I/O.

Set configureJournal(application="name") for multiple entry points, or pass
a shared local filesystem path. All processes must agree on limits, and
use spawn. Never delete a journal or unfinished records to relieve pressure.
Quota/disk/permission/admission failures raise and diagnose on stderr.

File/stream destinations use a hash of their path or standard stream and
configured name. Arbitrary custom targets require an explicit stable
handler.recipient_id describing the actual destination. Reusing an ID for a
new destination would falsely assert that it is the same recipient: choose a
new ID instead. Replace handlers when changing targets; unknown or removed
original destinations retain tasks and diagnose. Journal.reassign(old, new)
is the explicit reassignment operation and requires both senders stopped.

Our HTTP/Telegram adapters require a fallback callable confirming True.
A confirmed FileHandler.deliver works without optional packages. Use a
session_factory for HTTP test/configuration hooks; create session resources
inside that factory on the library loop. Do not return an existing session
from the application loop. No ready-session constructor argument is accepted. Changing a network target
requires a replacement handler; URL, headers and fallback are immutable.
For a custom fallback callable, provide fallback_id or a stable recipient_id
covering both primary and fallback; file/standard-stream targets are inferred.
Our transport loggers are filtered during delivery to prevent feedback. Other
recursive calls from delivery callbacks explicitly reject acceptance.

Arbitrary stdlib/custom handlers keep standard error/close behaviour. They do
not prove actual delivery if emit swallows errors. Implement an explicit
`deliver(records)` returning exactly True only on success for strict ACKs.
QueueHandler/MemoryHandler and other stdlib buffering adapters retain their
ordinary processing contract; durable acceptance does not transform them
into remote delivery confirmation.

## Previous aiologging API

Use aiologging.legacy and aiologging.legacy_config temporarily for old
AsyncLoggerManager, Async* handlers, captureStdlib, drop policies and inline
alerts. That pipeline retains its historical memory-only semantics. It is
covered by the retained historical regression/stress suites. The main API
has no silent drop modes or priority bypass of durable per-recipient order.
