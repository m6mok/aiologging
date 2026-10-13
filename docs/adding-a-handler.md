# Adding a confirmed handler

1. Use logging.Handler as the synchronous configuration/filter/formatter
   base. Set a stable recipient_id describing the actual destination.
2. Implement deliver(records), returning exactly True only after the whole
   batch succeeds. Raise or return False for failure. A partially successful
   batch may be repeated: the contract is at least once.
3. Blocking methods run on a dedicated worker. Async deliver/aclose methods
   run on the library loop; create loop-bound resources lazily there, through
   factories, and close them with aclose. Do not retain application sessions.
4. Set batch_size, primary_timeout, retry_delay and optional fallback_timeout.
   Our network extensions require a separately confirming fallback; success
   is recorded without removing the primary task. Budgets cannot forcibly
   stop arbitrary blocking Python code.
5. Keep configuration targets stable. Construct a replacement handler to
   change the endpoint; never repurpose an ID while old journal tasks exist.
6. Add exports, optional package extras when needed, offline examples, strict
   mypy annotations and tests for failure/recovery/lifecycle/ownership, not
   just the successful API path. Register through normal basicConfig or
   aiologging.config dictConfig/fileConfig; no separate ConfigManager exists
   in the durable API.

Use extensions/files.py and extensions/http.py as examples. Ordinary custom
handlers with only emit/handle keep stdlib behaviour; returning from a method
which hides errors in handleError is not confirmed actual delivery.

The historical AsyncHandler/BufferedAsyncHandler extension contract lives in
legacy-adding-a-handler.md and belongs to aiologging.legacy only.
