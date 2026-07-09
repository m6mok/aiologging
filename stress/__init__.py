"""
Stress-test harness for aiologging.

Not a unit-test suite: scenarios push the library well past normal
load (queue overflow, slow/failing/hanging handlers, event-loop
churn, producer threads) and verify both throughput numbers and
correctness invariants (accounting identities, per-producer ordering,
no lost records).

Run with::

    python -m stress run            # full run
    python -m stress run --quick    # scaled-down smoke run
    python -m stress list           # available scenarios

Everything runs offline: HTTP scenarios use ``httpx.MockTransport``,
file scenarios write to a per-scenario temporary directory.
"""
