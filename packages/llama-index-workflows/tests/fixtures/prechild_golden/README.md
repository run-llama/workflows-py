# Pre-child golden fixtures

Provenance: generated from **pre-child `origin/main`** (commit `db8cd1b6`, the
last mainline commit before any child-workflow work landed), NOT from this
branch. Capturing them from the branch would bake in the never-shipped child
fields (`namespace_started`, `child_brokers`, per-row `invocation_namespace`,
etc.), making a load-compat test trivially green or wrongly red.

The recursive-broker refactor must load and replay these unchanged:

- `snapshot.json` — a `Context.to_dict()` (v2 `SerializedContext`) taken mid-run
  from a HITL workflow suspended on a `ctx.wait_for_event` waiter. Note it has
  **no** `child_brokers`/`namespace_started`/`active_invocation_namespaces` keys
  and the serialized waiter has **no** `invocation_namespace` field — the pre-
  child shape. New-code `from_dict_auto` + `from_serialized` must accept it.
- `snapshot_meta.json` — `{"expected_result_after_resume": ...}`: resuming the
  snapshot and delivering `HumanResponseEvent(response="42")` must yield this.
- `journal.json` — `{"result": 12, "ticks": [...]}`: a full tick journal for a
  fan-out + `collect_events` run. Replaying the ticks from a canonical
  `BrokerState.from_workflow` must reach `StopEvent(result=12)`.

Regenerate only from that same pre-child commit; see
`tests/test_prechild_golden_fixtures.py` for the workflow definitions the
fixtures were produced from.
