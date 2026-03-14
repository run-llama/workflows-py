# llama-agents-dbos

## 0.1.2

### Patch Changes

- 5e7f9e5: Add event input/output summaries to step spans and rehydrate span context across serialization boundaries. Log instead of fail cancelled steps from cancelled workflows. Do not fail from wait_for_event exceptions.

## 0.1.1

### Patch Changes

- 6605457: Bump dependency requirements
- 6ec262c: Fix graceful teardown leading to poisoned DBOS workflow

## 0.1.0

### Minor Changes

- d56be47: Add postgres and DBOS support to the workflow server
- 57902d5: Add alternate DBOS runtime plugin for running workflows against a DBOS backend

### Patch Changes

- 77a3f9c: Add workflow release for idle DBOS workflows (with replica support)
- 96e437e: Move task execution into the runtime, for maximal control of specific runtime semantics around determinism

## 0.1.0-rc.1

### Patch Changes

- 3720c61: Add workflow release for idle DBOS workflows (with replica support)
- a2aad32: Move task execution into the runtime, for maximal control of specific runtime semantics around determinism

## 0.1.0-rc.0

### Minor Changes

- c2e7f17: Add postgres and DBOS support to the workflow server
- 79159f0: Add alternate DBOS runtime plugin for running workflows against a DBOS backend
