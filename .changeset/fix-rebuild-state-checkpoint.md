---
"llama-index-workflows": patch
---

fix: rebuild_state_from_ticks clears in_progress before replaying

Fixed ctx.to_dict() failing with "Worker X not found in in_progress" when checkpointing resumed workflows. The function now clears in_progress and moves events to the queue before replaying ticks, matching what rewind_in_progress() does at runtime.
