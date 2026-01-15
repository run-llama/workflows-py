---
"llama-index-workflows": patch
---

fix: rebuild_state_from_ticks clears in_progress before replaying

Fixed ctx.to_dict() failing with "Worker X not found in in_progress" when checkpointing resumed workflows. The function now also rewinds in progress when recreating from ticks, to match the actual behavior when resuming a workflow.
