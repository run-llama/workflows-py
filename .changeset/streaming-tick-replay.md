---
"llama-index-workflows": patch
"llama-agents-server": patch
"llama-agents-dbos": patch
"llama-agents-agentcore": patch
---

Stream ticks during resume so peak memory is bounded by batch size rather than total tick history. Adds `rebuild_state_from_ticks_stream`, `AbstractWorkflowStore.query_ticks`/`stream_ticks`, and switches idle-release and persistence-runtime replay to the streaming path.
