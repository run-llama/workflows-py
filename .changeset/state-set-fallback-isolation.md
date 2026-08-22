---
"llama-index-workflows": patch
---

Fix ctx.store.set writing through to committed state when the path runs through a container it cannot rebuild. A dataclass, tuple, or plain object on the write path made the write visible to lockless readers before it committed; those paths now copy state first.
