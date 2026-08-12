---
"llama-index-workflows": patch
---

Fix an off-by-one that made retry wait strategies skip their first configured delay, so every retry delay is now one step shorter than before.
