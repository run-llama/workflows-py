---
"llama-index-workflows": patch
---

Reloading a persisted workflow failure no longer crashes when the original exception cannot be faithfully rebuilt; it degrades to an UnreconstructedException carrying the original type name and message, and exception type imports now honor the serializer's allowed_types allowlist.
