---
"llama-agents-server": patch
---

fix(server): await background task cleanup in runtime `destroy()` so deferred-release and server-stop tasks cannot leak past shutdown
