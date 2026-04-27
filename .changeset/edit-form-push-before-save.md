---
"llamactl": patch
---

Editing a push-mode (Local repo) deployment now pushes local code before calling update, so switching branches or saving new commits works on the first try and the server resolves git_ref to the actual latest SHA.
