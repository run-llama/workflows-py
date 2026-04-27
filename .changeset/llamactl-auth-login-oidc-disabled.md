---
"llamactl": patch
---

`llamactl auth login` now prints a friendly hint pointing to `llamactl auth token` when the server has no OIDC browser-login configured, instead of dumping a raw 400 from the discovery endpoint.
