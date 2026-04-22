---
"llama-agents-appserver": patch
---

Fix appserver install when the target template is a uv workspace member. Install now targets whichever venv `uv run` resolves to, instead of a hard-coded `<template>/.venv`, so `llamactl dev validate` and `llamactl serve` work in workspace layouts.
