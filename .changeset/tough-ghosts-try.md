---
"llama-index-workflows": minor
---

Add support for injecting resources more flexibly

- Add support for injecting Resources recursively, so a Resource can depend on another Resource or ResourceConfig
- Add support for injecting ResourceConfig directly into steps
- Fix issues with resolving from String quoted types
