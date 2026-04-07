# LlamaIndex Agent Instructions

## Using LlamaIndex APIs

This project uses LlamaIndex. Use the MCP server for documentation lookup
when developing agent apps or working with LlamaIndex APIs:

**MCP endpoint:** `https://developers.llamaindex.ai/mcp`

The server provides these tools:
- **search_docs** — lexical search over LlamaIndex documentation
- **grep_docs** — exact pattern matching with regex
- **read_doc** — retrieve full page contents by path

Use these tools to look up API usage, workflow patterns, and integration
guides before writing or modifying code.

## Fallback

If MCP is unavailable, fetch documentation from:
`https://developers.llamaindex.ai/llms.txt`
