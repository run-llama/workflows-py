---
title: Agent Data
sidebar:
  order: 20
---

:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::

### What is Agent Data?

Skip the database setup. LlamaAgents workflows and JavaScript UIs share a persistent Agent Data store built into the LlamaCloud API. It uses the same authentication as the rest of the API.

Agent Data is a queryable store for JSON records produced by your agents. Each record is linked to a `deployment_name` (the deployed agent) and an optional `collection` (a logical bucket; defaults to `default`). Use it to persist extractions, events, metrics, and other structured output, then search and aggregate across records.

Key concepts:
- **deployment_name**: the identifier of the agent deployment the data belongs to. Access is authorized against that agent’s project.
- **collection**: a logical namespace within an agent for organizing different data types or apps. Storage is JSON. We recommend storing homogeneous data types within a single collection.
- **data**: the JSON payload shaped by your app. SDKs provide typed wrappers.

Important behavior and constraints:
- **Deployment required**: The `deployment_name` must correspond to an existing deployment. Data is associated with that deployment and its project.
- **Local development**: When running locally, omit `deployment_name` to use the shared `_public` Agent Data store. Use distinct `collection` names to separate apps during local development.
- **Access control**: You can only read/write data for agents in projects you can access. `_public` data is visible across agents within the same project.
- **Filtering/Sorting**: You can filter on any `data` fields and on the top‑level `created_at` and `updated_at`. Sorting accepts a comma‑separated list; prefix fields inside `data` with `data.` (for example, `data.name desc, created_at`).
- **Aggregation**: Group by one or more data fields and optionally return per‑group counts and/or the first item.

Project scoping:
- You can scope requests to a specific project by providing the `Project-Id` header (UUID). This is especially important if your API key has access to multiple projects. Read more in the [Configuration Reference](/python/cloud/llamaagents/configuration-reference#authorization).

### Filter DSL

When searching or aggregating, you can filter on fields inside `data` and on the top‑level `created_at` and `updated_at` fields.

Example:

```json
{
  "age": {"gte": 21, "lt": 65},
  "status": {"eq": "active"},
  "tag": {"includes": ["python", "ml"]},
  "created_at": {"gte": "2024-01-01T00:00:00Z"}
}
```

Supported operators:

Filter operators are specified using a simple JSON DSL and support the following per‑field operators:
- `eq` - Filters based on equality. For example, `{"age": {"eq": 30}}` matches age exactly 30.
- `gt` - Filters based on greater than. For example, `{"age": {"gt": 30}}` matches age greater than 30.
- `gte` - Filters based on greater than or equal to. For example, `{"age": {"gte": 30}}` matches age 30 or greater.
- `lt` - Filters based on less than. For example, `{"age": {"lt": 30}}` matches age less than 30.
- `lte` - Filters based on less than or equal to. For example, `{"age": {"lte": 30}}` matches age less than or equal to 30.
- `includes` - Filters based on inclusion. For example, `{"age": {"includes": [30, 31]}}` matches age containing 30 or 31. An empty array matches nothing.

All provided filters must match (logical AND).

Nested fields are addressed using dot notation. For example, `{"data.age": {"gt": 30}}` matches an age greater than 30 in the `data` object. Note: array index access is not supported.

SDKs and environments:
- The **JavaScript SDK** can be used in the browser. When your UI is deployed on LlamaCloud alongside your agent, it is automatically authenticated. In other environments, provide an API key. You can also set `Project-Id` on the underlying HTTP client to pin all requests to a project.
- The **Python SDK** runs server‑side and uses your API key and an optional base URL.

Next steps:
- Python usage: see [Agent Data (Python)](/python/llamaagents/llamactl/agent-data-python)
- JavaScript usage: see [Agent Data (JavaScript)](/python/llamaagents/llamactl/agent-data-javascript)
