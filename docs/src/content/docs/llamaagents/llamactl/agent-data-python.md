---
title: Agent Data (Python)
sidebar:
  order: 21
---
:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::

See the [Agent Data Overview](/python/llamaagents/llamactl/agent-data-overview) for concepts, constraints, and environment details.

### Install

```bash
uv add llama-cloud-services
```

### Client overview

The Python `llama-cloud-services` SDK provides `AsyncAgentDataClient` for working with the Agent Data API.

```python
import httpx
import os
from pydantic import BaseModel
from llama_cloud_services.beta.agent_data import AsyncAgentDataClient
from llama_cloud.client import AsyncLlamaCloud

class ExtractedPerson(BaseModel):
    name: str
    age: int
    email: str

project_id = os.getenv("LLAMA_DEPLOY_PROJECT_ID")

# Base URL and API key (if running outside LlamaCloud)
base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Reusable async HTTP client with optional project scoping
http_client = httpx.AsyncClient(headers={"Project-Id": project_id} if project_id else None)

# Optional: base client for other SDK operations
base_client = AsyncLlamaCloud(
    base_url=base_url,
    token=api_key,
    httpx_client=http_client,
)

# Only set when deployed in LlamaCloud (falls back inside the Agent Data client)
deployment_name = os.getenv("LLAMA_DEPLOY_DEPLOYMENT_NAME")

client = AsyncAgentDataClient(
    type=ExtractedPerson,
    collection="extracted_people",
    # If omitted, uses LLAMA_DEPLOY_DEPLOYMENT_NAME or "_public"
    deployment_name=deployment_name,
    client=base_client,
)
```

### Create, Get, Update, Delete

```python
person = ExtractedPerson(name="John Doe", age=30, email="john@example.com")
created = await client.create_item(person)
fetched = await client.get_item(created.id)
updated = await client.update_item(created.id, ExtractedPerson(name="Jane", age=31, email="jane@example.com"))
await client.delete_item(updated.id)
```

Retry behavior: Network errors (timeouts, connection errors, retriable HTTP statuses) are retried up to 3 times with exponential backoff.

Notes:
- Updates overwrite the entire `data` object.
- `get_item` raises an `httpx.HTTPStatusError` with status code 404 if not found.

### Search

You can filter by `data` fields and by `created_at`/`updated_at` (top-level fields). Sort using a comma-delimited list of fields; the `data.` prefix is required when sorting by data fields. The default page size is 50 (max 1000).

```python
results = await client.search(
    filter={
        # Data fields
        "age": {"gte": 21, "lt": 65},
        "status": {"eq": "active"},
        "tags": {"includes": ["python", "ml"]},
        # Top-level timestamps (ISO strings accepted)
        "created_at": {"gte": "2024-01-01T00:00:00Z"},
    },
    order_by="data.name desc, created_at",
    page_size=50,
    offset=0,
    include_total=True,  # request only on the first page if needed
)

for item in results.items:
    print(item.data)

print(results.has_more, results.total)
```

Sorting:
- Example: `"data.name desc, created_at"`.
- If no sort is provided, results default to `created_at desc`.

Pagination:
- Use `offset` and `page_size`. The server may return `has_more` and a `next_page_token` (SDK exposes `has_more`).

### Aggregate

Group data by one or more `data` fields, optionally count items per group, and/or fetch the first item per group.

```python
agg = await client.aggregate(
    filter={"status": {"eq": "active"}},
    group_by=["department", "role"],
    count=True,
    first=True,  # return the earliest item per group (by created_at)
    order_by="data.department asc, data.role asc",
    page_size=100,
)

for group in agg.items:  # items are groups
    print(group.group_key)  # {"department": "Sales", "role": "AE"}
    print(group.count)      # optional
    print(group.first_item) # optional dict
```

Details:
- `group_by`: dot-style data paths (e.g., `"department"`, `"contact.email"`).
- `count`: adds a `count` per group.
- `first`: returns the first `data` item per group (earliest `created_at`).
- `order_by`: uses the same semantics as search (applies to group key expressions).
- Pagination uses `offset` and `page_size` similarly to search.
