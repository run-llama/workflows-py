---
title: Agent Data
sidebar:
  order: 30
---

:::caution
Cloud deployments of LlamaAgents are now in beta preview and broadly available for feedback. You can try them out locally or deploy to LlamaCloud and send us feedback with the in-app button.
:::

### What is Agent Data?

Skip the database setup. LlamaAgents workflows and JavaScript UIs share a persistent Agent Data store built into the LlamaCloud API. It uses the same authentication as the rest of the API.

Agent Data is a queryable store for JSON records produced by your agents. Each record is linked to a `deployment_name` (the deployed agent) and an optional `collection` (a logical bucket; defaults to `default`). Use it to persist extractions, events, metrics, and other structured output, then search and aggregate across records.

Key concepts:
- **deployment_name**: the identifier of the agent deployment the data belongs to. Access is authorized against that agent's project.
- **collection**: a logical namespace within an agent for organizing different data types or apps. Storage is JSON. We recommend storing homogeneous data types within a single collection.
- **data**: the JSON payload shaped by your app. SDKs provide typed wrappers.

Important behavior and constraints:
- **Deployment required**: The `deployment_name` must correspond to an existing deployment. Data is associated with that deployment and its project.
- **Local development**: When running locally, omit `deployment_name` to use the shared `_public` Agent Data store. Use distinct `collection` names to separate apps during local development.
- **Access control**: You can only read/write data for agents in projects you can access. `_public` data is visible across agents within the same project.

### SDK Reference

For CRUD operations, search, filtering, sorting, aggregation, and deletion, see the generated SDK reference:

**[Agent Data API Reference](https://developers.llamaindex.ai/reference/resources/beta/subresources/agent_data/)**

The reference covers all available operations:
- **Create / Get / Update / Delete** individual records
- **Search** with filtering, sorting, and pagination
- **Aggregate** by grouping fields with counts and first-item retrieval
- **Delete by query** for bulk deletion using the filter DSL

SDK packages:
- **Python**: `llama-cloud-services` (`llama_cloud_services.beta.agent_data.AsyncAgentDataClient`)
- **JavaScript**: `llama-cloud-services` (`@llama-cloud-services/beta/agent`)

### ExtractedData wrapper

`ExtractedData` is a specialized wrapper type available in the Python SDK (`llama-cloud-services`) and the JavaScript UI library (`@llamaindex/ui`). It is not part of the generated API reference, so it is documented here.

`ExtractedData[T]` is designed for extraction workflows where data goes through review and approval stages. Use it as the type parameter for your Agent Data client.

**Fields:**

| Field | Description |
|-------|-------------|
| `original_data` | The data as originally extracted (preserved for change tracking) |
| `data` | The current state of the data (updated by human review) |
| `status` | Workflow status: `pending_review`, `accepted`, `rejected`, `error`, or custom string |
| `overall_confidence` | Aggregated confidence score (auto-calculated from field_metadata) |
| `field_metadata` | Dict mapping field paths to metadata including confidence scores and citations |
| `file_id` | LlamaCloud file ID of the source document |
| `file_name` | Name of the source file |
| `file_hash` | Content hash for deduplication |
| `metadata` | Additional application-specific metadata |

**Python usage:**

```python
from pydantic import BaseModel
from llama_cloud_services.beta.agent_data import AsyncAgentDataClient, ExtractedData

class Invoice(BaseModel):
    vendor: str | None = None
    total: float | None = None
    date: str | None = None

# Client stores ExtractedData[Invoice] records
client = AsyncAgentDataClient(
    type=ExtractedData[Invoice],
    collection="invoices",
    deployment_name=deployment_name,
    client=base_client,
)
```

**Creating from LlamaExtract results:**

The `from_extraction_result` factory method creates an `ExtractedData` instance directly from a LlamaExtract result, automatically capturing field metadata (confidence scores, citations):

```python
from llama_cloud_services import LlamaExtract
from llama_cloud_services.beta.agent_data import ExtractedData

extractor = LlamaExtract()
result = await extractor.aextract(data_schema=Invoice, files="invoice.pdf")

extracted = ExtractedData.from_extraction_result(
    result=result,
    schema=Invoice,
    status="pending_review",  # optional, defaults to "pending_review"
)

await client.create_item(extracted)
```

**Creating manually:**

Use `ExtractedData.create` when constructing extracted data from other sources or transforming to a different schema:

```python
from llama_cloud_services.beta.agent_data import ExtractedData

invoice = Invoice(vendor="Acme Corp", total=1500.00, date="2024-01-15")

extracted = ExtractedData.create(
    data=invoice,
    status="pending_review",
    file_id="file-abc123",
    file_name="invoice.pdf",
    file_hash="sha256:...",
    field_metadata={
        "vendor": {"confidence": 0.95, "citation": [{"page": 1, "matching_text": "Acme Corp"}]},
        "total": {"confidence": 0.92},
    },
)
```

**JavaScript usage:**

In `@llamaindex/ui`, `ExtractedData` is available as a TypeScript type:

```ts
import { type ExtractedData, StatusType } from "@llama-cloud-services/beta/agent";
```

Use it as the type parameter when creating an Agent Data client to get full type safety for extraction workflows.
