---
title: Agent Data (JavaScript)
sidebar:
  order: 22
---

:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::

Agent Data is a JSON store tied to a `deploymentName` and `collection`. Use the official JavaScript SDK with strong typing for CRUD, search, and aggregation.

See the [Agent Data Overview](/python/llamaagents/llamactl/agent-data-overview) for concepts, constraints, and environment details.

Install:
```bash
npm i -S llama-cloud-services
```

Key imports:
```ts
import {
  AgentClient,
  createAgentDataClient,
  type TypedAgentData,
  type TypedAgentDataItems,
  type TypedAggregateGroupItems,
  type SearchAgentDataOptions,
  type AggregateAgentDataOptions,
} from "@llama-cloud-services/beta/agent";
```

### Create client

The helper infers the `deploymentName` from environment variables or the browser URL when possible, defaulting to `"_public"`.

```ts
type Person = { name: string; age: number; email: string };

const client = createAgentDataClient<Person>({
  // Optional: infer agent from env
  env: process.env as Record<string, string>,
  // Optional: infer from browser URL when not localhost
  windowUrl: typeof window !== "undefined" ? window.location.href : undefined,
  // Optional overrides
  // deploymentName: "person-extraction-agent",
  collection: "extracted_people",
});
```

Alternatively, construct a client directly:
```ts
const direct = new AgentClient<Person>({
  // client: default (from SDK) or a custom @hey-api/client-fetch instance
  deploymentName: "person-extraction-agent",
  collection: "extracted_people",
});
```

Browser usage:
- The TypeScript SDK works in the browser. When your app is deployed in LlamaCloud alongside your agent, requests are automatically authenticated.
- In other environments (local dev, custom hosting), provide an API key to the underlying client.
- To scope to a specific project, set `Project-Id` on the client’s headers.

### CRUD operations

```ts
// Create
const created = await client.createItem({ name: "John", age: 30, email: "john@example.com" });

// Get (returns null on 404)
const item = await client.getItem(created.id);

// Update (overwrites data)
const updated = await client.updateItem(created.id, { name: "Jane", age: 31, email: "jane@example.com" });

// Delete
await client.deleteItem(updated.id);
```

SDK responses are strongly typed and camel‑cased.
- `TypedAgentData<T>` fields: `id`, `deploymentName`, `collection?`, `data`, `createdAt`, `updatedAt`.

### Search

```ts
const options: SearchAgentDataOptions = {
  filter: {
    age: { gte: 21, lt: 65 },
    status: { eq: "active" },
    created_at: { gte: "2024-01-01T00:00:00Z" }, // top-level timestamp
  },
  orderBy: "data.name desc, created_at",
  pageSize: 50,
  offset: 0,
  includeTotal: true, // request on the first page only
};

const results: TypedAgentDataItems<Person> = await client.search(options);
for (const r of results.items) {
  console.log(r.data.name);
}
```

See the [Agent Data Overview](/python/llamaagents/llamactl/agent-data-overview#filter-dsl) for more details on filters.

- Filter keys target `data` fields, except `created_at`/`updated_at` which are top-level.
- Sort with comma-separated specs; prefix data fields in `orderBy` (e.g., `"data.name desc, created_at"`).
- Default `pageSize` is 50 (max 1000). Use `includeTotal` only on the first page.

Pagination: The default page size is 50 (max 1000). The response may include `nextPageToken` and `totalSize`.

### Aggregate

```ts
const aggOptions: AggregateAgentDataOptions = {
  filter: { status: { eq: "active" } },
  groupBy: ["department", "role"],
  count: true,
  first: true, // earliest by created_at per group
  orderBy: "data.department asc, data.role asc",
  pageSize: 100,
};

const groups: TypedAggregateGroupItems<Person> = await client.aggregate(aggOptions);
for (const g of groups.items) {
  console.log(g.groupKey, g.count, g.firstItem);
}
```
