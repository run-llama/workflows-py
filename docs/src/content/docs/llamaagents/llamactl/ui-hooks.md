---
title: Workflow React Hooks
sidebar:
  order: 15
---

:::caution
Cloud deployments of LlamaAgents are now in beta preview and broadly available for feedback. You can try them out locally or deploy to LlamaCloud and send us feedback with the in-app button.
:::

Our React library, `@llamaindex/ui`, is the recommended way to integrate your UI with a LlamaAgents workflow server and LlamaCloud. It comes pre-installed in any of our templates containing a UI. The library provides both React hooks for custom integrations and standard components.

### Workflows Hooks

Our React hooks provide an idiomatic way to observe and interact with your LlamaAgents workflows remotely from a frontend client.

There are 4 hooks you can use:
1. **useWorkflow**: Get actions for a specific workflow (create handlers, run to completion).
2. **useHandler**: Get state and actions for a single handler (stream events, send events).
3. **useHandlers**: List and monitor handlers with optional filtering.
4. **useWorkflows**: List all available workflows.

### Client setup

Configure the hooks with a workflow client. Wrap your app with an `ApiProvider` that points to your deployment:

```tsx
import { ApiProvider, type ApiClients, createWorkflowsClient } from "@llamaindex/ui";

const deploymentName =
  (import.meta as any).env?.VITE_LLAMA_DEPLOY_DEPLOYMENT_NAME || "default";

const clients: ApiClients = {
  workflowsClient: createWorkflowsClient({
    baseUrl: `/deployments/${deploymentName}`,
  }),
};

export function Providers({ children }: { children: React.ReactNode }) {
  return <ApiProvider clients={clients}>{children}</ApiProvider>;
}
```

### List available workflows

Use `useWorkflows` to list all workflows available in the deployment:

```tsx
import { useWorkflows } from "@llamaindex/ui";

export function WorkflowList() {
  const { state, sync } = useWorkflows();

  if (state.loading) return <div>Loading…</div>;

  return (
    <div>
      <button onClick={() => sync()}>Refresh</button>
      <ul>
        {Object.values(state.workflows).map((w) => (
          <li key={w.name}>{w.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

### Start a run

Start a workflow by name with `useWorkflow`. Call `createHandler` with a JSON input payload to get back a handler state immediately.

```tsx
import { useState } from "react";
import { useWorkflow } from "@llamaindex/ui";

export function CreateHandler() {
  const workflow = useWorkflow("stream");
  const [handlerId, setHandlerId] = useState<string | null>(null);

  async function handleClick() {
    const handlerState = await workflow.createHandler({});
    setHandlerId(handlerState.handler_id);
  }

  return (
    <div>
      <button onClick={handleClick}>Create Handler</button>
      {handlerId && <div>Created: {handlerId}</div>}
    </div>
  );
}
```

### Watch a run and stream events

Subscribe to a handler's live event stream using `subscribeToEvents`:

```tsx
import { useEffect, useState } from "react";
import { useWorkflow, useHandler, WorkflowEvent, isStopEvent } from "@llamaindex/ui";

export function StreamEvents() {
  const workflow = useWorkflow("stream");
  const [handlerId, setHandlerId] = useState<string | null>(null);
  const handler = useHandler(handlerId);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);

  async function start() {
    setEvents([]);
    const h = await workflow.createHandler({});
    setHandlerId(h.handler_id);
  }

  useEffect(() => {
    if (!handlerId) return;
    const sub = handler.subscribeToEvents({
      onData: (event) => setEvents((prev) => [...prev, event]),
    });
    return () => sub.unsubscribe();
  }, [handlerId]);

  const stop = events.find(isStopEvent);

  return (
    <div>
      <button onClick={start}>Start & Stream</button>
      {handlerId && <div>Status: {handler.state.status}</div>}
      {stop && <pre>{JSON.stringify(stop.data, null, 2)}</pre>}
      {!stop && events.length > 0 && <div>{events.length} events received</div>}
    </div>
  );
}
```

### Monitor multiple workflow runs

Use `useHandlers` to query and monitor a filtered list of workflow handlers. This is useful for progress indicators or "Recent runs" views.

```tsx
import { useHandlers } from "@llamaindex/ui";

export function RecentRuns() {
  const { state, sync } = useHandlers({
    query: { status: ["running", "completed"] },
  });

  if (state.loading) return <div>Loading…</div>;

  const handlers = Object.values(state.handlers);

  return (
    <div>
      <button onClick={() => sync()}>Refresh</button>
      <ul>
        {handlers.map((h) => (
          <li key={h.handler_id}>
            {h.handler_id.slice(0, 8)}… — {h.status}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

The `sync` option controls whether to fetch handlers on mount. Call `sync()` manually to refresh the list from the server at any time.

### Hook Reference

| Hook | Purpose | Key Methods/Properties |
|------|---------|----------------------|
| `useWorkflow(name)` | Work with a specific workflow | `createHandler(input)`, `runToCompletion(input)`, `state.graph` |
| `useHandler(handlerId)` | Work with a specific handler | `sendEvent(event)`, `subscribeToEvents(callbacks)`, `sync()`, `state.status`, `state.result` |
| `useHandlers({ query, sync })` | List/filter handlers | `sync()`, `setHandler(h)`, `actions(id)`, `state.handlers` |
| `useWorkflows({ sync })` | List all workflows | `sync()`, `state.workflows` |
