---
title: Workflow React Hooks
sidebar:
  order: 15
---

:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::

Our React library, `@llamaindex/ui`, is the recommended way to integrate your UI with a LlamaAgents workflow server and LlamaCloud. It comes pre-installed in any of our templates containing a UI. The library provides both React hooks for custom integrations and standard components.

### Workflows Hooks

Our React hooks provide an idiomatic way to observe and interact with your LlamaAgents workflows remotely from a frontend client.

There are 3 hooks you can use:
1. **useWorkflowRun**: Start a workflow run and observe its status.
2. **useWorkflowHandler**: Observe and interact with a single run; stream and send events.
3. **useWorkflowHandlerList**: Monitor and update a list of recent or in-progress runs.

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

### Start a run

Start a workflow by name with `useWorkflowRun`. Provide a JSON input payload. You get a `handler_id` back immediately.

```tsx
import { useState } from "react";
import { useWorkflowRun } from "@llamaindex/ui";

export function RunButton() {
  const { runWorkflow, isCreating, error } = useWorkflowRun();
  const [handlerId, setHandlerId] = useState<string | null>(null);

  async function handleClick() {
    const handler = await runWorkflow("my_workflow", { user_id: "123" });
    // e.g., navigate to a details page using handler.handler_id
    console.log("Started:", handler.handler_id);
    setHandlerId(handler.handler_id);
  }

  return (
    <>
      <button disabled={isCreating} onClick={handleClick}>
        {isCreating ? "Starting…" : "Run Workflow"}
      </button>
      {/* Then, use the handler ID to show details or send events */}
      <HandlerDetails handlerId={handlerId} />
    </>
  );
}
```

### Watch a run and stream events

Subscribe to a single handler’s live event stream and show status with `useWorkflowHandler`.

```tsx
import { useWorkflowHandler } from "@llamaindex/ui";

export function HandlerDetails({ handlerId }: { handlerId: string | null }) {
  // Note, the state will remain empty if the handler ID is empty
  const { handler, events, sendEvent } = useWorkflowHandler(handlerId ?? "", true);

  // Find the final StopEvent to extract the workflow result (if provided)
  const stop = events.find(
    (e) =>
      e.type.endsWith(
        ".StopEvent"
      ) /* event type contains the event's full Python module path, e.g., workflows.events.StopEvent */
  );

  return (
    <div>
      <div>
        <strong>{handler.handler_id}</strong> — {handler.status}
      </div>
      {stop ? (
        <pre>{JSON.stringify(stop.data, null, 2)}</pre>
      ) : (
        <pre style={{ maxHeight: 240, overflow: "auto" }}>
          {JSON.stringify(events, null, 2)}
        </pre>
      )}
    </div>
  );
}
```

You can subscribe to the same handler with multiple hooks and access a shared events list. This is useful when, for example, one component shows toast messages for certain event types while another component shows the final result.

### Monitor multiple workflow runs

Subscribe to the full list or a filtered list of workflow runs with `useWorkflowHandlerList`. This is useful for a progress indicator or a lightweight “Recent runs” view.

```tsx
import { useWorkflowHandlerList } from "@llamaindex/ui";

export function RecentRuns() {
  const { handlers, loading, error } = useWorkflowHandlerList();
  if (loading) return <div>Loading…</div>;
  if (error) return <div>Error: {error}</div>;
  return (
    <ul>
      {handlers.map((h) => (
        <li key={h.handler_id}>{h.handler_id} — {h.status}</li>
      ))}
    </ul>
  );
}
```
