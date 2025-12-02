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

### Start a run

Start a workflow by name with `useWorkflow`. Call `createHandler` with a JSON input payload to get back a handler state immediately, then use `useHandler` to stream events.

```tsx
import { useState } from "react";
import { useWorkflow, useHandler } from "@llamaindex/ui";

export function RunButton() {
  const workflow = useWorkflow("my_workflow");
  const [handlerId, setHandlerId] = useState<string | null>(null);
  const handler = useHandler(handlerId);

  async function handleClick() {
    const handlerState = await workflow.createHandler({ user_id: "123" });
    // handlerState contains handler_id, status, etc.
    setHandlerId(handlerState.handler_id);
  }

  return (
    <>
      <button onClick={handleClick}>Run Workflow</button>
      {handlerId && <HandlerDetails handlerId={handlerId} />}
    </>
  );
}
```

### Run to completion

For simple request/response workflows, use `runToCompletion` to run the workflow and wait for the final result:

```tsx
import { useWorkflow } from "@llamaindex/ui";
import { useEffect, useState } from "react";

export function SimpleQuery() {
  const workflow = useWorkflow("metadata");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    workflow.runToCompletion({ query: "What is the answer?" })
      .then((handlerState) => {
        if (handlerState.status === "completed") {
          setResult(handlerState.result?.data);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading...</div>;
  return <pre>{JSON.stringify(result, null, 2)}</pre>;
}
```

### Watch a run and stream events

Subscribe to a handler's live event stream using `subscribeToEvents`:

```tsx
import { useEffect, useState } from "react";
import { useHandler, WorkflowEvent, isStopEvent } from "@llamaindex/ui";

export function HandlerDetails({ handlerId }: { handlerId: string }) {
  const handler = useHandler(handlerId);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);

  useEffect(() => {
    if (!handlerId) return;

    const subscription = handler.subscribeToEvents({
      onData: (event) => {
        setEvents((prev) => [...prev, event]);
      },
      onSuccess: (allEvents) => {
        console.log("Workflow completed with", allEvents.length, "events");
      },
      onError: (error) => {
        console.error("Workflow failed:", error);
      },
    });

    return () => subscription.unsubscribe();
  }, [handlerId]);

  // Find the final StopEvent to extract the workflow result
  const stop = events.find(isStopEvent);

  return (
    <div>
      <div>
        <strong>{handler.state.handler_id}</strong> — {handler.state.status}
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

### Send events to a handler

Use `sendEvent` to send events back to the workflow, enabling human-in-the-loop patterns:

```tsx
import { useHandler } from "@llamaindex/ui";

export function SendEventExample({ handlerId }: { handlerId: string }) {
  const handler = useHandler(handlerId);

  const sendMove = (direction: string) => {
    handler.sendEvent({
      type: "PlayerMoveEvent",
      value: { direction },
    } as any);
  };

  return (
    <div>
      <button onClick={() => sendMove("north")}>Go North</button>
      <button onClick={() => sendMove("south")}>Go South</button>
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
    query: { workflow_name: ["my_workflow"], status: ["running", "completed"] },
    sync: true, // auto-fetch on mount (default)
  });

  if (state.loading) return <div>Loading…</div>;
  if (state.loadingError) return <div>Error: {state.loadingError}</div>;

  const handlers = Object.values(state.handlers);

  return (
    <div>
      <button onClick={() => sync()}>Refresh</button>
      <ul>
        {handlers.map((h) => (
          <li key={h.handler_id}>
            {h.handler_id} — {h.status}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

The `sync` option controls whether to fetch handlers on mount. Call `sync()` manually to refresh the list from the server at any time.

### Track running workflows with actions

`useHandlers` also provides actions for working with handlers in the list, including subscribing to their events:

```tsx
import { useHandlers, WorkflowEvent } from "@llamaindex/ui";
import { useEffect, useRef } from "react";

export function WorkflowProgress({ workflowName }: { workflowName: string }) {
  const handlersService = useHandlers({
    query: { workflow_name: [workflowName], status: ["running"] },
  });
  const subscribed = useRef<Set<string>>(new Set());

  const runningHandlers = Object.values(handlersService.state.handlers).filter(
    (h) => h.status === "running"
  );

  // Subscribe to events for each running handler
  useEffect(() => {
    for (const handler of runningHandlers) {
      if (!subscribed.current.has(handler.handler_id)) {
        subscribed.current.add(handler.handler_id);
        handlersService.actions(handler.handler_id).subscribeToEvents({
          onData(event) {
            console.log("Event:", event.type, event.data);
          },
          onComplete() {
            subscribed.current.delete(handler.handler_id);
          },
        });
      }
    }
  }, [runningHandlers.map((h) => h.handler_id).join(",")]);

  return (
    <div>
      {runningHandlers.length} running workflow{runningHandlers.length === 1 ? "" : "s"}
    </div>
  );
}
```

### List available workflows

Use `useWorkflows` to list all workflows available in the deployment:

```tsx
import { useWorkflows } from "@llamaindex/ui";

export function WorkflowList() {
  const { state } = useWorkflows();

  if (state.loading) return <div>Loading…</div>;

  return (
    <ul>
      {Object.values(state.workflows).map((w) => (
        <li key={w.name}>{w.name}</li>
      ))}
    </ul>
  );
}
```

### Hook Reference

| Hook | Purpose | Key Methods/Properties |
|------|---------|----------------------|
| `useWorkflow(name)` | Work with a specific workflow | `createHandler(input)`, `runToCompletion(input)`, `state.graph` |
| `useHandler(handlerId)` | Work with a specific handler | `sendEvent(event)`, `subscribeToEvents(callbacks)`, `sync()`, `state.status`, `state.result` |
| `useHandlers({ query, sync })` | List/filter handlers | `sync()`, `setHandler(h)`, `actions(id)`, `state.handlers` |
| `useWorkflows({ sync })` | List all workflows | `sync()`, `state.workflows` |
