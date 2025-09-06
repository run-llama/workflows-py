# @llamaindex/workflows-client

TypeScript client for LlamaIndex Workflows server.

## Installation

```bash
pnpm add @llamaindex/workflows-client
# or
npm install @llamaindex/workflows-client
```

## Usage

```typescript
import { client } from '@llamaindex/workflows-client';

// Initialize the client
client.setConfig({
  baseUrl: 'http://localhost:8000'
});

// List available workflows
const { data: workflows } = await client.GET('/workflows');

// Run a workflow synchronously
const { data: result } = await client.POST('/workflows/{name}/run', {
  params: {
    path: { name: 'my-workflow' }
  },
  body: {
    context: {
      // Your context data
    },
    kwargs: {
      // Additional arguments
    }
  }
});

// Run a workflow asynchronously
const { data: async_result } = await client.POST('/workflows/{name}/run-nowait', {
  params: {
    path: { name: 'my-workflow' }
  },
  body: {
    // Same as above
  }
});

// Get result later
const { data: final_result } = await client.GET('/results/{handler_id}', {
  params: {
    path: { handler_id: async_result.handler_id }
  }
});

// Stream events
const { data: events } = await client.GET('/events/{handler_id}', {
  params: {
    path: { handler_id: async_result.handler_id },
    query: { sse: true }
  }
});
```

## Development

This client is auto-generated from the OpenAPI schema. To regenerate:

```bash
# From the root of the workflows-py repository
python scripts/generate_sdk.py

# Or using pnpm scripts from this directory
pnpm run generate
pnpm run build
```

## License

MIT
