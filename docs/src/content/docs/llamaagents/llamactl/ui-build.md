---
title: Configuring a UI
sidebar:
  order: 10
---
:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::

This page explains how to configure a custom frontend that builds and communicates with your LlamaAgents workflow server. If you've started from a template, you're good to go. Read on to learn more.

The LlamaAgents toolchain is unopinionated about your UI stack — bring your own UI. Most templates use Vite with React, but any framework will work that can:

- build static assets for production, and
- read a few environment variables during build and development

## How the integration works

`llamactl` starts and proxies your frontend during development by calling your `npm run dev` command. When you deploy, it builds your UI statically with `npm run build`. These commands are configurable; see [UIConfig](/python/llamaagents/llamactl/configuration-reference#uiconfig-fields) in the configuration reference. You can also use other package managers if you have [corepack](https://nodejs.org/download/release/v19.9.0/docs/api/corepack.html) enabled.

During development, `llamactl` starts its workflow server (port `4501` by default) and starts the UI, passing a `PORT` environment variable (set to `4502` by default) and a `LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH` (for example, `/deployments/<name>/ui`) where the UI will be served. It then proxies requests from the server to the client app from that base path.

Once deployed, the Kubernetes operator builds your application with the configured npm script (`build` by default) and serves your static assets at the same base path.

## Required configuration

1. Serve the dev UI on the configured `PORT`. This environment variable tells your dev server which port to use during development. Many frameworks, such as Next.js, read this automatically.
2. Set your app's base path to the value of `LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH`. LlamaAgents applications rely on this path to route to multiple workflow deployments. The proxy leaves this path intact so your application can link internally using absolute paths. Your development server and router need to be aware of this base path. Most frameworks provide a way to configure it. For example, Vite uses [`base`](https://vite.dev/config/shared-options.html#base).
3. Re-export the `LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH` env var to your application. Read this value (for example, in React Router) to configure a base path. This is also often necessary to link static assets correctly.
4. If you're integrating with LlamaCloud, re-export the `LLAMA_DEPLOY_PROJECT_ID` env var to your application and use it to scope your LlamaCloud requests to the same project. Read more in the [Configuration Reference](/python/llamaagents/llamactl/configuration-reference#authentication).
5. We also recommend re-exporting `LLAMA_DEPLOY_DEPLOYMENT_NAME`, which can be helpful for routing requests to your workflow server correctly.

## Examples

### Vite (React)

Configure `vite.config.ts` to read the injected environment and set the base path and port:

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const basePath = process.env.LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH;
  const port = process.env.PORT ? parseInt(process.env.PORT) : undefined;
  return {
    plugins: [react()],
    server: { port, host: true, hmr: { port } },
    base: basePath,
    // Pass-through env for client usage
    define: {
      ...(basePath && {
        "import.meta.env.VITE_LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH": JSON.stringify(basePath),
      }),
      ...(process.env.LLAMA_DEPLOY_DEPLOYMENT_NAME && {
        "import.meta.env.VITE_LLAMA_DEPLOY_DEPLOYMENT_NAME": JSON.stringify(
          process.env.LLAMA_DEPLOY_DEPLOYMENT_NAME,
        ),
      }),
      ...(process.env.LLAMA_DEPLOY_PROJECT_ID && {
        "import.meta.env.VITE_LLAMA_DEPLOY_PROJECT_ID": JSON.stringify(
          process.env.LLAMA_DEPLOY_PROJECT_ID,
        ),
      }),
    },
  };
});
```

Scripts in `package.json` typically look like:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  }
}
```

### Next.js (static export)

Next.js supports static export. Configure `next.config.mjs` to use the provided base path and enable static export:

```js
// next.config.mjs
const basePath = process.env.LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH || "";
const deploymentName = process.env.LLAMA_DEPLOY_DEPLOYMENT_NAME;
const projectId = process.env.LLAMA_DEPLOY_PROJECT_ID;

export default {
  // Mount app under /deployments/<name>/ui
  basePath,
  // For assets when hosted behind a path prefix
  assetPrefix: basePath || undefined,
  // Enable static export for production
  output: "export",
  // Expose base path to browser for runtime URL construction
  env: {
    NEXT_PUBLIC_LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH: basePath,
    NEXT_PUBLIC_LLAMA_DEPLOY_DEPLOYMENT_NAME: deploymentName,
    NEXT_PUBLIC_LLAMA_DEPLOY_PROJECT_ID: projectId,
  },
};
```

Ensure your scripts export to a directory (default: `out/`):

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build && next export"
  }
}
```

The dev server binds to the `PORT` the app server sets; no additional configuration is needed. For dynamic routes or server features not compatible with static export, you can omit the export and rely on proxying to the Python app server. However, production static hosting requires a build output directory.

#### Runtime URL construction (images/assets)

- Vite: use the configured `base` or `import.meta.env.BASE_URL` (or the pass-through variable) to prefix asset URLs you build at runtime:

```tsx
const base = import.meta.env.VITE_LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH || import.meta.env.BASE_URL || "/";
<img src={`${base.replace(/\/$/, "")}/images/logo.png`} />
```

- Next.js static export: use the exposed `NEXT_PUBLIC_LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH` so routes resolve absolute asset paths correctly:

```tsx
const base = process.env.NEXT_PUBLIC_LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH || "";
export default function Logo() {
  return <img src={`${base}/images/logo.png`} alt="logo" />;
}
```

## Configure the UI output directory

Your UI must output static assets that the platform can locate. Configure `ui.directory` and `ui.build_output_dir` as described in the [Deployment Config Reference](/python/llamaagents/llamactl/configuration-reference#uiconfig-fields). Default: `${ui.directory}/dist`.
