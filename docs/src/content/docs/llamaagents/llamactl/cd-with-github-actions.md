---
title: Continuous Deployment with GitHub Actions
sidebar:
  order: 19
---
:::caution
LlamaAgents is currently in alpha. You can try it out locally or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs).
:::

As described in the [Getting Started](/python/cloud/llamaagents/getting-started) guide, deploying a LlamaAgent to LlamaCloud requires connecting it to a `git` source (specifically, GitHub).

However, when you push changes to GitHub, they are not automatically applied to your LlamaAgent. To keep your deployment in sync with your `git` reference, you must manually run `llamactl deployments update`.

This manual approach provides fine-grained control over when changes to your production branch are deployed, but it may not fit every workflow.

If you prefer to automate the process using continuous deployment, you can use the `run-llama/llamactl-deploy` GitHub Action.

To use this action, you’ll need:

- A LlamaCloud API key
- The Project ID associated with your LlamaAgent deployment
- The ID of the deployed LlamaAgent

Optionally, you can specify a `git` reference for the deployment (such as a commit SHA or branch name).

Here’s an example workflow configuration:

```yaml
name: Update LlamaAgent Deployment

on:
  push:
    branches:
      - main

jobs:
  update-deployment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Update deployment
        uses: run-llama/llamactl-deploy@v0.1.0
        with:
          llama-cloud-api-key: ${{ secrets.LLAMA_CLOUD_API_KEY }}
          llama-cloud-project-id: ${{ secrets.LLAMA_CLOUD_PROJECT_ID }}
          deployment-id: "your-deployment-id"
          git-reference: ${{ github.sha }} # optional
```

You can find the reference for this GitHub Action in the [dedicated repository](https://github.com/run-llama/llamactl-deploy).
