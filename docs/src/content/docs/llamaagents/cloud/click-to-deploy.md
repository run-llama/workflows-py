---
title: Click-to-Deploy from LlamaCloud
sidebar:
  order: 3
---

:::caution
Cloud deployments of LlamaAgents are now in beta preview and broadly available for feedback. You can try them out locally or deploy to LlamaCloud and send us feedback with the in-app button.
:::

LlamaAgents allows you to deploy document workflow agents directly from the [LlamaCloud UI](https://cloud.llamaindex.ai/) with a single click. Choose a pre-built starter template, configure secrets, and deploy—no command line required.

## Get started with starter templates

From the LlamaCloud dashboard, navigate to **Agents** in your project. If you have no deployments, you'll see a "Jumpstart your Agent" section showing available starter templates.

Each starter template is a complete, working document workflow application that demonstrates how to combine LlamaCloud's document primitives—[Parse](/python/cloud/llamaparse/getting_started), [Extract](/python/cloud/llamaextract/getting_started), and [Classify](/python/cloud/llamaclassify/getting_started)—into a multi-step pipeline.

### Available starters

| Template | Description | Key Features |
|----------|-------------|--------------|
| **SEC Insights** | Classify financial PDFs and extract structured insights | Parse → Classify → Extract pipeline for SEC filings |
| **Invoice + Contract Matching** | Parse invoices and match with contracts, identifying discrepancies | Multi-document reconciliation workflow |

### Deploy a starter

1. Click on a starter template card to open the deployment dialog
2. Enter a **name** for your deployment (letters, numbers, and dashes only)
3. If the starter requires API keys (e.g., `OPENAI_API_KEY`), enter them in the **Required secrets** section
4. Click **Deploy**

LlamaCloud will clone the template repository, build your application, and deploy it. This typically takes 1–3 minutes. Once deployed, your agent will appear in the deployments list with its status.

### View your agent

Once deployment status shows **Running**, click **Visit** to open your agent's UI. Most starters include a web interface where you can:

- Upload documents for processing
- View extracted data and classifications
- Review and correct results (for extraction-review workflows)

Many starters also include sample data files. Click **Example Data** on the deployment card to download test documents.

## Customize your deployment

Starter templates are fully customizable. To modify the workflow logic, UI, or configuration:

### Fork and edit

1. Click **Customize** on the deployment card, or select **Edit** from the dropdown menu
2. Follow the link to **fork the repository on GitHub**
3. Make your changes in your forked repository
4. Update the deployment's **Repository URL** to point to your fork
5. Click **Update** to redeploy with your changes

### What you can customize

Every LlamaAgents deployment is a standard Python project with:

- **Workflows** (`src/`): LlamaIndex Workflow definitions using Parse, Extract, Classify, and other LlamaCloud services
- **UI** (`ui/`): React frontend using `@llamaindex/ui` hooks
- **Configuration** (`pyproject.toml`): Workflow registration, environment settings, and build configuration

For details on the project structure, see [Configuration Reference](/python/llamaagents/llamactl/configuration-reference).

## Manage deployments

After deploying, you can manage your agent from the LlamaCloud UI:

### Update to latest version

If you've pushed changes to your repository:
1. Click the **⋮** menu on the deployment card
2. Select **Update Version**
3. Confirm to pull and deploy the latest commit from your configured branch

### Rollback

If a deployment update causes issues:
1. Click the **⋮** menu and select **Rollback**
2. Choose a previous release from the history list
3. Click **Rollback** to restore that version

### Edit settings

To change the source repository or branch:
1. Select **Edit** from the deployment menu
2. Update the **Repository URL** or **Branch**
3. Click **Update**

:::note
Private repositories require installing the [LlamaAgents GitHub App](https://github.com/apps/llama-deploy). You'll be prompted to install it when configuring a private repository.
:::

### Delete

To remove a deployment:
1. Select **Delete** from the deployment menu
2. Confirm deletion

:::warning
Deleting a deployment is permanent. All associated resources and data will be removed.
:::

## Next steps

Ready to dive into the code? Learn how to author and configure workflows in [Serving your Workflows](/python/llamaagents/llamactl/workflow-api).
