---
title: Agent Builder
sidebar:
  order: 10
---

Agent Builder is a natural language interface for creating document workflows in LlamaCloud. Describe what you want to extract from your documents in plain English, watch as an AI coding agent generates a complete workflow, and deploy it with a single click.

<!-- Screenshot placeholder: Agent Builder landing page with chat interface -->

## How It Works

Agent Builder uses an AI coding agent to transform your descriptions into working document pipelines:

1. **Describe** your extraction needs in plain English
2. **Watch** as the AI generates your workflow in real-time
3. **Review** the visual workflow graph showing each step
4. **Deploy** to LlamaCloud with one click

### The Workflow Visualization

As the agent builds your workflow, you'll see a visual graph that shows exactly what your pipeline does. Each node represents a step—parsing documents, classifying them, extracting data, or validating results.

<!-- Screenshot placeholder: Workflow visualization sidebar showing a multi-step pipeline with nodes for parse, classify, and extract -->

This visualization is the core output of Agent Builder. It makes the abstract concrete: instead of reading code, you see your pipeline as a flowchart you can understand at a glance.

The agent understands LlamaCloud services and their configuration. You can ask it to adjust extraction schemas, change classification rules, or modify any aspect of your workflow through conversation.

## Building Your First Workflow

### Start a New Session

From the LlamaCloud dashboard, navigate to **Agents** and click **Start Building**. This opens Agent Builder's chat interface where you'll describe your workflow.

### Describe What You Want

Tell the agent what documents you have and what data you want to extract. Be specific about your needs:

> "I have SEC 10-K filings and want to extract revenue, net income, and risk factors"

> "Extract line items, totals, and vendor information from invoices"

> "Given court documents, classify them as complaints, motions, or orders, then extract different fields based on type"

The agent will ask clarifying questions if it needs more detail about your requirements.

### Watch Your Workflow Take Shape

As the agent works, you'll see:

- **Real-time activity** showing files being created and modified
- **The workflow visualization** updating as steps are added
- **Explanations** of what's being built and why

The visualization shows your complete pipeline—you can see how documents flow through parsing, classification, extraction, and any validation steps.

### Review and Iterate

Once generation completes, review the workflow visualization. You can continue chatting to refine it:

- Adjust the extraction schema to capture additional fields
- Add validation rules to check extracted data
- Change how documents are classified
- Modify any aspect of the workflow

For details on extraction schemas and configuration, see [LlamaExtract documentation](/python/cloud/llamaextract/getting_started).

## Deploying Your Workflow

When you're satisfied with your workflow, click **Deploy** to make it live. The deployment process connects your workflow to GitHub and deploys it to LlamaCloud.

:::note
Deployment requires a GitHub account. Your workflow code will be stored in a GitHub repository, enabling version control and future customization.
:::

### Step 1: Connect GitHub

<!-- Screenshot placeholder: GitHub OAuth authorization step -->

Click **Connect GitHub** to authorize LlamaCloud to access your GitHub account. This uses standard GitHub OAuth and allows LlamaCloud to create repositories on your behalf.

### Step 2: Create or Select a Repository

Choose where to store your workflow code:

- **Create new repository**: LlamaCloud creates a new repo in your account with your workflow code
- **Select existing**: Choose an existing repository (useful for updates or adding to an existing project)

### Step 3: Install the LlamaCloud GitHub App

:::caution
This is a separate step from OAuth. The GitHub App grants LlamaCloud's deployment infrastructure access to your repository.
:::

You'll be prompted to install the **LlamaCloud GitHub App** on your repository. This grants LlamaCloud permission to:

- Read your repository contents
- Deploy updates when you push changes

**Why both OAuth and the GitHub App?** OAuth lets you authorize actions as yourself (like creating repositories). The GitHub App lets LlamaCloud's deployment infrastructure access your repo independently to build and deploy your workflow.

### Step 4: Configure and Deploy

Review the deployment configuration:

- **Environment variables**: Add any required API keys (e.g., `OPENAI_API_KEY` for LLM-powered steps)
- **Deployment name**: A unique identifier for this deployment

Click **Deploy**. LlamaCloud will build and deploy your workflow.

### Your Workflow is Live

Once deployment status shows **Running**, your workflow is ready to use:

- Click **Visit** to open the workflow's web interface
- Upload documents to process them through your pipeline
- View extracted data and results

## After Deployment

### Customizing Your Workflow Code

Your workflow is a real Python project in your GitHub repository. You can customize it beyond what the chat interface provides:

1. Clone the repository locally
2. Edit the workflow code directly
3. Push changes to GitHub
4. Update the deployment to pull your changes

For details on project structure, see [Configuration Reference](/python/llamaagents/llamactl/configuration-reference).

### Updating a Deployment

After pushing code changes to GitHub:

1. Go to your deployment in LlamaCloud
2. Click the **...** menu and select **Update Version**
3. LlamaCloud pulls and deploys your latest code

### Managing Deployments

From the LlamaCloud dashboard, you can:

- **Rollback** to a previous version if something breaks
- **Edit settings** to change the repository or branch
- **Delete** deployments you no longer need

For full deployment management details, see [Click-to-Deploy](/python/llamaagents/llamactl/click-to-deploy).

## Next Steps

- [LlamaExtract](/python/cloud/llamaextract/getting_started) — Learn about extraction schemas and configuration
- [LlamaClassify](/python/cloud/llamaclassify/getting_started) — Understand document classification
- [Agent Workflows](/python/llamaagents/workflows/) — Deep dive into workflow architecture
- [Configuration Reference](/python/llamaagents/llamactl/configuration-reference) — Project structure and settings
