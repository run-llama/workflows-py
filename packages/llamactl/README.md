# llamactl


A command-line interface for managing LlamaDeploy projects and deployments.

For an end-to-end introduction, see [Getting started with LlamaAgents](https://developers.llamaindex.ai/python/cloud/llamaagents/getting-started).

## Installation

Install from PyPI:

```bash
pip install llamactl
```

Or using uv:

```bash
uv add llamactl
```

## Quick Start

1. **Configure your profile**: Set up connection to your LlamaDeploy control plane
   ```bash
   llamactl profile configure
   ```

2. **Check health**: Verify connection to the control plane
   ```bash
   llamactl health
   ```

3. **Create a project**: Initialize a new deployment project
   ```bash
   llamactl project create my-project
   ```

4. **Deploy**: Deploy your project to the control plane
   ```bash
   llamactl deployment create my-deployment --project-name my-project
   ```

## Commands

### Profile Management
- `llamactl profile configure` - Configure connection to control plane
- `llamactl profile show` - Show current profile configuration
- `llamactl profile list` - List all configured profiles

### Project Management
- `llamactl project create <name>` - Create a new project
- `llamactl project list` - List all projects
- `llamactl project show <name>` - Show project details
- `llamactl project delete <name>` - Delete a project

### Deployment Management
- `llamactl deployment create <name>` - Create a new deployment
- `llamactl deployment list` - List all deployments
- `llamactl deployment show <name>` - Show deployment details
- `llamactl deployment delete <name>` - Delete a deployment
- `llamactl deployment logs <name>` - View deployment logs

### Health & Status
- `llamactl health` - Check control plane health
- `llamactl serve` - Start local development server

## Configuration

llamactl stores configuration in your home directory at `~/.llamactl/`.

### Profile Configuration
Profiles allow you to manage multiple control plane connections:

```bash
# Configure default profile
llamactl profile configure

# Configure named profile
llamactl profile configure --profile production

# Use specific profile for commands
llamactl --profile production deployment list
```

## Development

This CLI is part of the LlamaDeploy ecosystem. For development setup:

1. Clone the repository
2. Install dependencies: `uv sync`
3. Run tests: `uv run pytest`

## Requirements

- Python 3.12+
- Access to a LlamaDeploy control plane

## License

This project is licensed under the MIT License.
