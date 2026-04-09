# Common base stage. Intentionally does NOT install the git CLI: the
# control plane reads git repositories via dulwich (pure Python) and the
# git binary is dead weight + supply-chain surface in that image.
# Stages that DO need git (currently just appserver, because user code
# may declare git+https:// pip dependencies) inherit from base-appserver.
FROM python:3.12-slim AS base-common

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /uvx /bin/

# Create a non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Create data directory and set permissions
RUN mkdir -p /app && chown -R appuser:appgroup /app

# Set working directory
WORKDIR /app

# Copy workspace configuration files
COPY pyproject.toml uv.lock README.md ./

ENV PATH=/app/.venv/bin:$PATH


##############################################################################
###                Base for stages that need the git CLI                   ###
##############################################################################
#
# git is required by user-code "pip install git+https://..." dependencies
# during appserver bootstrap.
FROM base-common AS base-appserver

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*


##############################################################################
##############################################################################
############################## CONTROL PLANE #################################
##############################################################################
##############################################################################

##############################################################################
###                    Control Plane - BUILDER                             ###
##############################################################################
#
# pulls in all packages, and installs as non-editable
FROM base-common AS controlplane-builder

ARG PACKAGE_NAME=llama-agents-control-plane

# installs the 3rd party deps for the package
RUN uv sync --frozen --no-dev --package $PACKAGE_NAME --no-install-workspace
# Copy the 1st party deps
COPY packages/ packages/

# Now install the 1st party deps into the venv (non-editable)
RUN uv sync --frozen --no-dev --package $PACKAGE_NAME --no-editable

##############################################################################
###                   Control Plane - RUNNER                               ###
##############################################################################

FROM base-common AS controlplane

# Copy the builder stage
COPY --from=controlplane-builder /app/.venv /app/.venv

USER appuser

# Expose ports
EXPOSE 8000 8001

# Production command without reload
CMD ["python", "-m", "llama_agents.control_plane.main", "--host", "0.0.0.0", "--manage-api-port", "8000", "--build-api-port", "8001", "--log-level", "info", "--log-format", "json"]



##############################################################################
##############################################################################
################################# APP SERVER #################################
##############################################################################
##############################################################################


##############################################################################
###                       App Server - BUILDER                             ###
##############################################################################
#
# pulls in all packages, and installs as non-editable
FROM base-appserver AS appserver-builder

ARG PACKAGE_NAME=llama-agents-appserver

# install to system python. Makes things easier for the cloned repo to "inherit" the appserver dependencies
# installs the 3rd party deps for the package
RUN uv sync --frozen --no-dev --package $PACKAGE_NAME --no-install-workspace
# Copy the 1st party deps
COPY packages/ packages/

# Now install the 1st party deps into the venv (non-editable)
RUN uv sync --frozen --no-dev --package $PACKAGE_NAME --no-editable
RUN uv build --package $PACKAGE_NAME --sdist
RUN uv build --package llama-agents-core --sdist

##############################################################################
###                   App Server - RUNNER                                  ###
##############################################################################

FROM base-appserver AS appserver

# Install Node.js and pnpm
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && corepack enable \
    && rm -rf /var/lib/apt/lists/*

# Copy the builder stage
COPY --from=appserver-builder /app/.venv /app/.venv
COPY --from=appserver-builder /app/dist/ /app/dist
ENV LLAMA_DEPLOY_BOOTSTRAP_SDISTS=/app/dist/

USER appuser

ENV LLAMA_DEPLOY_APISERVER_PORT=8080
ENV LLAMA_DEPLOY_APISERVER_HOST=0.0.0.0

# Expose ports
EXPOSE 8080
