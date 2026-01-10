#!/usr/bin/env python3
"""
Example demonstrating resource nodes in workflow graph visualization.

This example shows how resources (dependencies injected via Annotated types)
are rendered in both Mermaid and Pyvis diagrams.

Run this script to generate:
- workflow_with_resources.mermaid - A Mermaid diagram file
- workflow_with_resources.html - An interactive Pyvis HTML visualization

You can view the Mermaid diagram at https://mermaid.live/ by pasting the contents.
The HTML file can be opened directly in any web browser.
"""

import argparse
from typing import Annotated

from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_all_possible_flows_mermaid,
)
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource

# --- Mock resource types ---


class DatabaseClient:
    """A database client for persistent storage."""

    def __init__(self, connection_string: str = "postgres://localhost/db"):
        self.connection_string = connection_string

    def query(self, sql: str) -> list:
        """Execute a SQL query."""
        return []


class CacheClient:
    """A cache client for fast data retrieval."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port

    def get(self, key: str) -> str | None:
        """Get a value from cache."""
        return None

    def set(self, key: str, value: str) -> None:
        """Set a value in cache."""
        pass


class LLMClient:
    """A client for interacting with a large language model."""

    def __init__(self, api_key: str = "sk-..."):
        self.api_key = api_key

    async def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        return f"Response to: {prompt}"


# --- Resource factory functions ---


def get_database_client() -> DatabaseClient:
    """Factory function to create a database client.

    This function creates a PostgreSQL database client configured
    for the application's data storage needs.
    """
    return DatabaseClient(connection_string="postgres://localhost/myapp")


def get_cache_client() -> CacheClient:
    """Factory function to create a Redis cache client.

    Provides fast caching for frequently accessed data.
    """
    return CacheClient(host="localhost", port=6379)


def get_llm_client() -> LLMClient:
    """Factory function to create an LLM client.

    Creates a client for the language model API.
    """
    return LLMClient(api_key="sk-example-key")


# --- Event types ---


class QueryProcessedEvent(Event):
    """Event emitted after processing a user query."""

    query: str
    cached: bool = False


class ContextRetrievedEvent(Event):
    """Event emitted after retrieving context from the database."""

    context: str


class ResponseGeneratedEvent(Event):
    """Event emitted after generating an LLM response."""

    response: str


# --- Workflow with resources ---


class RAGWorkflow(Workflow):
    """A RAG (Retrieval-Augmented Generation) workflow demonstrating resource usage.

    This workflow shows how different steps can depend on shared resources
    like database clients, cache clients, and LLM clients.
    """

    @step
    async def process_query(
        self,
        ev: StartEvent,
        cache: Annotated[CacheClient, Resource(get_cache_client)],
    ) -> QueryProcessedEvent:
        """Process the incoming query, checking cache first."""
        query = getattr(ev, "query", "default query")

        # Check if query result is cached
        cached_result = cache.get(f"query:{query}")
        if cached_result:
            return QueryProcessedEvent(query=query, cached=True)

        return QueryProcessedEvent(query=query, cached=False)

    @step
    async def retrieve_context(
        self,
        ev: QueryProcessedEvent,
        db: Annotated[DatabaseClient, Resource(get_database_client)],
        cache: Annotated[CacheClient, Resource(get_cache_client)],
    ) -> ContextRetrievedEvent:
        """Retrieve relevant context from the database."""
        if ev.cached:
            context = "Cached context"
        else:
            # Query the database for relevant documents
            results = db.query(
                f"SELECT content FROM documents WHERE query = '{ev.query}'"
            )
            context = " ".join(str(r) for r in results) or "No context found"

            # Cache the result
            cache.set(f"context:{ev.query}", context)

        return ContextRetrievedEvent(context=context)

    @step
    async def generate_response(
        self,
        ev: ContextRetrievedEvent,
        llm: Annotated[LLMClient, Resource(get_llm_client)],
    ) -> ResponseGeneratedEvent:
        """Generate a response using the LLM with the retrieved context."""
        prompt = f"Context: {ev.context}\n\nGenerate a response."
        response = await llm.complete(prompt)
        return ResponseGeneratedEvent(response=response)

    @step
    async def finalize_response(
        self,
        ev: ResponseGeneratedEvent,
        cache: Annotated[CacheClient, Resource(get_cache_client)],
    ) -> StopEvent:
        """Finalize and cache the response."""
        # Cache the final response
        cache.set("last_response", ev.response)
        return StopEvent(result=ev.response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate workflow visualizations with resource nodes"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output files (default: current directory)",
    )
    parser.add_argument(
        "--mermaid-only",
        action="store_true",
        help="Only generate Mermaid output (print to stdout)",
    )
    args = parser.parse_args()

    # Create the workflow
    workflow = RAGWorkflow()

    print("=" * 60)
    print("Workflow Graph Visualization with Resource Nodes")
    print("=" * 60)

    # Generate Mermaid diagram
    mermaid_file = f"{args.output_dir}/workflow_with_resources.mermaid"
    mermaid_output = draw_all_possible_flows_mermaid(
        workflow,
        filename="" if args.mermaid_only else mermaid_file,
    )

    print("\n--- Mermaid Diagram ---")
    print(mermaid_output)
    print()

    if not args.mermaid_only:
        print(f"Mermaid diagram saved to: {mermaid_file}")

        # Generate Pyvis HTML
        html_file = f"{args.output_dir}/workflow_with_resources.html"
        draw_all_possible_flows(workflow, filename=html_file)
        print(f"Interactive Pyvis diagram saved to: {html_file}")

    print("\n--- Resource Nodes Summary ---")
    print(
        """
The diagram shows:
- HEXAGON nodes (plum color): Resource dependencies
  - DatabaseClient: Database connection via get_database_client()
  - CacheClient: Cache connection via get_cache_client()
  - LLMClient: LLM API client via get_llm_client()

- Edge labels on resource connections show the variable name used in the step
  e.g., "db", "cache", "llm"

- Resources are deduplicated: CacheClient appears once even though
  it's used by multiple steps (process_query, retrieve_context, finalize_response)

To view the Mermaid diagram:
  1. Go to https://mermaid.live/
  2. Paste the diagram content above
  3. See the interactive visualization

To view the Pyvis diagram:
  1. Open workflow_with_resources.html in a web browser
  2. Hover over nodes to see metadata (type, getter, source location, docstring)
  3. Drag nodes to rearrange the layout
"""
    )


if __name__ == "__main__":
    main()
