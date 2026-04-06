# DBOS

Durable workflows backed by [DBOS](https://www.dbos.dev/). A shared [`docker-compose.yml`](./docker-compose.yml) provides the Postgres instance used by all examples.

Key files:
- [`durable_workflow.py`](./durable_workflow.py) — minimal DBOS-backed counter workflow
- [`server_quickstart.py`](./server_quickstart.py) — quick-start durable workflow server
- [`server_replicas.py`](./server_replicas.py) — multi-replica durable server demo
- [`idle_release_demo.py`](./idle_release_demo.py) — idle release and resume behavior
