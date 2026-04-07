# Docker Example

A minimal container image that serves a workflow with `WorkflowServer`.

## Files

| File | Purpose |
| --- | --- |
| [`app.py`](app.py) | The workflow and `WorkflowServer` entrypoint. |
| [`Dockerfile`](Dockerfile) | Builds a slim Python image with the app and its dependencies. |
| [`requirements.txt`](requirements.txt) | Runtime dependencies installed into the image. |

## Running

```bash
docker build -t workflows-example examples/docker
docker run --rm -p 8000:8000 workflows-example
```

Then call it from the host:

```bash
curl -X POST http://localhost:8000/workflows/echo/run -d '{"start_event": {"message": "hi"}}'
```
