# Writing durable workflows

Workflows are ephemeral by default, meaning that once the `run()` method returns its result, the workflow state is lost. A subsequent call to `run()` on the same workflow instance will start from a fresh state.

If the use case requires to persist the workflow state  across multiple runs and possibly different processes, there are a few strategies that can be used to make workflows more durable.

## Storing data in the workflow instance

Workflows are regular Python classes, and data can be stored in class or instance variables, so that subsequent `run()` invocations can access it.

```python
class DbWorkflow(Workflow):
    def __init__(self, db: Client, *args, **kwargs):
        self.db = db
        super().__init__(*args, **kwargs)

    @step
    def count(self, ev: StartEvent) -> StopEvent:
        num_rows = self.db.exec("select COUNT(*) from t;")
        return StopEvent(result=num_rows)
```

In this case, multiple calls to `run()` will reuse the same database client.

| Persists over `run` calls | ✅ |
| --- | --- |
| Persists over process restarts | ❌ |
| Survives runtime errors | ❌ |

## Storing data in the context object

Each workflow comes with a special object responsible for its runtime operations called `Context`. The context instance is available to any step of a workflow and comes with a `store` property that can be used to store and load state data. Using the state store has two major advantages compared to class and instance variables:

- It’s async safe and supports concurrent access
- It can be serialized

```python
w = MyWorkflow()
handler = w.run()
context = handler.ctx
# Save the context to a database
db.save("id", context.to_dict())

#
# Restart the Python process...
#

w = MyWorkflow()
# Load the context from the database
context = Context.from_dict(w, db.load("id"))
# Pass the context containing the state to the workflow
result = await w.run(ctx=context)
```

| Persists over `run` calls | ✅ |
| --- | --- |
| Persists over process restarts | ✅ |
| Survives runtime errors | ❌ |

## Using external resources to checkpoint execution

To avoid any overhead, workflows don’t take snapshots of the current state automatically, so they can’t survive a fatal error on their own. However, any step can rely on some external database like Redis and snapshot the current context on sensitive parts of the code.

For example, given a long running workflow processing hundreds of documents, we could save the id of the last document successfully processed in the state store:

```python
class DurableWorkflow(Workflow):
    def __init__(self, r: Redis):
        self.redis = r

    @step
    def convert_documents(self, ev: StartEvent, ctx: Context) -> StopEvent:
        # Get the workflow input
        document_ids = ev.ids
        # Get the list of processed documents from the state store
        converted_ids = await ctx.store.get("converted_ids", default=[])
        for doc_id in document_ids:
		        # Ignore documents that were alredy processed
		        if doc_id in converted_ids:
		            continue
            convert()
            # Update the state store
            converted_id.append(doc_id)
            await ctx.store.set("converted_ids", converted_ids)
            # Create a snapshot of the current context
            self.redis.hset("ctx", mapping=ctx.to_dict())
```

The workflow will use a Redis collection to store a snapshot of the current context after every conversion. If the process running the workflow crashes, the process can be safely restarted with the same input. In fact, `ctx.store` will contain the list of documents already processed and the `for` loop will be able to skip them and continue to process the remaining work.

### Bonus: inject dependencies into the workflow to reduce boilerplate

Using the Resource feature of workflows, the Redis client can be injected into the step directly:

```python
def get_redis_client(*args, **kwargs):
		"""This can be reused across several workflows to reduce boilerplate"""
    return Redis(host='localhost', port=6379, decode_responses=True)


class DurableWorkflow(Workflow):
    @step
    def convert_documents(
        self,
        ev: StartEvent,
        ctx: Context,
        redis: Annotated[Redis, Resource(get_redis_client)]
    ) -> StopEvent:
        # Get the workflow input
        document_ids = ev.ids
        # Get the list of processed documents from the state store
        converted_ids = await ctx.store.get("converted_ids", default=[])
        for doc_id in document_ids:
		        # Ignore documents that were alredy processed
		        if doc_id in converted_ids:
		            continue
            convert()
            # Update the state store
            converted_id.append(doc_id)
            await ctx.store.set("converted_ids", converted_ids)
            # Create a snapshot of the current context
            redis.hset("ctx", mapping=ctx.to_dict())
```

| Persists over `run` calls | ✅ |
| --- | --- |
| Persists over process restarts | ✅ |
| Survives runtime errors | ✅ |
