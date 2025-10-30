---
title: Workflows API Reference
---

<!-- THIS FILE IS GENERATED. DO NOT EDIT MANUALLY. -->

**Version**: `2.8.3`

This reference is generated from the server's OpenAPI schema. It lists all endpoints, their parameters, request bodies, and responses.


### /events/{handler_id}

#### GET

**Summary**: Stream workflow events


Streams events produced by a workflow execution. Events are emitted as
newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
Event data is returned as an envelope that preserves backward-compatible fields
and adds metadata for type-safety on the client:
{
  "value": &lt;pydantic serialized value&gt;,
  "types": [&lt;class names from MRO excluding the event class and base Event&gt;],
  "type": &lt;class name&gt;,
  "qualified_name": &lt;python module path + class name&gt;,
}

Event queue is mutable. Elements are added to the queue by the workflow handler, and removed by any consumer of the queue.
The queue is protected by a lock that is acquired by the consumer, so only one consumer of the queue at a time is allowed.


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| handler_id | path | yes | string | Identifier returned from the no-wait run endpoint. |
| sse | query | no | boolean | If false, as NDJSON instead of Server-Sent Events. |
| include_internal | query | no | boolean | If true, include internal workflow events (e.g., step state changes). |
| acquire_timeout | query | no | number | Timeout for acquiring the lock to iterate over the events. |
| include_qualified_name | query | no | boolean | If true, include the qualified name of the event in the response body. |


**Responses**


#### 200
Streaming started

- Content type: `text/event-stream`
```json
{
  "type": "object",
  "description": "Server-Sent Events stream of event data.",
  "properties": {
    "value": {
      "type": "object",
      "description": "The event value."
    },
    "type": {
      "type": "string",
      "description": "The class name of the event."
    },
    "types": {
      "type": "array",
      "description": "Superclass names from MRO (excluding the event class and base Event).",
      "items": {
        "type": "string"
      }
    },
    "qualified_name": {
      "type": "string",
      "description": "The qualified name of the event."
    }
  },
  "required": [
    "value",
    "type"
  ]
}
```

#### 404
Handler not found



#### POST

**Summary**: Send event to workflow


Sends an event to a running workflow's context.


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| handler_id | path | yes | string | Workflow handler identifier. |


**Request Body**


- Required: yes

- Content type: `application/json`

```json
{
  "type": "object",
  "properties": {
    "event": {
      "description": "Serialized event. Accepts object or JSON-encoded string for backward compatibility.",
      "oneOf": [
        {
          "type": "string",
          "description": "JSON string of the event envelope or value.",
          "examples": [
            "{\"type\": \"ExternalEvent\", \"value\": {\"response\": \"hi\"}}"
          ]
        },
        {
          "type": "object",
          "properties": {
            "type": {
              "type": "string",
              "description": "The class name of the event."
            },
            "value": {
              "type": "object",
              "description": "The event value object (preferred over data)."
            }
          },
          "additionalProperties": true
        }
      ]
    },
    "step": {
      "type": "string",
      "description": "Optional target step name. If not provided, event is sent to all steps."
    }
  },
  "required": [
    "event"
  ]
}
```


**Responses**


#### 200
Event sent successfully

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": [
        "sent"
      ]
    }
  },
  "required": [
    "status"
  ]
}
```

#### 400
Invalid event data


#### 404
Handler not found


#### 409
Workflow already completed





### /handlers

#### GET

**Summary**: Get handlers


Returns all workflow handlers.


**Responses**


#### 200
List of handlers

- Content type: `application/json`
  - Schema: `HandlersList`

```json
{
  "type": "object",
  "properties": {
    "handlers": {
      "type": "array",
      "items": {
        "$ref": "#/components/schemas/Handler"
      }
    }
  },
  "required": [
    "handlers"
  ]
}
```




### /handlers/{handler_id}/cancel

#### POST

**Summary**: Stop and delete handler


Stops a running workflow handler by cancelling its tasks. Optionally removes the
handler from the persistence store if purge=true.


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| handler_id | path | yes | string | Workflow handler identifier. |
| purge | query | no | boolean | If true, also deletes the handler from the store, otherwise updates the status to cancelled. |


**Responses**


#### 200
Handler cancelled and deleted or cancelled only

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": [
        "deleted",
        "cancelled"
      ]
    }
  },
  "required": [
    "status"
  ]
}
```

#### 404
Handler not found





### /health

#### GET

**Summary**: Health check


Returns the server health status.


**Responses**


#### 200
Successful health check

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "example": "healthy"
    }
  },
  "required": [
    "status"
  ]
}
```




### /results/{handler_id}

#### GET

**Summary**: Get workflow result


Returns the final result of an asynchronously started workflow, if available


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| handler_id | path | yes | string | Workflow run identifier returned from the no-wait run endpoint. |


**Responses**


#### 200
Result is available

- Content type: `application/json`
  - Schema: `Handler`

```json
{
  "type": "object",
  "properties": {
    "handler_id": {
      "type": "string"
    },
    "workflow_name": {
      "type": "string"
    },
    "run_id": {
      "type": "string",
      "nullable": true
    },
    "status": {
      "type": "string",
      "enum": [
        "running",
        "completed",
        "failed",
        "cancelled"
      ]
    },
    "started_at": {
      "type": "string",
      "format": "date-time"
    },
    "updated_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "completed_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "error": {
      "type": "string",
      "nullable": true
    },
    "result": {
      "description": "Workflow result value"
    }
  },
  "required": [
    "handler_id",
    "workflow_name",
    "status",
    "started_at"
  ]
}
```

#### 202
Result not ready yet

- Content type: `application/json`
  - Schema: `Handler`

```json
{
  "type": "object",
  "properties": {
    "handler_id": {
      "type": "string"
    },
    "workflow_name": {
      "type": "string"
    },
    "run_id": {
      "type": "string",
      "nullable": true
    },
    "status": {
      "type": "string",
      "enum": [
        "running",
        "completed",
        "failed",
        "cancelled"
      ]
    },
    "started_at": {
      "type": "string",
      "format": "date-time"
    },
    "updated_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "completed_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "error": {
      "type": "string",
      "nullable": true
    },
    "result": {
      "description": "Workflow result value"
    }
  },
  "required": [
    "handler_id",
    "workflow_name",
    "status",
    "started_at"
  ]
}
```

#### 404
Handler not found


#### 500
Error computing result

- Content type: `text/plain`
```json
{
  "type": "string"
}
```




### /workflows

#### GET

**Summary**: List workflows


Returns the list of registered workflow names.


**Responses**


#### 200
List of workflows

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "workflows": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "workflows"
  ]
}
```




### /workflows/{name}/events

#### GET

**Summary**: List workflow events


Returns the list of registered workflow event schemas.


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| name | path | yes | string | Registered workflow name. |


**Responses**


#### 200
List of workflow event schemas

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "events": {
      "type": "array",
      "description": "List of workflow event JSON schemas",
      "items": {
        "type": "object"
      }
    }
  },
  "required": [
    "events"
  ]
}
```




### /workflows/{name}/representation

#### GET

**Summary**: Get the representation of the workflow


Get the representation of the workflow as a directed graph in JSON format


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| name | path | yes | string | Registered workflow name. |


**Request Body**


- Required: no


**Responses**


#### 200
JSON representation successfully retrieved

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "graph": {
      "description": "the elements of the JSON representation of the workflow"
    }
  },
  "required": [
    "graph"
  ]
}
```

#### 404
Workflow not found


#### 500
Error while getting JSON workflow representation





### /workflows/{name}/run

#### POST

**Summary**: Run workflow (wait)


Runs the specified workflow synchronously and returns the final result.
The request body may include an optional serialized start event, an optional
context object, and optional keyword arguments passed to the workflow run.


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| name | path | yes | string | Registered workflow name. |


**Request Body**


- Required: no

- Content type: `application/json`

```json
{
  "type": "object",
  "properties": {
    "start_event": {
      "type": "object",
      "description": "Plain JSON object representing the start event (e.g., {\"message\": \"...\"})."
    },
    "context": {
      "type": "object",
      "description": "Serialized workflow Context."
    },
    "handler_id": {
      "type": "string",
      "description": "Workflow handler identifier to continue from a previous completed run."
    },
    "kwargs": {
      "type": "object",
      "description": "Additional keyword arguments for the workflow."
    }
  }
}
```


**Responses**


#### 200
Workflow completed successfully

- Content type: `application/json`
  - Schema: `Handler`

```json
{
  "type": "object",
  "properties": {
    "handler_id": {
      "type": "string"
    },
    "workflow_name": {
      "type": "string"
    },
    "run_id": {
      "type": "string",
      "nullable": true
    },
    "status": {
      "type": "string",
      "enum": [
        "running",
        "completed",
        "failed",
        "cancelled"
      ]
    },
    "started_at": {
      "type": "string",
      "format": "date-time"
    },
    "updated_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "completed_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "error": {
      "type": "string",
      "nullable": true
    },
    "result": {
      "description": "Workflow result value"
    }
  },
  "required": [
    "handler_id",
    "workflow_name",
    "status",
    "started_at"
  ]
}
```

#### 400
Invalid start_event payload


#### 404
Workflow or handler identifier not found


#### 500
Error running workflow or invalid request body





### /workflows/{name}/run-nowait

#### POST

**Summary**: Run workflow (no-wait)


Starts the specified workflow asynchronously and returns a handler identifier
which can be used to query results or stream events.


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| name | path | yes | string | Registered workflow name. |


**Request Body**


- Required: no

- Content type: `application/json`

```json
{
  "type": "object",
  "properties": {
    "start_event": {
      "type": "object",
      "description": "Plain JSON object representing the start event (e.g., {\"message\": \"...\"})."
    },
    "context": {
      "type": "object",
      "description": "Serialized workflow Context."
    },
    "handler_id": {
      "type": "string",
      "description": "Workflow handler identifier to continue from a previous completed run."
    },
    "kwargs": {
      "type": "object",
      "description": "Additional keyword arguments for the workflow."
    }
  }
}
```


**Responses**


#### 200
Workflow started

- Content type: `application/json`
  - Schema: `Handler`

```json
{
  "type": "object",
  "properties": {
    "handler_id": {
      "type": "string"
    },
    "workflow_name": {
      "type": "string"
    },
    "run_id": {
      "type": "string",
      "nullable": true
    },
    "status": {
      "type": "string",
      "enum": [
        "running",
        "completed",
        "failed",
        "cancelled"
      ]
    },
    "started_at": {
      "type": "string",
      "format": "date-time"
    },
    "updated_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "completed_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "error": {
      "type": "string",
      "nullable": true
    },
    "result": {
      "description": "Workflow result value"
    }
  },
  "required": [
    "handler_id",
    "workflow_name",
    "status",
    "started_at"
  ]
}
```

#### 400
Invalid start_event payload


#### 404
Workflow or handler identifier not found





### /workflows/{name}/schema

#### GET

**Summary**: Get JSON schema for start event


Gets the JSON schema of the start and stop events from the specified workflow and returns it under "start" (start event) and "stop" (stop event)


**Parameters**


| Name | In | Required | Type | Description |
| --- | --- | :---: | --- | --- |
| name | path | yes | string | Registered workflow name. |


**Request Body**


- Required: no


**Responses**


#### 200
JSON schema successfully retrieved for start event

- Content type: `application/json`
```json
{
  "type": "object",
  "properties": {
    "start": {
      "description": "JSON schema for the start event"
    },
    "stop": {
      "description": "JSON schema for the stop event"
    }
  },
  "required": [
    "start",
    "stop"
  ]
}
```

#### 404
Workflow not found


#### 500
Error while getting the JSON schema for the start or stop event





### Components


These are the component schemas referenced above.


#### Handler


```json
{
  "type": "object",
  "properties": {
    "handler_id": {
      "type": "string"
    },
    "workflow_name": {
      "type": "string"
    },
    "run_id": {
      "type": "string",
      "nullable": true
    },
    "status": {
      "type": "string",
      "enum": [
        "running",
        "completed",
        "failed",
        "cancelled"
      ]
    },
    "started_at": {
      "type": "string",
      "format": "date-time"
    },
    "updated_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "completed_at": {
      "type": "string",
      "format": "date-time",
      "nullable": true
    },
    "error": {
      "type": "string",
      "nullable": true
    },
    "result": {
      "description": "Workflow result value"
    }
  },
  "required": [
    "handler_id",
    "workflow_name",
    "status",
    "started_at"
  ]
}
```



#### HandlersList


```json
{
  "type": "object",
  "properties": {
    "handlers": {
      "type": "array",
      "items": {
        "$ref": "#/components/schemas/Handler"
      }
    }
  },
  "required": [
    "handlers"
  ]
}
```


