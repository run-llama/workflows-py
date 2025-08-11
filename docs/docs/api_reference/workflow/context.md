::: workflows.context.Context
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - __init__
        - collect_events
        - from_dict
        - get_result
        - is_running
        - send_event
        - store
        - to_dict
        - wait_for_event
        - write_event_to_stream


::: workflows.context.state_store
    options:
      members:
        - DictState
        - InMemoryStateStore

::: workflows.context.serializers
    options:
      members:
        - BaseSerializer
        - JsonSerializer
        - PickleSerializer
