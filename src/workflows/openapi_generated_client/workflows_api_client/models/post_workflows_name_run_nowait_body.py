from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.post_workflows_name_run_nowait_body_context import PostWorkflowsNameRunNowaitBodyContext
    from ..models.post_workflows_name_run_nowait_body_kwargs import PostWorkflowsNameRunNowaitBodyKwargs
    from ..models.post_workflows_name_run_nowait_body_start_event import PostWorkflowsNameRunNowaitBodyStartEvent


T = TypeVar("T", bound="PostWorkflowsNameRunNowaitBody")


@_attrs_define
class PostWorkflowsNameRunNowaitBody:
    """
    Attributes:
        start_event (Union[Unset, PostWorkflowsNameRunNowaitBodyStartEvent]): Plain JSON object representing the start
            event (e.g., {"message": "..."}).
        context (Union[Unset, PostWorkflowsNameRunNowaitBodyContext]): Serialized workflow Context.
        kwargs (Union[Unset, PostWorkflowsNameRunNowaitBodyKwargs]): Additional keyword arguments for the workflow.
    """

    start_event: Union[Unset, "PostWorkflowsNameRunNowaitBodyStartEvent"] = UNSET
    context: Union[Unset, "PostWorkflowsNameRunNowaitBodyContext"] = UNSET
    kwargs: Union[Unset, "PostWorkflowsNameRunNowaitBodyKwargs"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start_event: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.start_event, Unset):
            start_event = self.start_event.to_dict()

        context: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.context, Unset):
            context = self.context.to_dict()

        kwargs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.kwargs, Unset):
            kwargs = self.kwargs.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if start_event is not UNSET:
            field_dict["start_event"] = start_event
        if context is not UNSET:
            field_dict["context"] = context
        if kwargs is not UNSET:
            field_dict["kwargs"] = kwargs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.post_workflows_name_run_nowait_body_context import PostWorkflowsNameRunNowaitBodyContext
        from ..models.post_workflows_name_run_nowait_body_kwargs import PostWorkflowsNameRunNowaitBodyKwargs
        from ..models.post_workflows_name_run_nowait_body_start_event import PostWorkflowsNameRunNowaitBodyStartEvent

        d = dict(src_dict)
        _start_event = d.pop("start_event", UNSET)
        start_event: Union[Unset, PostWorkflowsNameRunNowaitBodyStartEvent]
        if isinstance(_start_event, Unset):
            start_event = UNSET
        else:
            start_event = PostWorkflowsNameRunNowaitBodyStartEvent.from_dict(_start_event)

        _context = d.pop("context", UNSET)
        context: Union[Unset, PostWorkflowsNameRunNowaitBodyContext]
        if isinstance(_context, Unset):
            context = UNSET
        else:
            context = PostWorkflowsNameRunNowaitBodyContext.from_dict(_context)

        _kwargs = d.pop("kwargs", UNSET)
        kwargs: Union[Unset, PostWorkflowsNameRunNowaitBodyKwargs]
        if isinstance(_kwargs, Unset):
            kwargs = UNSET
        else:
            kwargs = PostWorkflowsNameRunNowaitBodyKwargs.from_dict(_kwargs)

        post_workflows_name_run_nowait_body = cls(
            start_event=start_event,
            context=context,
            kwargs=kwargs,
        )

        post_workflows_name_run_nowait_body.additional_properties = d
        return post_workflows_name_run_nowait_body

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
