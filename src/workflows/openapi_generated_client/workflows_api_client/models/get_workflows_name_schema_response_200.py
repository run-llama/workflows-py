from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="GetWorkflowsNameSchemaResponse200")


@_attrs_define
class GetWorkflowsNameSchemaResponse200:
    """
    Attributes:
        start (Any): JSON schema for the start event
        stop (Any): JSON schema for the stop event
    """

    start: Any
    stop: Any
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start = self.start

        stop = self.stop

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "start": start,
                "stop": stop,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        start = d.pop("start")

        stop = d.pop("stop")

        get_workflows_name_schema_response_200 = cls(
            start=start,
            stop=stop,
        )

        get_workflows_name_schema_response_200.additional_properties = d
        return get_workflows_name_schema_response_200

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
