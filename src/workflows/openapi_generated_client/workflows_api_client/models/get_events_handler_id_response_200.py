from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.get_events_handler_id_response_200_value import GetEventsHandlerIdResponse200Value


T = TypeVar("T", bound="GetEventsHandlerIdResponse200")


@_attrs_define
class GetEventsHandlerIdResponse200:
    """Server-Sent Events stream of event data.

    Attributes:
        value (GetEventsHandlerIdResponse200Value): The event value.
        qualified_name (str): The qualified name of the event.
    """

    value: "GetEventsHandlerIdResponse200Value"
    qualified_name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = self.value.to_dict()

        qualified_name = self.qualified_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "value": value,
                "qualified_name": qualified_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.get_events_handler_id_response_200_value import GetEventsHandlerIdResponse200Value

        d = dict(src_dict)
        value = GetEventsHandlerIdResponse200Value.from_dict(d.pop("value"))

        qualified_name = d.pop("qualified_name")

        get_events_handler_id_response_200 = cls(
            value=value,
            qualified_name=qualified_name,
        )

        get_events_handler_id_response_200.additional_properties = d
        return get_events_handler_id_response_200

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
