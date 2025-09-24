from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.post_events_handler_id_response_200_status import PostEventsHandlerIdResponse200Status

T = TypeVar("T", bound="PostEventsHandlerIdResponse200")


@_attrs_define
class PostEventsHandlerIdResponse200:
    """
    Attributes:
        status (PostEventsHandlerIdResponse200Status):
    """

    status: PostEventsHandlerIdResponse200Status
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        status = PostEventsHandlerIdResponse200Status(d.pop("status"))

        post_events_handler_id_response_200 = cls(
            status=status,
        )

        post_events_handler_id_response_200.additional_properties = d
        return post_events_handler_id_response_200

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
