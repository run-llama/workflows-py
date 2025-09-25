from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.handler import Handler


T = TypeVar("T", bound="HandlersList")


@_attrs_define
class HandlersList:
    """
    Attributes:
        handlers (list['Handler']):
    """

    handlers: list["Handler"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        handlers = []
        for handlers_item_data in self.handlers:
            handlers_item = handlers_item_data.to_dict()
            handlers.append(handlers_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "handlers": handlers,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.handler import Handler

        d = dict(src_dict)
        handlers = []
        _handlers = d.pop("handlers")
        for handlers_item_data in _handlers:
            handlers_item = Handler.from_dict(handlers_item_data)

            handlers.append(handlers_item)

        handlers_list = cls(
            handlers=handlers,
        )

        handlers_list.additional_properties = d
        return handlers_list

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
