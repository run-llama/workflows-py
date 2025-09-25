import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.handler_status import HandlerStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="Handler")


@_attrs_define
class Handler:
    """
    Attributes:
        handler_id (str):
        workflow_name (str):
        status (HandlerStatus):
        started_at (datetime.datetime):
        run_id (Union[None, Unset, str]):
        updated_at (Union[None, Unset, datetime.datetime]):
        completed_at (Union[None, Unset, datetime.datetime]):
        error (Union[None, Unset, str]):
        result (Union[Unset, Any]): Workflow result value
    """

    handler_id: str
    workflow_name: str
    status: HandlerStatus
    started_at: datetime.datetime
    run_id: Union[None, Unset, str] = UNSET
    updated_at: Union[None, Unset, datetime.datetime] = UNSET
    completed_at: Union[None, Unset, datetime.datetime] = UNSET
    error: Union[None, Unset, str] = UNSET
    result: Union[Unset, Any] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        handler_id = self.handler_id

        workflow_name = self.workflow_name

        status = self.status.value

        started_at = self.started_at.isoformat()

        run_id: Union[None, Unset, str]
        if isinstance(self.run_id, Unset):
            run_id = UNSET
        else:
            run_id = self.run_id

        updated_at: Union[None, Unset, str]
        if isinstance(self.updated_at, Unset):
            updated_at = UNSET
        elif isinstance(self.updated_at, datetime.datetime):
            updated_at = self.updated_at.isoformat()
        else:
            updated_at = self.updated_at

        completed_at: Union[None, Unset, str]
        if isinstance(self.completed_at, Unset):
            completed_at = UNSET
        elif isinstance(self.completed_at, datetime.datetime):
            completed_at = self.completed_at.isoformat()
        else:
            completed_at = self.completed_at

        error: Union[None, Unset, str]
        if isinstance(self.error, Unset):
            error = UNSET
        else:
            error = self.error

        result = self.result

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "handler_id": handler_id,
                "workflow_name": workflow_name,
                "status": status,
                "started_at": started_at,
            }
        )
        if run_id is not UNSET:
            field_dict["run_id"] = run_id
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at
        if completed_at is not UNSET:
            field_dict["completed_at"] = completed_at
        if error is not UNSET:
            field_dict["error"] = error
        if result is not UNSET:
            field_dict["result"] = result

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        handler_id = d.pop("handler_id")

        workflow_name = d.pop("workflow_name")

        status = HandlerStatus(d.pop("status"))

        started_at = isoparse(d.pop("started_at"))

        def _parse_run_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        run_id = _parse_run_id(d.pop("run_id", UNSET))

        def _parse_updated_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                updated_at_type_0 = isoparse(data)

                return updated_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        updated_at = _parse_updated_at(d.pop("updated_at", UNSET))

        def _parse_completed_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                completed_at_type_0 = isoparse(data)

                return completed_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        completed_at = _parse_completed_at(d.pop("completed_at", UNSET))

        def _parse_error(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        error = _parse_error(d.pop("error", UNSET))

        result = d.pop("result", UNSET)

        handler = cls(
            handler_id=handler_id,
            workflow_name=workflow_name,
            status=status,
            started_at=started_at,
            run_id=run_id,
            updated_at=updated_at,
            completed_at=completed_at,
            error=error,
            result=result,
        )

        handler.additional_properties = d
        return handler

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
