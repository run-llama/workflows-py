# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import base64
import json
import pickle
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .utils import get_qualified_name, import_module_from_qualified_name


class BaseSerializer(ABC):
    @abstractmethod
    def serialize(self, value: Any) -> str: ...

    @abstractmethod
    def deserialize(self, value: str) -> Any: ...


class JsonSerializer(BaseSerializer):
    def _serialize_value(self, value: Any) -> Any:
        """Helper to serialize a single value."""
        # Note: to avoid circular dependencies we cannot import BaseComponent from llama_index.core
        # if we want to use isinstance(value, BaseComponent) instead of guessing type from the presence
        # of class_name, we need to move BaseComponent out of core
        if hasattr(value, "class_name"):
            retval = {
                "__is_component": True,
                "value": value.to_dict(),
                "qualified_name": get_qualified_name(value),
            }
            return retval

        if isinstance(value, BaseModel):
            return {
                "__is_pydantic": True,
                "value": value.model_dump(mode="json"),
                "qualified_name": get_qualified_name(value),
            }

        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]

        return value

    def serialize(self, value: Any) -> str:
        try:
            serialized_value = self._serialize_value(value)
            return json.dumps(serialized_value)
        except Exception:
            raise ValueError(f"Failed to serialize value: {type(value)}: {value!s}")

    def _deserialize_value(self, data: Any) -> Any:
        """Helper to deserialize a single value."""
        if isinstance(data, dict):
            if data.get("__is_pydantic") and data.get("qualified_name"):
                module_class = import_module_from_qualified_name(data["qualified_name"])
                return module_class.model_validate(data["value"])
            elif data.get("__is_component") and data.get("qualified_name"):
                module_class = import_module_from_qualified_name(data["qualified_name"])
                return module_class.from_dict(data["value"])
            return {k: self._deserialize_value(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_value(item) for item in data]
        return data

    def deserialize(self, value: str) -> Any:
        data = json.loads(value)
        return self._deserialize_value(data)


class PickleSerializer(JsonSerializer):
    def serialize(self, value: Any) -> str:
        """Serialize while prioritizing JSON, falling back to Pickle."""
        try:
            return super().serialize(value)
        except Exception:
            return base64.b64encode(pickle.dumps(value)).decode("utf-8")

    def deserialize(self, value: str) -> Any:
        """
        Deserialize while prioritizing Pickle, falling back to JSON.
        To avoid malicious exploits of the deserialization, deserialize objects
        only when you deem it safe to do so.
        """
        try:
            return pickle.loads(base64.b64decode(value))
        except Exception:
            return super().deserialize(value)


JsonPickleSerializer = PickleSerializer
