from pydantic import BaseModel, ConfigDict

base_config = ConfigDict(
    from_attributes=True,
    arbitrary_types_allowed=True,
    use_enum_values=False,
    # ===== timedelta serialization =====
    # Serialize timedelta as float
    # This was the default behavior in pydantic v1, but was changed in v2 to be "iso8601"
    # We want to keep the same behavior as v1 for now
    # ================================
    ser_json_timedelta="float",
    # NOTE: we often use data model with fields that have "model_" prefix,
    # so we need to set protected_namespaces=() to avoid conflict
    protected_namespaces=(),
)


class Base(BaseModel):
    model_config = base_config
