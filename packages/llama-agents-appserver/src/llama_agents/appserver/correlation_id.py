import random
import string
from contextvars import ContextVar

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    correlation_id_var.set(correlation_id)


def create_correlation_id() -> str:
    return random_alphanumeric_string(8)


_alphanumeric_chars = string.ascii_letters + string.digits


def random_alphanumeric_string(length: int) -> str:
    return "".join(random.choices(_alphanumeric_chars, k=length))
