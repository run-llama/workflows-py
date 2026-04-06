from llama_agents.cli.utils.redact import redact_api_key


def test_redact_api_key_long() -> None:
    value = "sk-1234567890abcdef"
    masked = redact_api_key(value)
    assert masked.startswith("sk-123")
    assert masked.endswith("cdef")
    assert "****" in masked


def test_redact_api_key_short() -> None:
    value = "abc1234"
    masked = redact_api_key(value)
    assert masked.startswith("abc123")
    assert masked.endswith("34")
    assert "****" in masked


def test_redact_api_key_none_and_empty() -> None:
    assert redact_api_key(None) == "-"
    assert redact_api_key("") == "-"
