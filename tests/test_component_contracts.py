from __future__ import annotations

import pytest

from image2editable.component_contracts import (
    AGENT_PROVIDERS,
    MAX_REPAIR_ROUNDS,
    validate_agent_provider,
)


def test_component_agent_provider_contract_is_frozen() -> None:
    assert AGENT_PROVIDERS == frozenset({"host", "local"})
    assert MAX_REPAIR_ROUNDS == 5


@pytest.mark.parametrize("value", ["host", "local"])
def test_validate_agent_provider_accepts_supported_lowercase_values(value: str) -> None:
    assert validate_agent_provider(value) == value


@pytest.mark.parametrize("value", ["", "HOST", "remote", None])
def test_validate_agent_provider_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError, match="agent_provider"):
        validate_agent_provider(value)
