from __future__ import annotations


AGENT_PROVIDERS = frozenset({"host", "local"})
MAX_REPAIR_ROUNDS = 5


def validate_agent_provider(value: object) -> str:
    if type(value) is not str or value not in AGENT_PROVIDERS:
        raise ValueError(
            "Invalid agent_provider; expected one of: host, local"
        )
    return value
