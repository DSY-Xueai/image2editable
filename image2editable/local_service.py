from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LocalServiceConfig:
    base_url: str
    model: str
    api_key: str | None


def load_config(
    environment: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
) -> LocalServiceConfig:
    values = _dotenv_values(
        Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    )
    values.update(os.environ if environment is None else environment)
    base_url = values.get("IMAGE2EDITABLE_LOCAL_BASE_URL", "").strip().rstrip("/")
    model = values.get("IMAGE2EDITABLE_LOCAL_MODEL", "").strip()
    if not base_url:
        raise RuntimeError(
            "Local model service is not configured: set IMAGE2EDITABLE_LOCAL_BASE_URL"
        )
    if not model:
        raise RuntimeError(
            "Local model service is not configured: set IMAGE2EDITABLE_LOCAL_MODEL"
        )
    api_key = values.get("IMAGE2EDITABLE_LOCAL_API_KEY", "").strip() or None
    return LocalServiceConfig(base_url=base_url, model=model, api_key=api_key)


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def complete(
    config: LocalServiceConfig,
    *,
    messages: list[dict[str, object]],
    timeout_seconds: int = 600,
) -> str:
    payload = json.dumps(
        {
            "model": config.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = Request(
        f"{config.base_url}/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        document = json.loads(response.read().decode("utf-8"))
    try:
        content = document["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Local model service returned no chat completion") from error
    if not isinstance(content, str):
        raise RuntimeError("Local model service returned a non-text completion")
    return content
