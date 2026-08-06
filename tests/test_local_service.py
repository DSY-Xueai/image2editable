from __future__ import annotations

import pytest


def test_load_config_requires_a_user_supplied_local_service() -> None:
    from image2editable.local_service import load_config

    with pytest.raises(RuntimeError, match="IMAGE2EDITABLE_LOCAL_BASE_URL"):
        load_config({})


def test_load_config_uses_user_selected_endpoint_and_model() -> None:
    from image2editable.local_service import load_config

    config = load_config(
        {
            "IMAGE2EDITABLE_LOCAL_BASE_URL": "http://127.0.0.1:8000/v1/",
            "IMAGE2EDITABLE_LOCAL_MODEL": "my-local-vlm",
            "IMAGE2EDITABLE_LOCAL_API_KEY": "secret",
        }
    )

    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.model == "my-local-vlm"
    assert config.api_key == "secret"


def test_load_config_reads_project_dotenv_without_overriding_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from image2editable.local_service import load_config

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "IMAGE2EDITABLE_LOCAL_BASE_URL=http://127.0.0.1:1234/v1\n"
        "IMAGE2EDITABLE_LOCAL_MODEL=from-dotenv\n"
        "IMAGE2EDITABLE_LOCAL_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )

    config = load_config(
        {"IMAGE2EDITABLE_LOCAL_MODEL": "from-environment"},
        dotenv_path=dotenv,
    )

    assert config.base_url == "http://127.0.0.1:1234/v1"
    assert config.model == "from-environment"
    assert config.api_key == "dotenv-key"


def test_complete_uses_openai_compatible_local_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from image2editable.local_service import LocalServiceConfig, complete

    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'

    def fake_open(request: object, timeout: int) -> Response:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = request.data
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("image2editable.local_service.urlopen", fake_open)
    result = complete(
        LocalServiceConfig("http://127.0.0.1:8000/v1", "my-vlm", "secret"),
        messages=[{"role": "user", "content": "plan"}],
    )

    assert result == '{"ok":true}'
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert b'"model": "my-vlm"' in captured["payload"]
