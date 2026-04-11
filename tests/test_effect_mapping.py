from conftest import load_skill_module


effect_mapping = load_skill_module("effect_mapping")


def test_map_effect_type_accepts_supported_shadow():
    result = effect_mapping.map_effect_type(
        {"effect_type": "shadow", "confidence": 0.95, "complexity": "simple"}
    )
    assert result["mapped"] is True
    assert result["effect_type"] == "shadow"


def test_map_effect_type_rejects_unsupported_blend_mode():
    result = effect_mapping.map_effect_type(
        {"effect_type": "blend_mode", "confidence": 0.95, "complexity": "complex"}
    )
    assert result["mapped"] is False
