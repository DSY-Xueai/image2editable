from types import SimpleNamespace

from conftest import load_skill_module


effect_mapping = load_skill_module("effect_mapping")


def test_map_effect_type_allows_simple_high_confidence_direct_effect():
    effect = SimpleNamespace(effect_type="shadow", confidence=0.9, complexity="simple")

    mapped = effect_mapping.map_effect_type(effect)

    assert mapped == "shadow"


def test_map_effect_type_rejects_non_direct_effect():
    effect = SimpleNamespace(effect_type="blur", confidence=1.0, complexity="simple")

    assert effect_mapping.map_effect_type(effect) is None


def test_map_effect_type_rejects_low_confidence_or_complex_effect():
    low_confidence = SimpleNamespace(effect_type="glow", confidence=0.89, complexity="simple")
    complex_effect = SimpleNamespace(effect_type="glow", confidence=0.95, complexity="complex")

    assert effect_mapping.map_effect_type(low_confidence) is None
    assert effect_mapping.map_effect_type(complex_effect) is None


def test_filter_mappable_effects_keeps_only_mapped_effects():
    effects = [
        SimpleNamespace(effect_type="shadow", confidence=0.95, complexity="simple"),
        SimpleNamespace(effect_type="blur", confidence=0.99, complexity="simple"),
        SimpleNamespace(effect_type="opacity", confidence=0.91, complexity="simple"),
    ]

    filtered = effect_mapping.filter_mappable_effects(effects)

    assert filtered == ["shadow", "opacity"]
