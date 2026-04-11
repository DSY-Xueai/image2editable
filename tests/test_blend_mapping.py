from conftest import load_skill_module


blend_mapping = load_skill_module("blend_mapping")


def test_group_blend_candidates_marks_complex_group_for_fallback():
    groups = blend_mapping.group_blend_candidates(
        [
            {"group_id": "g1", "effect_type": "opacity", "must_fallback": False},
            {"group_id": "g1", "effect_type": "blend_mode", "must_fallback": True},
        ]
    )
    assert groups[0]["must_fallback"] is True
