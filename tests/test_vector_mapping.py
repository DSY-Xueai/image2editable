from conftest import load_skill_module


vector_mapping = load_skill_module("vector_mapping")


def test_map_vector_candidate_accepts_simple_independent_shape():
    result = vector_mapping.map_vector_candidate(
        {
            "object_type": "vector",
            "complexity": "simple",
            "rebuildable": True,
            "must_fallback": False,
        }
    )
    assert result is not None
    assert result.shape_type == "freeform"


def test_map_vector_candidate_rejects_complex_fragmented_shape():
    result = vector_mapping.map_vector_candidate(
        {
            "object_type": "vector",
            "complexity": "fragmented",
            "rebuildable": True,
            "must_fallback": False,
        }
    )
    assert result is None
