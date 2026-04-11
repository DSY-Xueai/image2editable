from conftest import load_skill_module


models = load_skill_module("models")
LayeredObject = models.LayeredObject
VectorInstruction = models.VectorInstruction


def test_layered_object_tracks_layer_metadata():
    obj = LayeredObject(
        object_type="vector",
        left=10.0,
        top=20.0,
        width=30.0,
        height=40.0,
        z_index=3,
        rebuildable=True,
        must_fallback=False,
    )
    assert obj.object_type == "vector"
    assert obj.z_index == 3


def test_vector_instruction_tracks_shape_payload():
    instruction = VectorInstruction(
        shape_type="freeform",
        left=1.0,
        top=2.0,
        width=3.0,
        height=4.0,
        payload={"points": [(0, 0), (1, 1)]},
    )
    assert instruction.shape_type == "freeform"
    assert instruction.payload["points"][1] == (1, 1)
