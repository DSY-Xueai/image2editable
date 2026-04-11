from conftest import load_skill_module


layout_layers = load_skill_module("layout_layers")


def test_decompose_page_layers_marks_large_background_region():
    result = layout_layers.decompose_page_layers(
        page_width=1000,
        page_height=1500,
        objects=[
            {
                "object_type": "background",
                "left": 0,
                "top": 0,
                "width": 1000,
                "height": 1500,
            },
            {
                "object_type": "vector",
                "left": 20,
                "top": 20,
                "width": 100,
                "height": 50,
            },
        ],
    )
    assert len(result) == 2
    assert result[0].object_type == "background"
