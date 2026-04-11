from conftest import load_skill_module


page_planner = load_skill_module("page_planner")


def test_page_planner_accepts_layered_objects_and_vectors():
    plan = page_planner.build_page_plan(
        page_number=1,
        width_px=1000,
        height_px=1500,
        background_path="page.png",
        text_items=[],
        image_items=[],
        layered_objects=[
            {
                "object_type": "vector",
                "left": 0,
                "top": 0,
                "width": 10,
                "height": 10,
                "z_index": 0,
                "rebuildable": True,
                "must_fallback": False,
            }
        ],
        vector_instructions=[
            {
                "shape_type": "freeform",
                "left": 0,
                "top": 0,
                "width": 10,
                "height": 10,
                "payload": {"source": "vector_candidate"},
            }
        ],
    )
    assert len(plan.layered_objects) == 1
    assert len(plan.vector_instructions) == 1
