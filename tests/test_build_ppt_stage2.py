from types import SimpleNamespace

from PIL import Image

from conftest import load_skill_module


build_ppt = load_skill_module("build_ppt")


def test_build_presentation_writes_only_verified_text_blocks(tmp_path):
    background_path = tmp_path / "page.png"
    Image.new("RGB", (100, 200), "white").save(background_path)
    output_path = tmp_path / "result.pptx"
    page = SimpleNamespace(
        page_number=1,
        width_px=100,
        height_px=200,
        background_path=str(background_path),
        text_blocks=[
            SimpleNamespace(
                text="verified",
                left=0,
                top=0,
                width=10,
                height=10,
                font_size=12,
                color="#000000",
                alignment="left",
                confidence=0.99,
                fit_verified=True,
            ),
            SimpleNamespace(
                text="rejected",
                left=20,
                top=20,
                width=10,
                height=10,
                font_size=12,
                color="#000000",
                alignment="left",
                confidence=0.99,
                fit_verified=False,
            ),
        ],
        image_blocks=[],
        effect_blocks=[SimpleNamespace(kind="shadow")],
    )

    build_ppt.build_presentation([page], output_path)

    assert output_path.exists()
