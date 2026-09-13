import cv2
import hashlib
import numpy as np
from PIL import Image

from scripts.visual_segment import _flat_stroke_prompt_mask, execute_component_actions


def test_signed_points_partition_connected_strokes_without_pixel_loss():
    source = np.full((180, 220, 3), 245, np.uint8)
    color = (80, 115, 145)
    cv2.line(source, (30, 90), (175, 25), color, 5)
    cv2.line(source, (30, 90), (175, 155), color, 5)
    positive, negative = [[75, 70], [145, 38]], [[75, 110], [145, 142]]
    ignored = np.zeros(source.shape[:2], bool)
    upper = _flat_stroke_prompt_mask(source, {'box': None, 'positive': positive, 'negative': negative}, ignored)
    lower = _flat_stroke_prompt_mask(source, {'box': None, 'positive': negative, 'negative': positive}, ignored)
    assert upper is not None and lower is not None
    assert not np.any(upper & lower)
    assert np.array_equal(upper | lower, np.all(source == color, axis=2))
    assert upper[25, 175] and not upper[155, 175]
    assert lower[155, 175] and not lower[25, 175]


def test_flat_connector_repair_rejects_surfaces_and_text():
    source = np.full((180, 220, 3), 245, np.uint8)
    prompt = {'box': None, 'positive': [[75, 70], [145, 38]], 'negative': [[75, 110], [145, 142]]}
    assert _flat_stroke_prompt_mask(source, prompt, np.zeros(source.shape[:2], bool)) is None
    cv2.line(source, (30, 90), (175, 25), (80, 115, 145), 5)
    cv2.line(source, (30, 90), (175, 155), (80, 115, 145), 5)
    assert _flat_stroke_prompt_mask(source, prompt, np.ones(source.shape[:2], bool)) is None


def test_connector_action_keeps_signed_partition_without_sam_or_gap_expansion(tmp_path):
    source = np.full((180, 220, 3), 245, np.uint8)
    color = (80, 115, 145)
    cv2.line(source, (30, 90), (175, 25), color, 5)
    cv2.line(source, (30, 90), (175, 155), color, 5)
    mask = np.all(source == color, axis=2)
    (tmp_path / 'masks').mkdir()
    path = tmp_path / 'masks/branch.png'
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    graph = {'nodes': [{'id': 'branch', 'kind': 'parent', 'parent_id': None,
                       'state': 'pending', 'mask': 'masks/branch.png',
                       'mask_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'bbox': [27, 22, 179, 159], 'z_index': 0, 'text_ids': []}]}
    positive, negative = [[75, 70], [145, 38]], [[75, 110], [145, 142]]
    output = tmp_path / 'out'
    def no_sam(**kwargs):
        raise AssertionError('Verified flat connector must not invoke SAM')
    result = execute_component_actions(source, graph, [{
        'action': 'retry_with_points', 'object_ids': ['branch'],
        'parameters': {'positive': (np.asarray(positive) / [219, 179]).tolist(),
                       'negative': (np.asarray(negative) / [219, 179]).tolist()},
        'confidence': .99, 'evidence': ['source.png'],
    }], input_dir=tmp_path, output_dir=output, sam_batch_runner=no_sam)
    actual = np.asarray(Image.open(output / result['nodes'][0]['mask'])) > 0
    expected = _flat_stroke_prompt_mask(source, {
        'box': None, 'positive': positive, 'negative': negative,
    }, np.zeros(source.shape[:2], bool))
    assert np.array_equal(actual, expected)
