# Third-Party Notices

## SAM 2.1

image2editable uses the official SAM 2.1 Python API from Meta for visual segmentation.

- Copyright (c) Meta Platforms, Inc. and affiliates.
- License: [Apache License 2.0](third_party/licenses/SAM2-APACHE-2.0.txt)
- Source: https://github.com/facebookresearch/sam2
- SAM 2.1 large checkpoint: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

The SAM 2 source code and checkpoint weights are not stored in this repository. The source is installed from the official repository; the large checkpoint is downloaded to the user's local model cache when first needed.

## Grounding DINO

image2editable uses the Grounding DINO tiny model through Hugging Face Transformers to generate open-vocabulary semantic object proposals.

- License: [Apache License 2.0](third_party/licenses/SAM2-APACHE-2.0.txt)
- Source: https://github.com/IDEA-Research/GroundingDINO
- Model: https://huggingface.co/IDEA-Research/grounding-dino-tiny

The Grounding DINO source code and model weights are not stored in this repository. Transformers downloads the model files to the user's local model cache when first needed. The linked Apache License 2.0 text is the standard license used by both SAM 2 and Grounding DINO.
