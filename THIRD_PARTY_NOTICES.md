# Third-party software and models

Open Laya does not distribute model weights. The optional download script retrieves them directly
from their publishers on Hugging Face.

## laya-omni code

- Source: <https://github.com/ZHEQIUSHUI/laya-omni>
- License: Apache License 2.0
- The optional dependency is pinned to commit `dfea79876b16c6ac9c5eafbf5d241b2359f1a4e1`.
- `laya_service/media.py` adapts the project's Apache-2.0 video-loading helper.

## laya-omni weights

- Source and terms: <https://huggingface.co/zheqiushui/laya-omni>
- The fusion weights are released for research and non-commercial use.
- The included audio encoder is derived from Qwen3-ASR-0.6B and is Apache-2.0 licensed.

## Laya multilingual weights

- Source: <https://huggingface.co/convaiinnovations/laya-multilingual>
- License: Apache License 2.0

## SigLIP 2 image encoder

- Source: <https://huggingface.co/google/siglip2-base-patch16-256>
- License: Apache License 2.0

Users are responsible for reviewing and complying with the terms shown by each publisher. Open
Laya's application code and configuration do not change the terms of third-party software or model
weights.
