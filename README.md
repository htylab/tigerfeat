# tigerfeat

`tigerfeat` is a light-weight wrapper around [timm](https://github.com/huggingface/pytorch-image-models)
that makes it convenient to extract pooled feature vectors from any timm vision model.

## Installation

```bash
pip install https://github.com/htylab/tigerfeat/archive/main.zip
```

## Usage

```python
import tigerfeat

model = tigerfeat.init(model="dinov2_vits14")
feature_vector = model.feat("/path/to/image.jpg", pool="avgmax")
```

The `pool` argument supports:

- `"avg"` – Adaptive average pooling (default)
- `"max"` – Adaptive max pooling
- `"avgmax"` – Concatenation of average and max pooled features

Batch feature extraction is also supported by passing a list of paths or ``PIL.Image``
instances to ``model.feat``.
