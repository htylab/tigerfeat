# tigerfeat

`tigerfeat` is a light-weight wrapper that makes it convenient to initialise a vision
model and grab feature vectors without remembering each project's APIs. The
library now supports several backends – including
[timm](https://github.com/huggingface/pytorch-image-models),
[torchxrayvision](https://github.com/mlmed/torchxrayvision) and Hugging Face
vision models – while automatically choosing an available compute device
(`cuda`, `mps` or `cpu`).

## Installation

Install the latest release directly from the repository:

```bash
pip install https://github.com/htylab/tigerfeat/archive/main.zip
```

## Usage

```python
import tigerfeat

model = tigerfeat.init(model="timm@resnet50")
feature_vector = model.feat("/path/to/image.jpg")

print(feature_vector.shape)
# torch.Size([1536])
```

The `model` argument uses the `<backend>@<model_name>` syntax. Available
backends are:

- `timm` – any architecture supported by the `timm` library.
- `xray` – radiology models provided by `torchxrayvision` (requires the
  optional `torchxrayvision`, `scikit-image` and `torchvision` dependencies).
- `hf` – Hugging Face `transformers` vision checkpoints.

The `feat` method accepts either file paths or `PIL.Image` instances and always
returns a single feature vector on CPU. Use the `info()` helper to inspect the
resolved backend, model name, input size and device:

```python
details = model.info()
print(details)
# {"backend": "timm", "model_name": "resnet50", "input_size": (3, 224, 224), "device": "cuda"}
```

When targeting the X-ray backend, images are automatically converted to
single-channel inputs, and Hugging Face models make use of their corresponding
processors.
