"""Core functionality for the TigerFeat feature extraction API."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:  # pragma: no cover - optional dependency for X-ray backend
    import torchxrayvision as _xrv
except ImportError:  # pragma: no cover - handled lazily when backend is required
    _xrv = None

try:  # pragma: no cover - optional dependency for X-ray backend
    from skimage import io as skio
except ImportError:  # pragma: no cover - handled lazily when backend is required
    skio = None

try:  # pragma: no cover - torchvision is required only for X-ray models
    import torchvision.transforms as tv_transforms
except ImportError:  # pragma: no cover - handled lazily when backend is required
    tv_transforms = None

__all__ = ["TigerFeatModel", "init"]


class TigerFeatConfig(object):
    """Simple configuration container for :class:`TigerFeatModel`."""

    def __init__(
        self,
        model="timm@dinov2",
        pretrained=True,
        device=None,
        transform_kwargs=None,
    ):
        self.model = model
        self.pretrained = pretrained
        self.device = device
        self.transform_kwargs = transform_kwargs or {}


def _normalise_model_kwargs(kwargs):
    if "model_name" in kwargs and "model" not in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["model"] = new_kwargs.pop("model_name")
        return new_kwargs
    return kwargs


class TigerFeatModel(object):
    """A lightweight wrapper that exposes a ``feat`` method for vision models."""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
        elif kwargs:
            raise ValueError("Provide either a config object or keyword arguments, not both.")

        self.config = config
        self.device = self._resolve_device(config.device)

        if "@" not in self.config.model:
            raise ValueError(
                "Model identifier must be provided in the '<backend>@<model_name>' format."
            )
        backend, model_name = self.config.model.split("@", 1)
        self.backend = backend.strip().lower()
        self.model_name = model_name.strip()
        if not self.backend or not self.model_name:
            raise ValueError(
                "Model identifier must include both backend and model name separated by '@'."
            )

        self.transform = None
        self.input_size = None
        self.data_config = None
        self.processor = None
        self.model = None

        self._initialise_model()

    def _initialise_model(self):
        """Initialise model, backend, and transforms based on the requested config."""

        if self.backend == "timm":
            self._initialise_timm_model()
        elif self.backend == "xray":
            self._initialise_xray_model()
        elif self.backend == "hf":
            self._initialise_hf_model()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _initialise_timm_model(self):
        self.model = timm.create_model(
            self.model_name,
            pretrained=self.config.pretrained,
        )
        self.model.eval()
        self.model.to(self.device)

        self.data_config = resolve_data_config({}, model=self.model)
        transform_kwargs = dict(self.config.transform_kwargs)
        transform_kwargs.setdefault("is_training", False)
        self.transform = create_transform(**self.data_config, **transform_kwargs)
        input_size = self.data_config.get("input_size")
        self.input_size = tuple(input_size) if input_size is not None else None

    def _initialise_xray_model(self):
        """Initialise the torchxrayvision backend."""

        if _xrv is None:
            raise ValueError(
                "X-ray backend requested but torchxrayvision is not installed."
            )
        if skio is None:
            raise ValueError(
                "X-ray backend requires scikit-image; please install it to load radiograph images."
            )
        if tv_transforms is None:
            raise ValueError(
                "X-ray backend requires torchvision for preprocessing transforms."
            )

        try:
            self.model = _xrv.models.get_model(self.model_name)
        except Exception as exc:
            raise ValueError(
                f"Unknown X-ray model '{self.model_name}'."
            ) from exc

        self.model.eval()
        self.model.to(self.device)

        size = getattr(self.model, "img_size", 224)
        if isinstance(size, (tuple, list)):
            size = size[0]
        size = int(size)
        self.input_size = (1, size, size)
        self.transform = tv_transforms.Compose(
            [
                _xrv.datasets.XRayCenterCrop(),
                _xrv.datasets.XRayResizer(size),
            ]
        )

    def _initialise_hf_model(self):
        try:  # pragma: no cover - optional dependency
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - handled lazily
            raise ValueError(
                "Hugging Face backend requested but transformers is not installed."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        processor = getattr(self.processor, "image_processor", self.processor)
        size = getattr(processor, "size", None)
        channels = getattr(processor, "num_channels", None)
        if channels is None:
            image_mean = getattr(processor, "image_mean", None)
            if isinstance(image_mean, (list, tuple)) and image_mean:
                channels = len(image_mean)
        if channels is None:
            channels = 3

        resolved_size = None
        if isinstance(size, dict):
            height = size.get("height")
            width = size.get("width")
            shortest = (
                size.get("shortest_edge")
                or size.get("shortest_side")
                or size.get("shortest")
            )
            if height is not None and width is not None:
                resolved_size = (channels, int(height), int(width))
            elif shortest is not None:
                resolved_size = (channels, int(shortest), int(shortest))
        elif isinstance(size, (tuple, list)) and len(size) >= 2:
            resolved_size = (channels, int(size[-2]), int(size[-1]))
        elif isinstance(size, int):
            resolved_size = (channels, int(size), int(size))

        self.input_size = resolved_size or (channels, 224, 224)

    @staticmethod
    def _resolve_device(device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _prepare_timm_image(self, image):
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            with Image.open(image) as img_file:
                img = img_file.convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def _prepare_xray_image(self, image):
        if _xrv is None:
            raise ValueError("X-ray backend requested but torchxrayvision is not installed.")
        if isinstance(image, str):
            if skio is None:
                raise ValueError(
                    "X-ray backend requires scikit-image to read image files."
                )
            img = skio.imread(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            raise TypeError(
                "Images must be provided as file paths or PIL.Image instances; "
                f"received type {type(image)!r}."
            )

        img = _xrv.datasets.normalize(img, 255)
        if img.ndim == 2:
            img = img[None, ...]
        else:
            img = img.mean(2)[None, ...]
        img = img.astype("float32", copy=False)

        transformed = self.transform(img)
        if isinstance(transformed, torch.Tensor):
            tensor = transformed.float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(np.asarray(transformed, dtype="float32")).unsqueeze(0)
        return tensor.to(self.device)

    def _prepare_hf_image(self, image):
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            with Image.open(image) as img_file:
                img = img_file.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.to(self.device)

    @staticmethod
    def _select_features_from_dict(features_dict):
        if not features_dict:
            raise ValueError("Feature dictionary is empty and cannot be reduced.")
        preferred_keys = [
            "x_norm_clstoken",
            "x_norm_cls_token",
            "cls_token",
            "class_token",
            "image_features",
            "pooled_features",
            "features",
        ]

        for key in preferred_keys:
            if key in features_dict:
                return features_dict[key]

        return next(reversed(features_dict.values()))

    def _forward_timm(self, tensor):
        with torch.no_grad():
            feats = self.model.forward_features(tensor)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            elif isinstance(feats, dict):
                feats = self._select_features_from_dict(feats)
            if hasattr(self.model, "forward_head"):
                try:
                    feats = self.model.forward_head(feats, pre_logits=True)
                except TypeError:
                    feats = self.model.forward_head(feats)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            elif isinstance(feats, dict):
                feats = self._select_features_from_dict(feats)
            if isinstance(feats, torch.Tensor) and feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
        return feats

    def _forward_xray(self, tensor):
        with torch.no_grad():
            feat = self.model.features(tensor)
            feat = nn.AdaptiveAvgPool2d((1, 1))(feat)
            feat = feat.view(feat.size(0), -1)
        return feat

    def _forward_hf(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output"):
                feat = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Unknown output structure for HF model.")
        return feat

    def feat(self, image):
        if not isinstance(image, (str, Image.Image)):
            raise TypeError("Image input must be a file path or PIL.Image instance.")
        if self.backend == "timm":
            tensor = self._prepare_timm_image(image)
            feats = self._forward_timm(tensor)
        elif self.backend == "xray":
            tensor = self._prepare_xray_image(image)
            feats = self._forward_xray(tensor)
        elif self.backend == "hf":
            inputs = self._prepare_hf_image(image)
            feats = self._forward_hf(inputs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        return feats.detach().cpu().squeeze(0)

    def info(self):
        """Return a dictionary describing the current model instance."""

        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "input_size": self.input_size,
            "device": str(self.device),
        }


def init(**kwargs):
    """Initialise a :class:`TigerFeatModel` with the provided configuration."""

    config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
    return TigerFeatModel(config)
