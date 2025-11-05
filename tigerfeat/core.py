"""Core functionality for the TigerFeat feature extraction API."""

import numpy as np
import torch
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
        self.backend, self.model_name = self._parse_model_identifier(self.config.model)
        self.transform = None
        self.input_size = None
        self.data_config = None

        self._initialise_model()

    @staticmethod
    def _parse_model_identifier(identifier):
        if not isinstance(identifier, str) or "@" not in identifier:
            raise ValueError(
                "Model identifier must be a string in the format '<backend>@<model_name>'."
            )
        backend, model_name = identifier.split("@", 1)
        backend = backend.strip().lower()
        model_name = model_name.strip()
        if not backend or not model_name:
            raise ValueError("Both backend and model name must be provided in the model identifier.")
        return backend, model_name

    def _initialise_model(self):
        """Initialise model, backend, and transforms based on the requested config."""

        if self.backend == "timm":
            self._initialise_timm_model()
        elif self.backend == "xray":
            self._initialise_xray_model()
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
        self.input_size = (1, int(size), int(size))
        self.transform = tv_transforms.Compose(
            [
                _xrv.datasets.XRayCenterCrop(),
                _xrv.datasets.XRayResizer(int(size)),
            ]
        )

    @staticmethod
    def _resolve_device(device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _prepare_batch(self, images):
        if self.backend == "timm":
            return self._prepare_timm_batch(images)
        if self.backend == "xray":
            return self._prepare_xray_batch(images)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _prepare_timm_batch(self, images):
        processed = []
        for image in images:
            if isinstance(image, str):
                with Image.open(image) as img:
                    img = img.convert("RGB")
                    processed.append(self.transform(img))
            elif isinstance(image, Image.Image):
                processed.append(self.transform(image.convert("RGB")))
            else:
                raise TypeError(
                    "Images must be provided as file paths or PIL.Image instances; received type %r." % (type(image),)
                )
        batch = torch.stack(processed, dim=0)
        return batch.to(self.device)

    def _prepare_xray_batch(self, images):
        processed = []
        for image in images:
            if isinstance(image, str):
                img = skio.imread(image)
            elif isinstance(image, Image.Image):
                img = np.array(image)
            else:
                raise TypeError(
                    "Images must be provided as file paths or PIL.Image instances; "
                    "received type %r." % (type(image),)
                )

            img = _xrv.datasets.normalize(img, 255)
            if img.ndim == 2:
                img = img[None, ...]
            elif img.ndim == 3:
                img = img.mean(axis=2, keepdims=False)[None, ...]
            else:
                raise ValueError(
                    "Radiograph inputs must be 2D or RGB images; received array with "
                    f"shape {img.shape}."
                )

            img = img.astype("float32", copy=False)
            transformed = self.transform(img)
            if isinstance(transformed, torch.Tensor):
                tensor = transformed.float()
            else:
                tensor = torch.from_numpy(np.asarray(transformed)).float()

            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            processed.append(tensor)

        batch = torch.stack(processed, dim=0)
        return batch.to(self.device)

    def _forward_timm_features(self, batch):
        with torch.no_grad():
            features = self.model.forward_features(batch)

            if isinstance(features, (list, tuple)):
                features = features[-1]
            elif isinstance(features, dict):
                features = self._select_features_from_dict(features)

            if hasattr(self.model, "forward_head"):
                try:
                    features = self.model.forward_head(features, pre_logits=True)
                except TypeError:
                    features = self.model.forward_head(features)

            if isinstance(features, (list, tuple)):
                features = features[-1]
            elif isinstance(features, dict):
                features = self._select_features_from_dict(features)

        return features

    def _forward_xray_features(self, batch):
        with torch.no_grad():
            features = self.model.features(batch)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if isinstance(features, dict):
                features = self._select_features_from_dict(features)

            if isinstance(features, torch.Tensor) and features.ndim == 4:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            if isinstance(features, torch.Tensor):
                features = features.flatten(start_dim=1)

        return features

    @staticmethod
    def _select_features_from_dict(features_dict):
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

        for value in reversed(list(features_dict.values())):
            ndim = getattr(value, "ndim", None)
            if ndim is None:
                continue
            if ndim > 1 and getattr(value, "numel", lambda: 0)() > 1:
                return value

        return next(reversed(features_dict.values()))

    def _pool_features(self, features, pool):
        if isinstance(features, torch.Tensor) and features.ndim == 2 and features.shape[1] > 1:
            return features

        if isinstance(features, torch.Tensor) and features.ndim == 4:
            pool = pool.lower()
            if pool not in {"avg", "max", "avgmax"}:
                raise ValueError("pool must be one of 'avg', 'max', or 'avgmax'.")

            avg_feat = max_feat = None
            if pool in {"avg", "avgmax"}:
                avg_feat = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            if pool in {"max", "avgmax"}:
                max_feat = torch.nn.functional.adaptive_max_pool2d(features, (1, 1))

            outputs = []
            if avg_feat is not None:
                outputs.append(avg_feat)
            if max_feat is not None:
                outputs.append(max_feat)

            combined = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
            return combined.flatten(start_dim=1)

        if isinstance(features, torch.Tensor) and features.ndim >= 2:
            return features.reshape(features.shape[0], -1)

        if isinstance(features, torch.Tensor):
            return features.unsqueeze(0)

        return features

    def feat(self, image, pool="avg"):
        """Extract a feature vector from an image or a batch of images."""

        if isinstance(image, (str, Image.Image)):
            images = [image]
            single = True
        elif isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("Image sequence must not be empty.")
            images = image
            single = False
        else:
            raise TypeError("Image input must be a path, PIL.Image, or a sequence of those types.")

        batch = self._prepare_batch(images)

        if self.backend == "timm":
            features = self._forward_timm_features(batch)
        elif self.backend == "xray":
            features = self._forward_xray_features(batch)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        pooled = self._pool_features(features, pool)
        pooled = pooled.detach().cpu()

        if single:
            return pooled.squeeze(0)
        return pooled

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
