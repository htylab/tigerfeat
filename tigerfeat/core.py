"""Core functionality for the TigerFeat feature extraction API."""

import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

__all__ = ["TigerFeatModel", "init"]


class TigerFeatConfig(object):
    """Simple configuration container for :class:`TigerFeatModel`."""

    def __init__(self, model="dinov2", pretrained=True, device=None, transform_kwargs=None):
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
    """A lightweight wrapper that exposes a ``feat`` method for timm models."""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
        elif kwargs:
            raise ValueError("Provide either a config object or keyword arguments, not both.")

        self.config = config
        self.device = self._resolve_device(config.device)

        self.model = timm.create_model(
            config.model,
            pretrained=config.pretrained,
        )
        self.model.eval()
        self.model.to(self.device)

        data_config = resolve_data_config({}, model=self.model)
        transform_kwargs = dict(config.transform_kwargs)
        transform_kwargs.setdefault("is_training", False)
        self.transform = create_transform(**data_config, **transform_kwargs)

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

    def _forward_features(self, batch):
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
        features = self._forward_features(batch)
        pooled = self._pool_features(features, pool)
        pooled = pooled.detach().cpu()

        if single:
            return pooled.squeeze(0)
        return pooled


def init(**kwargs):
    """Initialise a :class:`TigerFeatModel` with the provided configuration."""

    config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
    return TigerFeatModel(config)
