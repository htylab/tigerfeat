"""Core functionality for the TigerFeat feature extraction API."""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    import torchxrayvision as _xrv
except ImportError:
    _xrv = None

try:
    from skimage import io as skio
except ImportError:
    skio = None

try:
    import torchvision.transforms as tv_transforms
except ImportError:
    tv_transforms = None


class TigerFeatConfig(object):
    """Simple configuration container for :class:`TigerFeatModel`."""

    def __init__(
        self,
        model="timm@resnet50",
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
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
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

    # === 改動：開檔錯誤處理 + 僅支援檔案路徑 ===

    def _safe_open_pil(self, path):
        if not isinstance(path, str):
            raise TypeError("Image input must be a file path string.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to open image {path}: {e}")

    def _prepare_timm_image(self, path):
        img = self._safe_open_pil(path)
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def _prepare_xray_image(self, path):
        if not isinstance(path, str):
            raise TypeError("Image input must be a file path string.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            img = skio.imread(path)
        except Exception as e:
            raise IOError(f"Failed to open image {path}: {e}")
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

    def _prepare_hf_image(self, path):
        img = self._safe_open_pil(path)
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.to(self.device)

    # === 原始 preferred_keys 完整保留 ===

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

    # === forward 與 batch 保留原樣，只加 filelist 回傳 ===

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

    # === feat(): 新增錯誤處理與回傳 filelist ===

    def feat(self, image_path):
        if not isinstance(image_path, str):
            raise TypeError("Image input must be a file path string.")
        try:
            if self.backend == "timm":
                tensor = self._prepare_timm_image(image_path)
                feats = self._forward_timm(tensor)
            elif self.backend == "xray":
                tensor = self._prepare_xray_image(image_path)
                feats = self._forward_xray(tensor)
            elif self.backend == "hf":
                inputs = self._prepare_hf_image(image_path)
                feats = self._forward_hf(inputs)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            return feats.detach().cpu().squeeze(0).numpy()
        except Exception as e:
            print(f"[Warning] Skipped {image_path}: {e}")
            return np.empty((0,), dtype=np.float32)

    # === feat_batch(): 保留 DataLoader 結構，增加錯誤防護 + filelist ===

    def feat_batch(
        self,
        images,
        batch_size=16,
        num_workers=None,
        pin_memory=True,
        show_progress=None,
        use_dataloader=None,
        dataloader_threshold=1000,
    ):
        if isinstance(images, str):
            images = [images]
        elif not isinstance(images, (list, tuple)):
            raise TypeError("Input must be a list/tuple of file paths.")

        images = [p for p in images if isinstance(p, str)]
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32), []

        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            _tqdm = None
        if show_progress is None:
            show_progress = n >= dataloader_threshold

        if use_dataloader is None:
            use_dataloader = (self.backend in ("timm", "xray")) and (n >= dataloader_threshold)

        if num_workers is None:
            num_workers = 8 if self.device.type == "cuda" else 0

        all_feats = []
        filelist = []

        if self.backend == "hf":
            for start in (_tqdm(range(0, n, batch_size), desc="Extracting") if (show_progress and _tqdm) else range(0, n, batch_size)):
                chunk = images[start:start + batch_size]
                valid_imgs, valid_files = [], []
                for p in chunk:
                    try:
                        valid_imgs.append(self._safe_open_pil(p))
                        valid_files.append(p)
                    except Exception as e:
                        print(f"[Warning] Skipped {p}: {e}")
                if not valid_imgs:
                    continue
                inputs = self.processor(images=valid_imgs, return_tensors="pt").to(self.device)
                feats = self._forward_hf(inputs)
                all_feats.append(feats.detach().cpu())
                filelist.extend(valid_files)
        else:
            if use_dataloader:
                from torch.utils.data import Dataset, DataLoader

                if self.backend == "timm":
                    transform = self.transform

                    class _TimmDataset(Dataset):
                        def __init__(self, paths): self.paths = paths
                        def __len__(self): return len(self.paths)
                        def __getitem__(self, idx):
                            path = self.paths[idx]
                            try:
                                with Image.open(path) as f:
                                    img = f.convert("RGB")
                                return transform(img), path
                            except Exception as e:
                                print(f"[Warning] Skipped {path}: {e}")
                                return None

                elif self.backend == "xray":
                    xray_transform = self.transform

                    class _XrayDataset(Dataset):
                        def __init__(self, paths): self.paths = paths
                        def __len__(self): return len(self.paths)
                        def __getitem__(self, idx):
                            path = self.paths[idx]
                            try:
                                img = skio.imread(path)
                                img = _xrv.datasets.normalize(img, 255)
                                if img.ndim == 2:
                                    img = img[None, ...]
                                else:
                                    img = img.mean(2)[None, ...]
                                img = img.astype("float32", copy=False)
                                out = xray_transform(img)
                                t = torch.from_numpy(np.asarray(out, dtype="float32"))
                                return t, path
                            except Exception as e:
                                print(f"[Warning] Skipped {path}: {e}")
                                return None

                else:
                    raise ValueError(f"Unsupported backend for DataLoader: {self.backend}")

                class _Collate:
                    def __call__(self, batch):
                        batch = [b for b in batch if b is not None]
                        if not batch:
                            return torch.empty(0), []
                        tensors, paths = zip(*batch)
                        return torch.stack(tensors), list(paths)

                loader = DataLoader(
                    (_TimmDataset(images) if self.backend == "timm" else _XrayDataset(images)),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory if self.device.type == "cuda" else False,
                    collate_fn=_Collate(),
                )

                iterator = _tqdm(loader, desc="Extracting") if (show_progress and _tqdm) else loader
                with torch.no_grad():
                    for batch, paths in iterator:
                        if batch.numel() == 0:
                            continue
                        batch = batch.to(self.device)
                        feats = self._forward_timm(batch) if self.backend == "timm" else self._forward_xray(batch)
                        all_feats.append(feats.detach().cpu())
                        filelist.extend(paths)
            else:
                rng = _tqdm(range(0, n, batch_size), desc="Extracting") if (show_progress and _tqdm) else range(0, n, batch_size)
                for start in rng:
                    chunk = images[start:start + batch_size]
                    tensors, valids = [], []
                    for p in chunk:
                        try:
                            if self.backend == "timm":
                                tensors.append(self._prepare_timm_image(p))
                            else:
                                tensors.append(self._prepare_xray_image(p))
                            valids.append(p)
                        except Exception as e:
                            print(f"[Warning] Skipped {p}: {e}")
                    if not tensors:
                        continue
                    batch = torch.cat(tensors, dim=0)
                    feats = self._forward_timm(batch) if self.backend == "timm" else self._forward_xray(batch)
                    all_feats.append(feats.detach().cpu())
                    filelist.extend(valids)

        if not all_feats:
            return np.empty((0,), dtype=np.float32), []

        feats = torch.cat(all_feats, dim=0).numpy()
        return feats, filelist

    def info(self):
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "input_size": self.input_size,
            "device": str(self.device),
        }


def init(**kwargs):
    config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
    return TigerFeatModel(config)


def save_parquet(path, features, filelist, compression="snappy"):
    """
    將特徵與檔案名稱儲存成單一 Parquet 檔案。

    Parameters
    ----------
    path : str
        輸出檔案路徑，例如 "features.parquet"。
    features : np.ndarray
        特徵矩陣 (N, D)。
    filelist : list[str]
        對應的影像檔名清單。
    compression : str, default="snappy"
        Parquet 壓縮方式，可選 "snappy", "brotli", "gzip"。
    """
    import pandas as pd
    import numpy as np
    import os

    if not isinstance(features, np.ndarray):
        raise TypeError("features 必須是 numpy.ndarray")
    if not isinstance(filelist, (list, tuple)):
        raise TypeError("filelist 必須是 list 或 tuple")
    if features.shape[0] != len(filelist):
        raise ValueError(f"features 筆數 ({features.shape[0]}) 與 filelist 長度 ({len(filelist)}) 不符。")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    df = pd.DataFrame({
        "file": filelist,
        "feature": [f.astype(np.float32).tolist() for f in features],
    })
    df.to_parquet(path, index=False, compression=compression)
    print(f"[save_parquet] Wrote {len(filelist)} records → {path}")
