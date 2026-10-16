from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from .config import ROOT, sha256_file, sha256_json
from .data import DatasetStore, load_manifest
from .utils import atomic_save_numpy, atomic_write_json, task_complete, task_context


class ManifestImageDataset(Dataset):
    def __init__(self, dataset: str, records: pd.DataFrame, image_size: int):
        self.dataset = dataset
        self.records = records.reset_index(drop=True)
        self.store: DatasetStore | None = None
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        if self.store is None:
            self.store = DatasetStore(download=False)
        row = self.records.iloc[index]
        image = self.store.image(
            self.dataset, str(row.source_split), int(row.source_index)
        )
        return self.transform(image), int(index)


def load_dinov2(device: str) -> torch.nn.Module:
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True, trust_repo=True
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def prefetch_dinov2() -> None:
    model = load_dinov2("cpu")
    del model


@torch.inference_mode()
def extract_dataset(dataset: str, device: str, config: dict, resume: bool = True) -> None:
    feature_cfg = config["features"]
    manifest_path = ROOT / "data" / "manifests" / f"{dataset}.csv"
    input_hash = sha256_json(
        {
            "manifest": sha256_file(manifest_path),
            "encoder": feature_cfg["encoder"],
            "image_size": feature_cfg["image_size"],
            "dimension": feature_cfg["dimension"],
        }
    )
    task_id = f"embeddings-{dataset}"
    if resume and task_complete(task_id, input_hash):
        return
    with task_context(task_id, input_hash):
        model = load_dinov2(device)
        manifest = load_manifest(dataset)
        manifest = manifest[manifest["split"].isin(["train", "val", "test"])].copy()
        output_dir = ROOT / "data" / "embeddings" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        split_meta = {}
        for split in ("train", "val", "test"):
            records = manifest[manifest["split"] == split].copy().reset_index(drop=True)
            ds = ManifestImageDataset(dataset, records, int(feature_cfg["image_size"]))
            loader = DataLoader(
                ds,
                batch_size=int(feature_cfg["batch_size"]),
                shuffle=False,
                num_workers=min(6, os.cpu_count() or 1),
                pin_memory=device.startswith("cuda"),
                persistent_workers=len(ds) > 0,
            )
            features = np.empty(
                (len(records), int(feature_cfg["dimension"])), dtype=np.float32
            )
            start = time.perf_counter()
            for images, positions in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    dtype=torch.float16,
                    enabled=device.startswith("cuda"),
                ):
                    output = model.forward_features(images)["x_norm_clstoken"]
                features[positions.numpy()] = output.float().cpu().numpy()
            x_path = output_dir / f"{split}_X.npy"
            atomic_save_numpy(x_path, features)
            records[["sample_id", "source_split", "source_index"]].to_csv(
                output_dir / f"{split}_ids.csv", index=False
            )
            if split != "test":
                atomic_save_numpy(
                    output_dir / f"{split}_y.npy",
                    records["label"].to_numpy(dtype=np.int64),
                )
            split_meta[split] = {
                "count": len(records),
                "x_sha256": sha256_file(x_path),
                "elapsed_seconds": time.perf_counter() - start,
                "labels_written": split != "test",
            }
        metadata = {
            "dataset": dataset,
            "device": device,
            "encoder": feature_cfg["encoder"],
            "embedding_dimension": feature_cfg["dimension"],
            "manifest_sha256": sha256_file(manifest_path),
            "splits": split_meta,
        }
        atomic_write_json(output_dir / "metadata.json", metadata)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_embeddings(dataset: str, split: str, allow_test: bool = False):
    if split == "test" and not allow_test:
        raise PermissionError(
            "Test embeddings are isolated. Use the dedicated post-freeze evaluation stage."
        )
    root = ROOT / "data" / "embeddings" / dataset
    x = np.load(root / f"{split}_X.npy", mmap_mode="r")
    ids = pd.read_csv(root / f"{split}_ids.csv")
    if split == "test":
        y = None
    else:
        y = np.load(root / f"{split}_y.npy")
    return x, y, ids

