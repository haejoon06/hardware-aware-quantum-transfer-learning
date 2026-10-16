from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image

from .config import ROOT


DATASETS = ("cifar10", "eurosat", "pathmnist")
CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_BYTES = 170_498_071
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"


def _md5(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar10_archive(root: str | Path, max_retries: int = 30) -> Path:
    """Download CIFAR-10 with byte-range resume and verify its official checksum.

    Torchvision opens an incomplete archive in write mode and therefore restarts a
    slow transfer after a connection failure. This downloader preserves received
    bytes and resumes them on the next pipeline invocation or retry.
    """

    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "cifar-10-python.tar.gz"
    if archive.exists() and archive.stat().st_size == CIFAR10_ARCHIVE_BYTES:
        if _md5(archive) == CIFAR10_ARCHIVE_MD5:
            return archive
        raise RuntimeError(f"CIFAR-10 archive has the expected size but an invalid MD5: {archive}")
    if archive.exists() and archive.stat().st_size > CIFAR10_ARCHIVE_BYTES:
        raise RuntimeError(f"CIFAR-10 archive is larger than the official file: {archive}")

    for attempt in range(1, max_retries + 1):
        offset = archive.stat().st_size if archive.exists() else 0
        if offset == CIFAR10_ARCHIVE_BYTES:
            break
        request = Request(
            CIFAR10_URL,
            headers={
                "Range": f"bytes={offset}-",
                "User-Agent": "Article09-Reproducible-Pipeline/1.0",
            },
        )
        try:
            with urlopen(request, timeout=90) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    raise RuntimeError(
                        f"Server refused byte-range resume at {offset} bytes (HTTP {status})"
                    )
                mode = "ab" if offset else "wb"
                with archive.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        handle.flush()
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"CIFAR-10 download did not complete after {max_retries} resumable attempts"
                ) from exc
            time.sleep(min(30, 2 * attempt))

    actual_size = archive.stat().st_size if archive.exists() else 0
    if actual_size != CIFAR10_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Incomplete CIFAR-10 archive: {actual_size}/{CIFAR10_ARCHIVE_BYTES} bytes"
        )
    actual_md5 = _md5(archive)
    if actual_md5 != CIFAR10_ARCHIVE_MD5:
        raise RuntimeError(
            f"CIFAR-10 checksum mismatch: expected {CIFAR10_ARCHIVE_MD5}, got {actual_md5}"
        )
    return archive


class DatasetStore:
    """Lazy access to canonical source images without duplicating raw datasets."""

    def __init__(self, root: str | Path | None = None, download: bool = False):
        self.root = Path(root) if root else ROOT / "data" / "raw"
        self.download = download
        self._objects: dict[tuple[str, str], Any] = {}

    def _cifar(self, split: str):
        from torchvision.datasets import CIFAR10

        key = ("cifar10", split)
        if key not in self._objects:
            if self.download:
                ensure_cifar10_archive(self.root / "cifar10")
            self._objects[key] = CIFAR10(
                root=str(self.root / "cifar10"),
                train=split == "train",
                download=self.download,
            )
        return self._objects[key]

    def _eurosat(self):
        from torchvision.datasets import EuroSAT

        key = ("eurosat", "all")
        if key not in self._objects:
            self._objects[key] = EuroSAT(
                root=str(self.root / "eurosat"), download=self.download
            )
        return self._objects[key]

    def _pathmnist(self, split: str):
        import medmnist

        key = ("pathmnist", split)
        if key not in self._objects:
            pathmnist_root = self.root / "pathmnist"
            pathmnist_root.mkdir(parents=True, exist_ok=True)
            self._objects[key] = medmnist.PathMNIST(
                split=split,
                root=str(pathmnist_root),
                download=self.download,
                size=28,
                as_rgb=True,
            )
        return self._objects[key]

    def records(self, dataset: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if dataset == "cifar10":
            for split in ("train", "test"):
                obj = self._cifar(split)
                for index, label in enumerate(obj.targets):
                    rows.append(
                        {
                            "dataset": dataset,
                            "sample_id": f"cifar10-{split}-{index:05d}",
                            "source_split": split,
                            "source_index": index,
                            "label": int(label),
                            "class_name": obj.classes[int(label)],
                        }
                    )
        elif dataset == "eurosat":
            obj = self._eurosat()
            for index, (_, label) in enumerate(obj.samples):
                rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": f"eurosat-all-{index:05d}",
                        "source_split": "all",
                        "source_index": index,
                        "label": int(label),
                        "class_name": obj.classes[int(label)],
                    }
                )
        elif dataset == "pathmnist":
            for split in ("train", "val", "test"):
                obj = self._pathmnist(split)
                labels = np.asarray(obj.labels).reshape(-1)
                info = obj.info["label"]
                for index, label in enumerate(labels):
                    label_int = int(label)
                    rows.append(
                        {
                            "dataset": dataset,
                            "sample_id": f"pathmnist-{split}-{index:06d}",
                            "source_split": split,
                            "source_index": index,
                            "label": label_int,
                            "class_name": str(info[str(label_int)]),
                        }
                    )
        else:
            raise KeyError(f"Unknown dataset: {dataset}")
        return pd.DataFrame(rows)

    def image(self, dataset: str, source_split: str, source_index: int) -> Image.Image:
        index = int(source_index)
        if dataset == "cifar10":
            image, _ = self._cifar(source_split)[index]
        elif dataset == "eurosat":
            image, _ = self._eurosat()[index]
        elif dataset == "pathmnist":
            image, _ = self._pathmnist(source_split)[index]
        else:
            raise KeyError(dataset)
        return image.convert("RGB")


def prepare_raw_datasets() -> dict[str, int]:
    store = DatasetStore(download=True)
    counts = {}
    for dataset in DATASETS:
        frame = store.records(dataset)
        path = ROOT / "data" / "manifests" / f"{dataset}_raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        counts[dataset] = len(frame)
    return counts


def load_manifest(dataset: str) -> pd.DataFrame:
    path = ROOT / "data" / "manifests" / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen manifest: {path}")
    return pd.read_csv(path)


def class_names(dataset: str) -> list[str]:
    frame = load_manifest(dataset)
    mapping = (
        frame[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
    )
    return mapping.tolist()

