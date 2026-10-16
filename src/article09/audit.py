from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

from .config import ROOT, sha256_file
from .data import DATASETS, DatasetStore
from .utils import atomic_write_json, dataframe_hash


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            nxt = int(self.parent[item])
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _pixel_sha(image: Image.Image) -> str:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _phash_int(image: Image.Image) -> int:
    return int(str(imagehash.phash(image.convert("RGB"), hash_size=8)), 16)


def _chunks(value: int) -> tuple[int, int, int, int, int]:
    # Five chunks guarantee at least one exact chunk for any pair at Hamming <= 4.
    return (
        value & ((1 << 13) - 1),
        (value >> 13) & ((1 << 13) - 1),
        (value >> 26) & ((1 << 13) - 1),
        (value >> 39) & ((1 << 13) - 1),
        (value >> 52) & ((1 << 12) - 1),
    )


def _confirmed_near_pairs(
    hashes: np.ndarray,
    thumbnails: np.memmap,
    hamming_max: int,
    ssim_min: float,
) -> Iterable[tuple[int, int, int, float]]:
    chunk_values = [_chunks(int(value)) for value in hashes]
    for segment in range(5):
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, parts in enumerate(chunk_values):
            buckets[parts[segment]].append(index)
        for members in buckets.values():
            for pos, left in enumerate(members):
                parts_left = chunk_values[left]
                for right in members[pos + 1 :]:
                    parts_right = chunk_values[right]
                    # A pair sharing several chunks is evaluated only at its first shared chunk.
                    if any(parts_left[k] == parts_right[k] for k in range(segment)):
                        continue
                    distance = (int(hashes[left]) ^ int(hashes[right])).bit_count()
                    if distance > hamming_max:
                        continue
                    score = float(
                        structural_similarity(
                            thumbnails[left], thumbnails[right], data_range=255
                        )
                    )
                    if score >= ssim_min:
                        yield left, right, distance, score


def _allocate_groups(
    frame: pd.DataFrame, ratios: tuple[float, float, float], seed: int
) -> pd.Series:
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=frame.index, dtype="object")
    split_names = ("train", "val", "test")
    for label, class_frame in frame.groupby("label", sort=True):
        groups = []
        for group_id, group in class_frame.groupby("duplicate_group", sort=False):
            groups.append((group_id, group.index.to_numpy(), len(group)))
        rng.shuffle(groups)
        total = sum(size for _, _, size in groups)
        targets = np.asarray(ratios, dtype=float) * total
        counts = np.zeros(3, dtype=int)
        for _, indices, size in groups:
            deficits = targets - counts
            deficits[np.asarray(ratios) <= 0] = -np.inf
            destination = int(np.argmax(deficits))
            assignment.loc[indices] = split_names[destination]
            counts[destination] += size
    return assignment


def _split_manifest(dataset: str, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["quarantine_reason"] = ""

    group_summary = frame.groupby("duplicate_group").agg(
        n_labels=("label", "nunique"),
        source_splits=("source_split", lambda values: "|".join(sorted(set(values)))),
    )
    ambiguous = set(group_summary.index[group_summary["n_labels"] > 1])
    if ambiguous:
        mask = frame["duplicate_group"].isin(ambiguous)
        frame.loc[mask, "quarantine_reason"] = "duplicate_group_crosses_labels"

    if dataset in {"cifar10", "pathmnist"}:
        crossing = set(
            group_summary.index[
                group_summary["source_splits"].str.contains("\\|", regex=True)
            ]
        )
        if crossing:
            mask = frame["duplicate_group"].isin(crossing)
            frame.loc[mask, "quarantine_reason"] = "duplicate_group_crosses_official_split"

    usable = frame[frame["quarantine_reason"] == ""].copy()
    if dataset == "cifar10":
        train_source = usable[usable["source_split"] == "train"]
        split = _allocate_groups(train_source, (0.9, 0.1, 0.0), seed)
        frame.loc[split.index, "split"] = split
        test_indices = usable.index[usable["source_split"] == "test"]
        frame.loc[test_indices, "split"] = "test"
    elif dataset == "pathmnist":
        mapping = {"train": "train", "val": "val", "test": "test"}
        frame.loc[usable.index, "split"] = usable["source_split"].map(mapping)
    elif dataset == "eurosat":
        split = _allocate_groups(usable, (0.70, 0.15, 0.15), seed)
        frame.loc[split.index, "split"] = split
    else:
        raise KeyError(dataset)

    frame.loc[frame["quarantine_reason"] != "", "split"] = "quarantine"
    frame["split"] = frame["split"].fillna("quarantine")
    return frame


def audit_dataset(dataset: str, config: dict) -> dict:
    audit_cfg = config["audit"]
    raw_path = ROOT / "data" / "manifests" / f"{dataset}_raw.csv"
    frame = pd.read_csv(raw_path)
    store = DatasetStore(download=False)
    count = len(frame)
    thumbs_path = ROOT / "data" / "audit" / f"{dataset}_gray64.dat"
    thumbnails = np.memmap(
        thumbs_path, mode="w+", dtype=np.uint8, shape=(count, 64, 64)
    )
    sha_values: list[str] = []
    phash_values = np.empty(count, dtype=np.uint64)
    decode_failures: list[dict] = []

    for pos, row in frame.iterrows():
        try:
            image = store.image(dataset, row.source_split, int(row.source_index))
            sha_values.append(_pixel_sha(image))
            phash_values[pos] = _phash_int(image)
            thumbnails[pos] = np.asarray(
                image.convert("L").resize((64, 64), Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )
        except Exception as exc:
            sha_values.append("")
            phash_values[pos] = 0
            decode_failures.append(
                {"sample_id": row.sample_id, "error": repr(exc), "position": int(pos)}
            )
    thumbnails.flush()

    frame["sha256_pixels"] = sha_values
    frame["phash64"] = [f"{int(value):016x}" for value in phash_values]
    uf = UnionFind(count)
    exact_pairs = 0
    for _, group in frame[frame["sha256_pixels"] != ""].groupby("sha256_pixels"):
        indices = group.index.to_numpy(dtype=int)
        if len(indices) > 1:
            first = int(indices[0])
            for other in indices[1:]:
                uf.union(first, int(other))
                exact_pairs += 1

    near_rows: list[dict] = []
    for left, right, distance, score in _confirmed_near_pairs(
        phash_values,
        thumbnails,
        int(audit_cfg["phash_hamming_max"]),
        float(audit_cfg["ssim_min"]),
    ):
        if frame.at[left, "sha256_pixels"] != frame.at[right, "sha256_pixels"]:
            uf.union(left, right)
            near_rows.append(
                {
                    "left_id": frame.at[left, "sample_id"],
                    "right_id": frame.at[right, "sample_id"],
                    "phash_distance": distance,
                    "ssim": score,
                }
            )

    roots = np.asarray([uf.find(i) for i in range(count)], dtype=np.int64)
    root_to_group = {root: f"{dataset}-g{n:06d}" for n, root in enumerate(sorted(set(roots)))}
    frame["duplicate_group"] = [root_to_group[int(root)] for root in roots]
    frame = _split_manifest(dataset, frame, int(config["project"]["split_seed"]))

    locked_dir = ROOT / "data" / "manifests" / "locked"
    locked_dir.mkdir(parents=True, exist_ok=True)
    locked_test_path = locked_dir / f"{dataset}_test.csv"
    frame.loc[frame["split"] == "test"].to_csv(locked_test_path, index=False)
    public_frame = frame.copy()
    test_mask = public_frame["split"] == "test"
    public_frame.loc[test_mask, "label"] = -1
    public_frame.loc[test_mask, "class_name"] = "LOCKED"
    manifest_path = ROOT / "data" / "manifests" / f"{dataset}.csv"
    public_frame.to_csv(manifest_path, index=False)
    near_path = ROOT / "data" / "audit" / f"{dataset}_near_duplicate_pairs.csv"
    pd.DataFrame(
        near_rows,
        columns=["left_id", "right_id", "phash_distance", "ssim"],
    ).to_csv(near_path, index=False)
    duplicate_frame = frame[frame.duplicated("duplicate_group", keep=False)].copy()
    duplicate_frame.to_csv(
        ROOT / "data" / "audit" / f"{dataset}_duplicate_groups.csv", index=False
    )
    quarantine = frame[frame["split"] == "quarantine"].copy()
    quarantine.to_csv(ROOT / "data" / "audit" / f"{dataset}_quarantine.csv", index=False)

    report = {
        "dataset": dataset,
        "total": count,
        "decode_failures": decode_failures,
        "exact_pair_links": exact_pairs,
        "confirmed_near_pair_links": len(near_rows),
        "duplicate_groups": int((frame.groupby("duplicate_group").size() > 1).sum()),
        "quarantined": int((frame["split"] == "quarantine").sum()),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "class_split_counts": frame.pivot_table(
            index="label", columns="split", values="sample_id", aggfunc="count", fill_value=0
        ).to_dict(),
        "manifest_sha256": sha256_file(manifest_path),
        "locked_test_sha256": sha256_file(locked_test_path),
        "manifest_content_hash": dataframe_hash(
            frame,
            ["sample_id", "label", "source_split", "source_index", "split", "duplicate_group"],
        ),
    }
    atomic_write_json(ROOT / "data" / "audit" / f"{dataset}_audit_report.json", report)
    del thumbnails
    return report


def audit_all(config: dict) -> dict:
    reports = {dataset: audit_dataset(dataset, config) for dataset in DATASETS}
    atomic_write_json(ROOT / "data" / "audit" / "audit_report.json", reports)
    return reports

