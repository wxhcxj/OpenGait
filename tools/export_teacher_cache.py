#!/usr/bin/env python3
"""Export minimal teacher cache for SUSTech1K-style folders.

This script is a minimal runnable baseline for distillation data preparation.
It scans sequences in <dataset_root>/<id>/<type>/<view>/, reads a pose pkl,
and exports per-sequence cache files containing:
  - T_pose3d: [T, J, C]
  - T_motion: [T, J, C]
  - T_global: [2*J*C]  (concat of temporal mean/std)

Why minimal:
- It does not depend on RobustCap runtime.
- It can run immediately on existing SUSTech1K pkl structure.
- You can replace `build_teacher_feature()` with RobustCap inference later.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def list_sequences(dataset_root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    """Yield (label, seq_type, view, seq_dir)."""
    for label in sorted(p.name for p in dataset_root.iterdir() if p.is_dir()):
        label_dir = dataset_root / label
        for seq_type in sorted(p.name for p in label_dir.iterdir() if p.is_dir()):
            type_dir = label_dir / seq_type
            for view in sorted(p.name for p in type_dir.iterdir() if p.is_dir()):
                seq_dir = type_dir / view
                yield label, seq_type, view, seq_dir


def find_pose_file(seq_dir: Path, suffix: str) -> Optional[Path]:
    files = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.name.endswith(suffix)])
    if not files:
        return None
    return files[0]


def _as_array_from_frame(frame: object) -> Optional[np.ndarray]:
    """Best-effort parse a single frame pose object into [J, C]."""
    if isinstance(frame, np.ndarray):
        arr = frame
    elif isinstance(frame, dict):
        # Common candidate keys for keypoints/pose
        for k in ["keypoints", "pose", "joints", "joints3d", "kp", "data"]:
            if k in frame:
                arr = np.asarray(frame[k])
                break
        else:
            return None
    elif isinstance(frame, (list, tuple)):
        arr = np.asarray(frame)
    else:
        return None

    if arr.ndim == 1:
        return None
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.asarray(arr, dtype=np.float32)


def load_pose_tensor(pose_pkl: Path) -> np.ndarray:
    """Load pkl and return [T, J, C] float32 tensor."""
    with pose_pkl.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (list, tuple)):
        frames: List[np.ndarray] = []
        for item in data:
            parsed = _as_array_from_frame(item)
            if parsed is not None:
                frames.append(parsed)
        if not frames:
            raise ValueError(f"No valid pose frames parsed from {pose_pkl}")
        min_j = min(x.shape[0] for x in frames)
        min_c = min(x.shape[1] for x in frames)
        frames = [x[:min_j, :min_c] for x in frames]
        arr = np.stack(frames, axis=0)
    elif isinstance(data, dict):
        for k in ["keypoints", "pose", "joints", "joints3d", "kp", "data"]:
            if k in data:
                arr = np.asarray(data[k])
                break
        else:
            raise ValueError(f"Unsupported dict keys in {pose_pkl}: {list(data.keys())[:8]}")
    else:
        raise TypeError(f"Unsupported pkl object type: {type(data)} @ {pose_pkl}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        # [J, C] => [1, J, C]
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expect 3D tensor [T,J,C], got shape={arr.shape} in {pose_pkl}")

    return arr


def build_teacher_feature(pose_tjc: np.ndarray) -> Dict[str, np.ndarray]:
    """Build minimal teacher cache fields from pose.

    Returns dict with keys: T_pose3d, T_motion, T_global.
    """
    pose = pose_tjc.astype(np.float32)

    # Minimal motion feature: first-order temporal difference.
    motion = np.zeros_like(pose, dtype=np.float32)
    motion[1:] = pose[1:] - pose[:-1]

    # Minimal global embedding: concat(mean, std) over time.
    feat_mean = pose.mean(axis=0).reshape(-1)
    feat_std = pose.std(axis=0).reshape(-1)
    global_feat = np.concatenate([feat_mean, feat_std], axis=0).astype(np.float32)

    return {
        "T_pose3d": pose,
        "T_motion": motion,
        "T_global": global_feat,
    }


def export_cache(
    dataset_root: Path,
    output_root: Path,
    pose_suffix: str,
    skip_existing: bool,
) -> Tuple[int, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    index: Dict[str, str] = {}

    ok = 0
    skipped = 0

    for label, seq_type, view, seq_dir in list_sequences(dataset_root):
        seq_key = f"{label}/{seq_type}/{view}"
        out_dir = output_root / label / seq_type / view
        out_file = out_dir / "teacher_cache.npz"

        if skip_existing and out_file.exists():
            index[seq_key] = str(out_file.relative_to(output_root))
            skipped += 1
            continue

        pose_pkl = find_pose_file(seq_dir, pose_suffix)
        if pose_pkl is None:
            continue

        try:
            pose = load_pose_tensor(pose_pkl)
            cache = build_teacher_feature(pose)
        except Exception as e:  # keep export robust for large datasets
            print(f"[WARN] skip {seq_key}: {e}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_file,
            T_pose3d=cache["T_pose3d"],
            T_motion=cache["T_motion"],
            T_global=cache["T_global"],
            pose_source=str(pose_pkl),
        )

        index[seq_key] = str(out_file.relative_to(output_root))
        ok += 1

    with (output_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return ok, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export minimal teacher cache for SUSTech1K-style data")
    parser.add_argument("--dataset-root", required=True, type=Path, help="root path of SUSTech1K-Released-pkl")
    parser.add_argument("--output-root", required=True, type=Path, help="where to save teacher cache files")
    parser.add_argument(
        "--pose-suffix",
        default="Camera-Pose.pkl",
        help="suffix used to find pose pkl inside each seq dir (default: Camera-Pose.pkl)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="skip sequence if output already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    ok, skipped = export_cache(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        pose_suffix=args.pose_suffix,
        skip_existing=args.skip_existing,
    )
    print(f"[DONE] exported={ok}, skipped_existing={skipped}, output={args.output_root}")


if __name__ == "__main__":
    main()
