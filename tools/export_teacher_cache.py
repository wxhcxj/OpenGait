#!/usr/bin/env python3
"""Export teacher cache for SUSTech1K-style folders.

This exporter supports two backends:
  1) minimal  : build teacher fields directly from pose sequence (no external deps)
  2) robustcap: call an external RobustCap inferencer through a Python adapter

It scans sequences in <dataset_root>/<id>/<type>/<view>/ and exports per-sequence:
  - T_pose3d: [T, J, C]
  - T_motion: [T, J, C]
  - T_global: [D]
plus an index.json mapping "<id>/<type>/<view>" -> relative npz path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def load_joint_map(joint_map_file: Optional[Path]) -> Optional[List[int]]:
    if joint_map_file is None:
        return None
    with joint_map_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("joint-map must be a JSON list of integer indices.")
    mapped = [int(i) for i in data]
    if len(mapped) == 0:
        raise ValueError("joint-map should not be empty.")
    return mapped


def _resample_time_linear(pose_tjc: np.ndarray, target_t: int) -> np.ndarray:
    t, j, c = pose_tjc.shape
    if t == target_t:
        return pose_tjc
    if target_t <= 1:
        return pose_tjc[:1]
    src_idx = np.linspace(0, t - 1, num=t, dtype=np.float32)
    dst_idx = np.linspace(0, t - 1, num=target_t, dtype=np.float32)
    out = np.empty((target_t, j, c), dtype=np.float32)
    for jj in range(j):
        for cc in range(c):
            out[:, jj, cc] = np.interp(dst_idx, src_idx, pose_tjc[:, jj, cc])
    return out


def preprocess_pose(
    pose_tjc: np.ndarray,
    joint_map: Optional[Sequence[int]] = None,
    fps_src: Optional[float] = None,
    fps_target: Optional[float] = None,
    root_joint: int = 0,
    root_relative: bool = True,
    conf_thresh: Optional[float] = None,
    bone_norm: bool = False,
) -> np.ndarray:
    """Normalize pose sequence before teacher feature extraction/inference."""
    pose = np.asarray(pose_tjc, dtype=np.float32)

    # Keep xyz channels for teacher cache contract.
    if pose.shape[-1] > 3:
        if conf_thresh is not None:
            conf = pose[..., -1:]
            xyz = pose[..., :3]
            xyz = np.where(conf >= conf_thresh, xyz, 0.0)
            pose = xyz
        else:
            pose = pose[..., :3]

    if joint_map is not None:
        pose = pose[:, joint_map, :]

    if fps_src is not None and fps_target is not None and fps_src > 0 and fps_target > 0:
        target_t = max(1, int(round(pose.shape[0] * fps_target / fps_src)))
        pose = _resample_time_linear(pose, target_t=target_t)

    if root_relative:
        if not (0 <= root_joint < pose.shape[1]):
            raise ValueError(f"root-joint={root_joint} out of range for J={pose.shape[1]}")
        root = pose[:, root_joint:root_joint+1, :]
        pose = pose - root

    if bone_norm:
        # lightweight scale normalization using mean joint distance to root.
        root = pose[:, root_joint:root_joint+1, :]
        dist = np.linalg.norm(pose - root, axis=-1)  # [T, J]
        scale = float(np.mean(dist[:, 1:])) if pose.shape[1] > 1 else float(np.mean(dist))
        if scale > 1e-6:
            pose = pose / scale

    return pose.astype(np.float32)


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


def build_teacher_feature_robustcap_with_kwargs(
    pose_tjc: np.ndarray,
    adapter_module: str,
    adapter_class: str,
    adapter_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    module = importlib.import_module(adapter_module)
    Inferencer = getattr(module, adapter_class)
    inferencer = Inferencer(**(adapter_kwargs or {}))
    if hasattr(inferencer, "infer"):
        out = inferencer.infer(pose_tjc)
    else:
        out = inferencer(pose_tjc)
    if not isinstance(out, dict):
        raise TypeError("RobustCap adapter output must be a dict.")
    for k in ["T_pose3d", "T_motion", "T_global"]:
        if k not in out:
            raise KeyError(f"RobustCap adapter output missing key: {k}")
    return {
        "T_pose3d": np.asarray(out["T_pose3d"], dtype=np.float32),
        "T_motion": np.asarray(out["T_motion"], dtype=np.float32),
        "T_global": np.asarray(out["T_global"], dtype=np.float32).reshape(-1),
    }


def export_cache(
    dataset_root: Path,
    output_root: Path,
    pose_suffix: str,
    skip_existing: bool,
    backend: str,
    robustcap_module: str,
    robustcap_class: str,
    joint_map: Optional[List[int]],
    fps_src: Optional[float],
    fps_target: Optional[float],
    root_joint: int,
    root_relative: bool,
    conf_thresh: Optional[float],
    bone_norm: bool,
    robustcap_root: Optional[Path],
    robustcap_weight: Optional[Path],
    robustcap_device: str,
    robustcap_imu_mode: str,
    robustcap_imu_acc_file: Optional[Path],
    robustcap_imu_ori_file: Optional[Path],
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
            raw_pose = load_pose_tensor(pose_pkl)
            pose = preprocess_pose(
                raw_pose,
                joint_map=joint_map,
                fps_src=fps_src,
                fps_target=fps_target,
                root_joint=root_joint,
                root_relative=root_relative,
                conf_thresh=conf_thresh,
                bone_norm=bone_norm,
            )
            if backend == "minimal":
                cache = build_teacher_feature(pose)
            elif backend == "robustcap":
                adapter_kwargs: Dict[str, object] = {
                    "robustcap_root": str(robustcap_root) if robustcap_root is not None else None,
                    "weight_path": str(robustcap_weight) if robustcap_weight is not None else None,
                    "device": robustcap_device,
                    "imu_mode": robustcap_imu_mode,
                    "imu_acc_file": str(robustcap_imu_acc_file) if robustcap_imu_acc_file is not None else None,
                    "imu_ori_file": str(robustcap_imu_ori_file) if robustcap_imu_ori_file is not None else None,
                }
                cache = build_teacher_feature_robustcap_with_kwargs(
                    pose,
                    adapter_module=robustcap_module,
                    adapter_class=robustcap_class,
                    adapter_kwargs=adapter_kwargs)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
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
            backend=backend,
        )

        index[seq_key] = str(out_file.relative_to(output_root))
        ok += 1

    with (output_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return ok, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export teacher cache for SUSTech1K-style data")
    parser.add_argument("--dataset-root", required=True, type=Path, help="root path of SUSTech1K-Released-pkl")
    parser.add_argument("--output-root", required=True, type=Path, help="where to save teacher cache files")
    parser.add_argument(
        "--pose-suffix",
        default="Camera-Pose.pkl",
        help="suffix used to find pose pkl inside each seq dir (default: Camera-Pose.pkl)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="skip sequence if output already exists")
    parser.add_argument("--backend", default="minimal", choices=["minimal", "robustcap"],
                        help="teacher backend: minimal(no external deps) or robustcap(adapter based)")
    parser.add_argument("--robustcap-module", default="tools.robustcap_adapter",
                        help="python module path for RobustCap adapter, used when backend=robustcap")
    parser.add_argument("--robustcap-class", default="RobustCapInferencer",
                        help="class name inside robustcap module")
    parser.add_argument("--robustcap-root", type=Path, default=None,
                        help="path to local RobustCap repo root (required for robustcap backend)")
    parser.add_argument("--robustcap-weight", type=Path, default=None,
                        help="optional robustcap weight file; default uses repo config path")
    parser.add_argument("--robustcap-device", default="cuda", choices=["cuda", "cpu"],
                        help="device for robustcap backend")
    parser.add_argument("--robustcap-imu-mode", default="zero", choices=["zero", "provided"],
                        help="how to provide IMU input for RobustCap")
    parser.add_argument("--robustcap-imu-acc-file", type=Path, default=None,
                        help="optional IMU acc npy file [T,6,3] for robustcap adapter")
    parser.add_argument("--robustcap-imu-ori-file", type=Path, default=None,
                        help="optional IMU ori npy file [T,6,3,3] for robustcap adapter")
    parser.add_argument("--joint-map", type=Path, default=None,
                        help="optional JSON list file for joint remapping")
    parser.add_argument("--fps-src", type=float, default=None,
                        help="source fps of pose sequence")
    parser.add_argument("--fps-target", type=float, default=None,
                        help="target fps for export cache")
    parser.add_argument("--root-joint", type=int, default=0,
                        help="root joint index for root-relative normalization")
    parser.add_argument("--no-root-relative", action="store_true",
                        help="disable root-relative normalization")
    parser.add_argument("--conf-thresh", type=float, default=None,
                        help="optional confidence threshold (when input has confidence channel)")
    parser.add_argument("--bone-norm", action="store_true",
                        help="enable simple bone-scale normalization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")
    if args.backend == "robustcap" and args.robustcap_root is None:
        raise ValueError("--robustcap-root is required when backend=robustcap")
    if args.backend == "robustcap" and args.robustcap_imu_mode == "provided":
        if args.robustcap_imu_acc_file is None or args.robustcap_imu_ori_file is None:
            raise ValueError("--robustcap-imu-acc-file and --robustcap-imu-ori-file are required when --robustcap-imu-mode=provided")
    joint_map = load_joint_map(args.joint_map)

    ok, skipped = export_cache(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        pose_suffix=args.pose_suffix,
        skip_existing=args.skip_existing,
        backend=args.backend,
        robustcap_module=args.robustcap_module,
        robustcap_class=args.robustcap_class,
        joint_map=joint_map,
        fps_src=args.fps_src,
        fps_target=args.fps_target,
        root_joint=args.root_joint,
        root_relative=not args.no_root_relative,
        conf_thresh=args.conf_thresh,
        bone_norm=args.bone_norm,
        robustcap_root=args.robustcap_root,
        robustcap_weight=args.robustcap_weight,
        robustcap_device=args.robustcap_device,
        robustcap_imu_mode=args.robustcap_imu_mode,
        robustcap_imu_acc_file=args.robustcap_imu_acc_file,
        robustcap_imu_ori_file=args.robustcap_imu_ori_file,
    )
    print(
        "[DONE] exported={}, skipped_existing={}, backend={}, output={}".format(
            ok, skipped, args.backend, args.output_root
        )
    )


if __name__ == "__main__":
    main()
