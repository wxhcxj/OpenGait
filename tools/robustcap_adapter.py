"""RobustCap adapter for export_teacher_cache.py.

Repo:
  https://github.com/wxhcxj/RobustCap

Design in this adapter:
  1) teacher features are derived from RobustCap network outputs
  2) IMU supports two modes:
      - provided: use real IMU sequence files
      - zero    : fallback to zero-IMU
  3) output is forced to the OpenGait distill cache contract:
      T_pose3d [T, 17, 3], T_motion [T, 17, 3], T_global [102]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class RobustCapInferencer:
    """RobustCap inferencer wrapper.

    Args:
      robustcap_root: local RobustCap repo root.
      weight_path: optional path to model weight .pt (if None use repo default).
      device: "cuda" or "cpu".
      imu_mode: "provided" or "zero".
      imu_acc_file: optional .npy file, shape [T, 6, 3], required for imu_mode=provided.
      imu_ori_file: optional .npy file, shape [T, 6, 3, 3], required for imu_mode=provided.
      joint_indices_17: optional 17-index list to map RobustCap 24 joints to 17 joints.
    """

    def __init__(
        self,
        robustcap_root: Optional[str] = None,
        weight_path: Optional[str] = None,
        device: str = "cuda",
        imu_mode: str = "zero",
        imu_acc_file: Optional[str] = None,
        imu_ori_file: Optional[str] = None,
        joint_indices_17: Optional[list] = None,
    ):
        if robustcap_root is None:
            raise ValueError("robustcap_root must be provided for RobustCapInferencer.")
        self.root = Path(robustcap_root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"robustcap_root not found: {self.root}")
        self.weight_path = Path(weight_path).resolve() if weight_path else None
        self.device_str = device
        self.imu_mode = imu_mode
        if self.imu_mode not in ["zero", "provided"]:
            raise ValueError(f"Unsupported imu_mode={self.imu_mode}, expected 'zero' or 'provided'.")
        self.imu_acc_file = Path(imu_acc_file).resolve() if imu_acc_file else None
        self.imu_ori_file = Path(imu_ori_file).resolve() if imu_ori_file else None

        # default 24->17 selection; can be overridden for specific dataset mapping.
        self.joint_indices_17 = joint_indices_17 if joint_indices_17 is not None else list(range(17))
        if len(self.joint_indices_17) != 17:
            raise ValueError("joint_indices_17 must contain 17 indices.")

        # Ensure RobustCap modules can be imported.
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        os.environ.setdefault("PYTHONPATH", "")
        if str(self.root) not in os.environ["PYTHONPATH"].split(":"):
            os.environ["PYTHONPATH"] = str(self.root) + ":" + os.environ["PYTHONPATH"]

        import torch  # noqa: WPS433
        import articulate as art  # noqa: WPS433
        import config as robustcap_config  # noqa: WPS433
        from net.sig_mp import Net  # noqa: WPS433

        self.torch = torch
        self.art = art
        self.config = robustcap_config
        self.net = Net().to(self._torch_device())
        self.net.live = True
        self.net.eval()

        if self.weight_path is None:
            default_weight = Path(self.config.paths.weight_dir) / self.net.name / "best_weights.pt"
            self.weight_path = default_weight
        if not self.weight_path.exists():
            raise FileNotFoundError(f"RobustCap weight not found: {self.weight_path}")

        state = torch.load(str(self.weight_path), map_location=self._torch_device())
        self.net.load_state_dict(state)
        self._load_imu_if_needed()

    def _torch_device(self):
        if self.device_str == "cuda" and self.torch.cuda.is_available():
            return self.torch.device("cuda")
        return self.torch.device("cpu")

    @staticmethod
    def _ensure_33_joints(pose_tjc: np.ndarray) -> np.ndarray:
        """RobustCap net.forward_online expects [33, 3] joints per frame."""
        pose = np.asarray(pose_tjc, dtype=np.float32)
        t, j, c = pose.shape
        if c < 3:
            raise ValueError(f"pose channel dimension should be >=3, got {c}")
        pose = pose[..., :3]
        if j == 33:
            return pose
        if j > 33:
            return pose[:, :33, :]
        out = np.zeros((t, 33, 3), dtype=np.float32)
        out[:, :j, :] = pose
        return out

    @staticmethod
    def _summary_global(x_tjc: np.ndarray) -> np.ndarray:
        mean = x_tjc.mean(axis=0).reshape(-1)
        std = x_tjc.std(axis=0).reshape(-1)
        return np.concatenate([mean, std], axis=0).astype(np.float32)

    def _load_imu_if_needed(self):
        if self.imu_mode != "provided":
            self.imu_acc = None
            self.imu_ori = None
            return
        if self.imu_acc_file is None or self.imu_ori_file is None:
            raise ValueError("imu_mode=provided requires imu_acc_file and imu_ori_file.")
        if not self.imu_acc_file.exists() or not self.imu_ori_file.exists():
            raise FileNotFoundError("imu_acc_file or imu_ori_file not found.")
        self.imu_acc = np.asarray(np.load(str(self.imu_acc_file)), dtype=np.float32)
        self.imu_ori = np.asarray(np.load(str(self.imu_ori_file)), dtype=np.float32)
        if self.imu_acc.ndim != 3 or self.imu_acc.shape[1:] != (6, 3):
            raise ValueError("imu_acc_file must have shape [T, 6, 3].")
        if self.imu_ori.ndim != 4 or self.imu_ori.shape[1:] != (6, 3, 3):
            raise ValueError("imu_ori_file must have shape [T, 6, 3, 3].")

    def _get_imu_frame(self, frame_idx: int, device):
        torch = self.torch
        if self.imu_mode == "provided":
            idx = min(frame_idx, self.imu_acc.shape[0] - 1)
            acc = torch.from_numpy(self.imu_acc[idx]).to(device=device, dtype=torch.float32)
            ori = torch.from_numpy(self.imu_ori[idx]).to(device=device, dtype=torch.float32)
        else:
            acc = torch.zeros(6, 3, device=device, dtype=torch.float32)
            ori = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(6, 1, 1)
        return acc, ori

    def infer(self, pose_tjc: np.ndarray) -> Dict[str, np.ndarray]:
        pose33 = self._ensure_33_joints(pose_tjc)
        torch = self.torch
        device = self._torch_device()

        pred_pose_aa = []
        with torch.no_grad():
            for t in range(pose33.shape[0]):
                uv = torch.from_numpy(pose33[t]).to(device=device, dtype=torch.float32)
                acc, ori = self._get_imu_frame(t, device)
                if t == 0:
                    pose_mat, _tran = self.net.forward_online(uv, acc, ori, first_frame=True)
                else:
                    pose_mat, _tran = self.net.forward_online(uv, acc, ori)

                # [24,3,3] -> [24,3] axis-angle from RobustCap prediction.
                pose_aa = self.art.math.rotation_matrix_to_axis_angle(pose_mat).reshape(24, 3)
                pred_pose_aa.append(pose_aa.detach().cpu().numpy())

        pred_pose_aa = np.asarray(pred_pose_aa, dtype=np.float32)  # [T, 24, 3]
        t_pose3d = pred_pose_aa[:, self.joint_indices_17, :]  # [T, 17, 3]
        t_motion = np.zeros_like(t_pose3d, dtype=np.float32)
        t_motion[1:] = t_pose3d[1:] - t_pose3d[:-1]
        t_global = self._summary_global(t_pose3d)
        if t_global.shape[0] != 102:
            raise ValueError(f"T_global dim should be 102, got {t_global.shape[0]}")

        return {
            "T_pose3d": t_pose3d,
            "T_motion": t_motion,
            "T_global": t_global,
        }
