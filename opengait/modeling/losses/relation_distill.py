import torch
import torch.nn.functional as F

from .base import BaseLoss


class RelationDistillLoss(BaseLoss):
    """Batch-wise relational distillation loss.

    Expected kwargs:
      - student_feat: Tensor [N, D] or [N, C, P]
      - teacher_feat: Tensor, same shape as student_feat
      - mask: Optional Tensor [N], 1 for valid sample, 0 for invalid sample
    """

    def __init__(self, mode='mse', normalize=True, loss_term_weight=1.0):
        super(RelationDistillLoss, self).__init__(loss_term_weight)
        if mode not in ['mse', 'l1']:
            raise ValueError("mode should be 'mse' or 'l1', got {}".format(mode))
        self.mode = mode
        self.normalize = normalize

    def _flatten_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 2:
            return feat
        return feat.reshape(feat.shape[0], -1)

    def forward(self, student_feat, teacher_feat, mask=None):
        s = self._flatten_feat(student_feat).float()
        t = self._flatten_feat(teacher_feat).float()

        if s.shape != t.shape:
            raise ValueError(
                "RelationDistillLoss shape mismatch: student {}, teacher {}".format(tuple(s.shape), tuple(t.shape))
            )

        if self.normalize:
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        if mask is not None:
            m = mask.float().reshape(-1)
            if m.shape[0] != s.shape[0]:
                raise ValueError(
                    "RelationDistillLoss mask length mismatch: mask {}, batch {}".format(m.shape[0], s.shape[0])
                )
            valid_idx = m > 0
            s = s[valid_idx]
            t = t[valid_idx]

        if s.shape[0] < 2:
            loss = s.mean() * 0.0
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        sim_s = torch.matmul(s, s.t())
        sim_t = torch.matmul(t, t.t())

        eye = torch.eye(sim_s.shape[0], device=sim_s.device, dtype=torch.bool)
        if self.mode == 'mse':
            diff = (sim_s - sim_t)[~eye]
            loss = (diff ** 2).mean()
        else:
            diff = torch.abs(sim_s - sim_t)[~eye]
            loss = diff.mean()

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
