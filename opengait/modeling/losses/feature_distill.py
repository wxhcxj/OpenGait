import torch
import torch.nn.functional as F

from .base import BaseLoss


class FeatureDistillLoss(BaseLoss):
    """Sequence-level feature distillation loss.

    Expected kwargs in training_feat:
      - student_feat: Tensor, shape [N, D] or [N, C, P]
      - teacher_feat: Tensor, shape compatible with student_feat
      - mask: Optional Tensor, shape [N] (1 for valid, 0 for invalid)

    Args:
      mode: "mse" or "cosine".
      normalize: L2 normalize features before computing loss.
      loss_term_weight: inherited from BaseLoss.
    """

    def __init__(self, mode='mse', normalize=True, loss_term_weight=1.0):
        super(FeatureDistillLoss, self).__init__(loss_term_weight)
        if mode not in ['mse', 'cosine']:
            raise ValueError("mode should be 'mse' or 'cosine', got {}".format(mode))
        self.mode = mode
        self.normalize = normalize

    def _flatten_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 2:
            return feat
        # e.g. [N, C, P] -> [N, C*P]
        return feat.reshape(feat.shape[0], -1)

    def forward(self, student_feat, teacher_feat, mask=None):
        s = self._flatten_feat(student_feat).float()
        t = self._flatten_feat(teacher_feat).float()

        if s.shape != t.shape:
            raise ValueError(
                "FeatureDistillLoss shape mismatch: student {}, teacher {}".format(tuple(s.shape), tuple(t.shape))
            )

        if self.normalize:
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        if self.mode == 'mse':
            per_sample = ((s - t) ** 2).mean(dim=-1)
        else:
            per_sample = 1.0 - F.cosine_similarity(s, t, dim=-1)

        if mask is not None:
            m = mask.float().reshape(-1)
            if m.shape[0] != per_sample.shape[0]:
                raise ValueError(
                    "FeatureDistillLoss mask length mismatch: mask {}, batch {}".format(m.shape[0], per_sample.shape[0])
                )
            valid = m.sum()
            if valid <= 0:
                loss = per_sample.mean() * 0.0
            else:
                loss = (per_sample * m).sum() / (valid + 1e-9)
        else:
            loss = per_sample.mean()

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
