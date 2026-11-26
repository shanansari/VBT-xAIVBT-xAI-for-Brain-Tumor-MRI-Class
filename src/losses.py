"""Loss, optimizer, scheduler and EMA used to train VBT-xAI (Cell 8 of the notebook).

L_total = 0.5 * L_SmoothCE + 0.5 * L_Focal + 0.3 * 0.5 * (L_aux_CNN + L_aux_ViT)   (Eq. 7)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LABEL_SMOOTH = 0.05
LR = 1e-5
WEIGHT_DECAY = 1e-5


class SmoothCE(nn.Module):
    """Class-weighted, label-smoothed cross-entropy."""

    def __init__(self, nc, sm=0.1, w=None):
        super().__init__()
        self.nc, self.sm, self.w = nc, sm, w

    def forward(self, pred, tgt):
        conf = 1 - self.sm
        sv = self.sm / max(1, self.nc - 1)
        oh = torch.zeros_like(pred).scatter_(1, tgt.unsqueeze(1), 1)
        st = oh * conf + (1 - oh) * sv
        loss = -(st * F.log_softmax(pred, dim=-1)).sum(-1)
        if self.w is not None:
            loss = loss * self.w[tgt]
        return loss.mean()


class FocalLoss(nn.Module):
    """Class-weighted focal loss (alpha = 0.25, gamma = 2)."""

    def __init__(self, a=0.25, g=2.0, w=None):
        super().__init__()
        self.a, self.g, self.w = a, g, w

    def forward(self, inp, tgt):
        ce = F.cross_entropy(inp, tgt, weight=self.w, reduction="none")
        pt = torch.exp(-ce)
        return (self.a * (1 - pt) ** self.g * ce).mean()


class CombLoss(nn.Module):
    """Main loss (smoothed CE + focal) plus auxiliary-branch supervision."""

    def __init__(self, nc, w=None, aux=0.3, label_smooth=LABEL_SMOOTH):
        super().__init__()
        self.main = SmoothCE(nc, label_smooth, w)
        self.focal = FocalLoss(w=w)
        self.aux = aux

    def forward(self, ml, cl, vl, tgt):
        lm = 0.5 * self.main(ml, tgt) + 0.5 * self.focal(ml, tgt)
        la = (self.main(cl, tgt) + self.main(vl, tgt)) / 2 if cl is not None else 0
        return lm + self.aux * la


def build_optimizer(m, base_lr=LR, weight_decay=WEIGHT_DECAY):
    """AdamW with differential LR: backbones at 0.1x, fusion/gate/head at base_lr."""
    return optim.AdamW([
        {"params": m.cnn.parameters(), "lr": base_lr * 0.1},
        {"params": m.vit.parameters(), "lr": base_lr * 0.1},
        {"params": list(m.fusion.parameters()) + list(m.gate.parameters()) +
                   list(m.head.parameters()) + list(m.cnn_aux.parameters()) +
                   list(m.vit_aux.parameters()), "lr": base_lr},
    ], weight_decay=weight_decay)


def build_scheduler(opt, total_epochs, warmup_epochs):
    """Linear warm-up followed by cosine decay (floor 1% of the base LR)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0))) * 0.99 + 0.01
    return optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class SimpleEMA:
    """Exponential moving average of model weights (decay 0.999)."""

    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]
        self.collected = None

    def update(self, params):
        with torch.no_grad():
            for s, p in zip(self.shadow, params):
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, params):
        self.collected = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, s in zip(params, self.shadow):
                p.copy_(s)

    def restore(self, params):
        if self.collected is None:
            return
        with torch.no_grad():
            for p, c in zip(params, self.collected):
                p.copy_(c)
        self.collected = None
