"""
Loss functions for CatCNNTorch.

Base: magnitude-weighted log1p-Huber on max_hat vs y_max.
Physics penalties: non-negativity, Ri/shear gating, Ellrod anti-correlation,
                   TV smoothness, climatological cap.
"""

import torch
import torch.nn.functional as F

DEFAULT_LOSS_CFG = dict(
    lambda_mean    = 0.2,    # auxiliary mean_edr term weight
    lambda_phys    = 0.1,    # total physics penalty weight
    lambda_tv      = 0.05,   # total variation smoothness
    lambda_cap     = 0.05,   # climatological cap penalty
    huber_delta    = 0.1,
    mag_weight_exp = 2.0,    # upweight severe events
    cap_value      = 1.05,   # softplus can exceed 1, penalise >cap
)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-8)
    return (a * b).sum() / denom


def compute_loss(out: dict, batch: dict, cfg: dict = None) -> tuple[torch.Tensor, dict]:
    if cfg is None:
        cfg = DEFAULT_LOSS_CFG
    c = {**DEFAULT_LOSS_CFG, **cfg}

    max_hat  = out["max_hat"]          # (B,)
    mean_hat = out["mean_hat"]         # (B,)
    field    = out["field"]            # (B,1,H,W)
    y_max    = batch["y_max"].float()  # (B,)
    y_mean   = batch["y_mean"].float() # (B,)

    # Magnitude weights (severe events weighted more)
    weights = (1.0 + y_max) ** c["mag_weight_exp"]
    weights = weights / weights.mean().clamp(min=1e-6)

    # log1p-Huber on max_hat
    p = torch.log1p(max_hat.clamp(min=0))
    t = torch.log1p(y_max)
    base_loss = (weights * F.huber_loss(p, t, delta=c["huber_delta"], reduction="none")).mean()

    # Auxiliary mean term
    pm = torch.log1p(mean_hat.clamp(min=0))
    tm = torch.log1p(y_mean)
    mean_loss = F.huber_loss(pm, tm, delta=c["huber_delta"])

    # ── Physics penalties ────────────────────────────────────────────────────
    phys_pen = torch.tensor(0.0, device=field.device)

    # Non-negativity (softplus output is always ≥ 0, but clamp for safety)
    neg_pen = F.relu(-field).mean()

    # Ri/shear gating: if Ri > 0.25 everywhere (stable), field should be small
    if "phys" in batch:
        ri_idx = 0   # first phys channel is Ri
        ri_mean = batch["phys"][:, ri_idx]            # (B,)
        stable  = (ri_mean > 0.25).float()
        gate_pen = (stable * field.mean(dim=(1, 2, 3))).mean()
        phys_pen = phys_pen + gate_pen

    # Ellrod TI anti-correlation: TI1 and Ri should anti-correlate with field
    # (high TI1 → high field; high Ri → low field) — use rank correlation proxy
    if "phys" in batch and batch["phys"].shape[1] >= 2:
        ti1 = batch["phys"][:, 2]   # TI1 channel
        fmax = field.amax(dim=(1, 2, 3))
        phys_pen = phys_pen - 0.01 * _pearson(ti1, fmax)   # want positive correlation

    # Total variation smoothness
    dy = field[:, :, 1:, :] - field[:, :, :-1, :]
    dx = field[:, :, :, 1:] - field[:, :, :, :-1]
    tv_pen = (dy.abs().mean() + dx.abs().mean()) / 2.0

    # Climatological cap
    cap_pen = F.relu(field - c["cap_value"]).mean()

    total = (base_loss
             + c["lambda_mean"]  * mean_loss
             + c["lambda_phys"]  * (phys_pen + neg_pen)
             + c["lambda_tv"]    * tv_pen
             + c["lambda_cap"]   * cap_pen)

    parts = dict(
        base=base_loss.item(),
        mean=mean_loss.item(),
        phys=phys_pen.item(),
        tv=tv_pen.item(),
        cap=cap_pen.item(),
    )
    return total, parts
