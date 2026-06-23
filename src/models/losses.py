"""
Loss for the physics-informed CAT CNN.

    total = huber(log1p(max_EDR̂), log1p(max_edr))           # scalar supervision (magnitude-weighted)
          + w_aux · huber(log1p(mean_EDR̂), log1p(mean_edr)) # optional auxiliary
          + Σ λ_i · physics_penalty_i(field, Ri, VWS, TI1)   # soft physical constraints

Physics penalties (the principles we penalise violations of):
  1. non-negativity     EDR ≥ 0  (softplus already guarantees this; backstop only)
  2. Ri/shear gating    no turbulence where the flow is dynamically stable (Ri≫1 & low shear)
  3. TI consistency     predicted field must not anti-correlate with the Ellrod TI field
  4. TV smoothness      suppress unphysical pixel noise; CAT patches are spatially coherent
  5. climatological cap  predictions must not exceed the observed EDR max by a margin

Penalties act on the *raw* (un-normalised) Ri/VWS/TI1 fields carried in batch['phys'] (order
fixed by dataset.PHYS_CHANNELS), so the physical thresholds are meaningful.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MAX_EDR = 0.95

DEFAULT_LOSS_CFG = dict(
    huber_delta=0.1,     # Huber transition in log-EDR space
    w_aux=0.2,           # weight on the mean-EDR auxiliary term
    w_mag=1.0,           # magnitude up-weighting strength (rare-severe emphasis)
    lam_nonneg=1.0,      # (1) softplus backstop
    lam_ri=0.1,          # (2) Ri/shear gating
    lam_ti=0.05,         # (3) Ellrod-TI anti-correlation
    lam_tv=0.01,         # (4) total-variation smoothness
    lam_bound=1.0,       # (5) climatological cap
    ri_crit=1.0,         # Richardson number above which flow is "stable"
    gate_sharp=2.0,      # sharpness of the stable/low-shear gates
)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample Pearson r over flattened pixels. a,b: (N, P) → (N,)."""
    am = a - a.mean(1, keepdim=True)
    bm = b - b.mean(1, keepdim=True)
    num = (am * bm).sum(1)
    den = torch.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1) + 1e-12)
    return num / (den + 1e-12)


def compute_loss(out: dict, batch: dict, cfg: dict | None = None):
    """Return (total_loss, parts_dict_of_floats)."""
    c = {**DEFAULT_LOSS_CFG, **(cfg or {})}
    field = out["field"]                      # (N,1,H,W) ≥ 0
    max_hat = out["max_hat"]                  # (N,1)
    mean_hat = out["mean_hat"]                # (N,1)
    y_max = batch["y_max"]                    # (N,1)
    y_mean = batch["y_mean"]                  # (N,1)

    N = field.shape[0]
    Ri  = batch["phys"][:, 0]                 # (N,H,W) raw
    VWS = batch["phys"][:, 1]
    TI1 = batch["phys"][:, 2]

    # ── base: Huber on log1p, weighted toward larger (rarer) EDR ─────────────
    log_hat = torch.log1p(max_hat.clamp_min(0))
    log_tgt = torch.log1p(y_max.clamp_min(0))
    w = 1.0 + c["w_mag"] * y_max              # magnitude weighting (severe ≫ calm)
    base = (w * F.huber_loss(log_hat, log_tgt, delta=c["huber_delta"], reduction="none")).mean()

    aux = F.huber_loss(torch.log1p(mean_hat.clamp_min(0)), torch.log1p(y_mean.clamp_min(0)),
                       delta=c["huber_delta"])

    # ── (1) non-negativity backstop ──────────────────────────────────────────
    p_nonneg = F.relu(-field).pow(2).mean()

    # ── (2) Ri/shear gating: penalise EDR in stable, low-shear air ───────────
    stable = torch.sigmoid(c["gate_sharp"] * (Ri - c["ri_crit"]))
    vws_scale = VWS.mean() + 1e-6
    low_shear = torch.sigmoid(c["gate_sharp"] * (1.0 - VWS / vws_scale))
    gate = (stable * low_shear).unsqueeze(1)          # (N,1,H,W)
    p_ri = (gate * field.pow(2)).mean()

    # ── (3) Ellrod-TI anti-correlation penalty ───────────────────────────────
    r = _pearson(field.view(N, -1), TI1.reshape(N, -1))
    p_ti = F.relu(-r).mean()                          # only penalise inversion

    # ── (4) total-variation smoothness ───────────────────────────────────────
    dx = (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()
    dy = (field[:, :, 1:, :] - field[:, :, :-1, :]).abs().mean()
    p_tv = dx + dy

    # ── (5) climatological cap ───────────────────────────────────────────────
    p_bound = F.relu(field - MAX_EDR).pow(2).mean() + F.relu(max_hat - MAX_EDR).pow(2).mean()

    total = (
        base
        + c["w_aux"] * aux
        + c["lam_nonneg"] * p_nonneg
        + c["lam_ri"] * p_ri
        + c["lam_ti"] * p_ti
        + c["lam_tv"] * p_tv
        + c["lam_bound"] * p_bound
    )

    parts = {
        "total": float(total.detach()), "base": float(base.detach()), "aux": float(aux.detach()),
        "ri": float(p_ri.detach()), "ti": float(p_ti.detach()),
        "tv": float(p_tv.detach()), "bound": float(p_bound.detach()),
        "nonneg": float(p_nonneg.detach()),
    }
    return total, parts
