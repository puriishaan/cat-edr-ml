"""
Physics-informed CAT CNN (PyTorch).

Two-stream encoder–decoder that emits a dense EDR heatmap Ŷ(x,y) ≥ 0 and soft-aggregates
it to the scalar max_EDR the labels actually provide:

    diag (+ broadcast climate) ─► U-Net encoder ──┐
                                                   ├─ bottleneck ─► U-Net decoder ─► 1×1 FNN head
    satellite scalars ─► MLP ──► broadcast ────────┘                                   │ softplus
                                                                                        ▼
                                                              Ŷ(x,y) ≥ 0  ──soft-pool──► max_EDR̂

Design choices (all the knobs Optuna searches live in `cfg`):
  • `same` padding everywhere so encoder/decoder/skip sizes line up (odd dims handled by
    interpolating the decoder up to each skip's exact size).
  • FiLM(γ,β) after every encoder block, generated from cyclic-time (+ optionally climate).
  • softplus output → EDR ≥ 0 as a hard physical bound, no tail saturation (vs sigmoid).
  • soft aggregation (softmax-weighted mean ≈ max, or top-k mean) bridges the dense field to
    the scalar label; temperature τ is searchable (τ→∞ ⇒ true max, τ→0 ⇒ mean).
  • satellite branch is fully optional: a present/absent mask lets the net ignore it, and
    cfg['use_sat']=False removes it entirely (the satellite-skill ablation).
  • NO residual: the field is predicted directly (direct-vs-residual is an open A/B).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Default config (Optuna overrides any subset) ─────────────────────────────
DEFAULT_CFG = dict(
    depth=3,                 # encoder blocks (24→12→6→3 at depth 3)
    base_width=32,           # channels in first block; doubles each block
    kernel=3,                # conv kernel (3 or 5)
    norm="group",            # group | batch | none
    act="gelu",              # relu | gelu
    pool="max",              # max | avg | stride
    dropout=0.1,             # spatial dropout in conv blocks
    # conditioning
    film_hidden=32,          # FiLM generator hidden units (0 ⇒ no FiLM)
    film_use_climate=False,  # also feed climate into FiLM
    broadcast_climate=True,  # tile climate as extra input channels
    # satellite stream
    use_sat=True,
    sat_hidden=16,
    # FNN head (1×1 convs == per-pixel MLP)
    fnn_depth=2,             # hidden layers (0 ⇒ linear readout)
    fnn_width=64,
    fnn_dropout=0.1,
    fnn_reinject="none",     # none | film | climate  (re-add conditioning at the head)
    # aggregation heatmap→scalar
    agg="logsumexp",         # logsumexp (softmax-weighted mean) | topk
    agg_tau=8.0,             # softmax temperature
    agg_k=8,                 # top-k pixels for topk mean
)


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU()}[name]


def _norm(name: str, ch: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(ch)
    if name == "group":
        groups = math.gcd(ch, 8) or 1
        return nn.GroupNorm(max(1, groups), ch)
    return nn.Identity()


class ConvBlock(nn.Module):
    """[Conv→Norm→Act→Dropout]×2 with `same` padding."""

    def __init__(self, cin, cout, k, norm, act, dropout):
        super().__init__()
        pad = (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=pad),
            _norm(norm, cout), _act(act), nn.Dropout2d(dropout),
            nn.Conv2d(cout, cout, k, padding=pad),
            _norm(norm, cout), _act(act),
        )

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    """Per-channel feature-wise linear modulation: γ⊙x + β from a conditioning vector."""

    def __init__(self, cond_dim, ch, hidden):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * ch)
        )
        self.ch = ch
        nn.init.zeros_(self.gen[-1].weight)      # start as identity (γ=1, β=0)
        nn.init.zeros_(self.gen[-1].bias)

    def forward(self, x, cond):
        gb = self.gen(cond)
        gamma, beta = gb[:, : self.ch], gb[:, self.ch :]
        return (1.0 + gamma)[:, :, None, None] * x + beta[:, :, None, None]


def _downsample(mode, ch):
    if mode == "stride":
        return nn.Conv2d(ch, ch, 2, stride=2)
    if mode == "avg":
        return nn.AvgPool2d(2, ceil_mode=True)
    return nn.MaxPool2d(2, ceil_mode=True)


class CatCNNTorch(nn.Module):
    def __init__(self, in_channels, climate_dim, time_dim, sat_dim, cfg=None):
        super().__init__()
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        c = self.cfg
        self.climate_dim = climate_dim
        self.time_dim = time_dim

        # conditioning dim for FiLM
        self.cond_dim = time_dim + (climate_dim if c["film_use_climate"] else 0)
        self.use_film = c["film_hidden"] > 0 and self.cond_dim > 0

        enc_in = in_channels + (climate_dim if c["broadcast_climate"] else 0)
        widths = [c["base_width"] * (2 ** i) for i in range(c["depth"])]

        # ── encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.enc_films = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = enc_in
        for w in widths:
            self.enc_blocks.append(ConvBlock(prev, w, c["kernel"], c["norm"], c["act"], c["dropout"]))
            self.enc_films.append(
                FiLM(self.cond_dim, w, c["film_hidden"]) if self.use_film else None
            )
            self.downs.append(_downsample(c["pool"], w))
            prev = w

        # ── satellite stream ─────────────────────────────────────────────────
        self.sat_emb_dim = 0
        if c["use_sat"]:
            self.sat_emb_dim = c["sat_hidden"]
            self.sat_mlp = nn.Sequential(
                nn.Linear(sat_dim + 1, c["sat_hidden"]), nn.GELU(),
                nn.Linear(c["sat_hidden"], c["sat_hidden"]), nn.GELU(),
            )

        # ── bottleneck (fuses satellite embedding) ───────────────────────────
        bott_in = widths[-1] + self.sat_emb_dim
        self.bottleneck = ConvBlock(bott_in, widths[-1], c["kernel"], c["norm"], c["act"], c["dropout"])

        # ── decoder (upsample + skip concat) ─────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(c["depth"])):
            skip_ch = widths[i]
            up_ch = widths[i + 1] if i + 1 < c["depth"] else widths[-1]
            self.dec_blocks.append(
                ConvBlock(up_ch + skip_ch, skip_ch, c["kernel"], c["norm"], c["act"], c["dropout"])
            )

        # ── FNN head (1×1 convs == shared per-pixel MLP) ─────────────────────
        head_in = widths[0]
        self.reinject_dim = 0
        if c["fnn_reinject"] == "film":
            self.reinject_dim = self.cond_dim
        elif c["fnn_reinject"] == "climate":
            self.reinject_dim = climate_dim
        head_in += self.reinject_dim

        layers = []
        prev = head_in
        for _ in range(c["fnn_depth"]):
            layers += [nn.Conv2d(prev, c["fnn_width"], 1), _act(c["act"]), nn.Dropout2d(c["fnn_dropout"])]
            prev = c["fnn_width"]
        layers += [nn.Conv2d(prev, 1, 1)]
        self.head = nn.Sequential(*layers)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _cond(self, time, climate):
        if not self.use_film:
            return None
        return torch.cat([time, climate], 1) if self.cfg["film_use_climate"] else time

    def aggregate(self, field):
        """Heatmap (N,1,H,W) ≥0 → scalar (N,1) predicted max_EDR."""
        N = field.shape[0]
        flat = field.view(N, -1)
        if self.cfg["agg"] == "topk":
            k = min(self.cfg["agg_k"], flat.shape[1])
            return flat.topk(k, dim=1).values.mean(1, keepdim=True)
        # softmax-weighted mean: τ→∞ ⇒ max, τ→0 ⇒ mean; bounded, stable
        tau = self.cfg["agg_tau"]
        w = torch.softmax(tau * flat, dim=1)
        return (w * flat).sum(1, keepdim=True)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, batch):
        x = batch["diag"]
        climate, time, sat, sat_mask = batch["climate"], batch["time"], batch["sat"], batch["sat_mask"]
        N, _, H, W = x.shape
        cond = self._cond(time, climate)

        if self.cfg["broadcast_climate"]:
            x = torch.cat([x, climate[:, :, None, None].expand(N, self.climate_dim, H, W)], 1)

        # encoder
        skips, sizes = [], []
        h = x
        for block, film, down in zip(self.enc_blocks, self.enc_films, self.downs):
            h = block(h)
            if film is not None:
                h = film(h, cond)
            skips.append(h)
            sizes.append(h.shape[-2:])
            h = down(h)

        # satellite fusion at bottleneck
        if self.sat_emb_dim:
            emb = self.sat_mlp(torch.cat([sat, sat_mask], 1))      # (N, sat_hidden)
            emb = emb[:, :, None, None].expand(-1, -1, h.shape[-2], h.shape[-1])
            h = torch.cat([h, emb], 1)
        h = self.bottleneck(h)

        # decoder
        for dec, skip, size in zip(self.dec_blocks, reversed(skips), reversed(sizes)):
            h = F.interpolate(h, size=size, mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, skip], 1))

        # head (optional conditioning re-injection)
        if self.reinject_dim:
            rc = cond if self.cfg["fnn_reinject"] == "film" else climate
            h = torch.cat([h, rc[:, :, None, None].expand(-1, self.reinject_dim, H, W)], 1)

        field = F.softplus(self.head(h))            # (N,1,H,W) ≥ 0  — the heatmap
        max_hat = self.aggregate(field)             # (N,1) predicted max_edr
        mean_hat = field.mean(dim=(2, 3))           # (N,1) predicted mean_edr (aux)
        return {"field": field, "max_hat": max_hat, "mean_hat": mean_hat}


def build_model(cfg, in_channels, climate_dim, time_dim, sat_dim) -> CatCNNTorch:
    return CatCNNTorch(in_channels, climate_dim, time_dim, sat_dim, cfg)
