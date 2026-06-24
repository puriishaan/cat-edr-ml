"""
CatCNNTorch — PyTorch U-Net with FiLM conditioning for CAT turbulence.

Input streams:
  diag    : (B, C_diag, H, W)  physics diagnostic channels
  climate : (B, 4)              climate indices
  time    : (B, 4)              cyclic time features
  sat     : (B, 5)              satellite TB stats
  sat_mask: (B, 1)              satellite availability

Output:
  field   : (B, 1, H, W)  spatial turbulence intensity map
  max_hat : (B,)            event max intensity (aggregated)
  mean_hat: (B,)            event mean intensity (aggregated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CFG = dict(
    grid_size       = 24,
    depth           = 3,         # encoder stages
    base_filters    = 32,
    kernel_size     = 3,
    dropout         = 0.1,
    norm            = "batch",   # batch / group / none
    pool            = "max",     # max / avg / stride
    act             = "gelu",    # relu / gelu / silu
    film            = True,
    sat_fusion      = True,
    aggregation     = "logsumexp",  # logsumexp / topk
    topk            = 4,
)


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[name]


def _norm(name: str, channels: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "group":
        g = min(8, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, channels)
    return nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, norm_type="batch", act_type="gelu", drop=0.0):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            _norm(norm_type, out_ch),
            _act(act_type),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, k, padding=pad, bias=False),
            _norm(norm_type, out_ch),
            _act(act_type),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale/shift from conditioning vector."""

    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.gen = nn.Linear(cond_dim, 2 * n_channels)
        nn.init.zeros_(self.gen.weight)
        nn.init.zeros_(self.gen.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  cond: (B, cond_dim)
        params = self.gen(cond)                         # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma[:, :, None, None] + 1.0           # identity init
        beta  = beta[:, :, None, None]
        return x * gamma + beta


def _downsample(channels: int, pool_type: str) -> nn.Module:
    if pool_type == "max":
        return nn.MaxPool2d(2)
    if pool_type == "avg":
        return nn.AvgPool2d(2)
    return nn.Conv2d(channels, channels, 2, stride=2)   # stride conv


class CatCNNTorch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        climate_dim: int = 4,
        time_dim: int = 4,
        sat_dim: int = 5,
        cfg: dict = None,
    ):
        super().__init__()
        c = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg = c

        cond_dim  = climate_dim + time_dim
        sat_out   = 16 if c["sat_fusion"] else 0
        enc_chs   = [c["base_filters"] * (2 ** i) for i in range(c["depth"])]

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_films  = nn.ModuleList()
        self.pools      = nn.ModuleList()
        in_ch = in_channels
        for out_ch in enc_chs:
            self.enc_blocks.append(
                ConvBlock(in_ch, out_ch, c["kernel_size"],
                          c["norm"], c["act"], c["dropout"])
            )
            self.enc_films.append(FiLM(cond_dim, out_ch) if c["film"] else nn.Identity())
            self.pools.append(_downsample(out_ch, c["pool"]))
            in_ch = out_ch

        # Bottleneck
        bot_ch = enc_chs[-1] * 2
        self.bottleneck = ConvBlock(in_ch, bot_ch, c["kernel_size"],
                                     c["norm"], c["act"], c["dropout"])
        self.bot_film = FiLM(cond_dim, bot_ch) if c["film"] else nn.Identity()

        # Satellite MLP fused at bottleneck
        if c["sat_fusion"]:
            self.sat_mlp = nn.Sequential(
                nn.Linear(sat_dim + 1, 32), nn.GELU(),
                nn.Linear(32, sat_out),
            )
            bot_ch_with_sat = bot_ch + sat_out
        else:
            self.sat_mlp = None
            bot_ch_with_sat = bot_ch

        # Decoder
        self.dec_ups    = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_in = bot_ch_with_sat
        for skip_ch in reversed(enc_chs):
            out_ch = skip_ch
            self.dec_ups.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                               nn.Conv2d(dec_in, out_ch, 1))
            )
            self.dec_blocks.append(
                ConvBlock(out_ch + skip_ch, out_ch, c["kernel_size"],
                          c["norm"], c["act"], c["dropout"])
            )
            dec_in = out_ch

        # Head: 1×1 conv → softplus
        self.head = nn.Sequential(
            nn.Conv2d(dec_in, 1, 1),
            nn.Softplus(),
        )

    def aggregate(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,1,H,W) → (max_hat (B,), mean_hat (B,))"""
        flat = field.view(field.size(0), -1)   # (B, H*W)
        if self.cfg["aggregation"] == "logsumexp":
            # log-mean-exp: logsumexp - log(N) ≈ max for sparse fields, mean for flat fields
            max_hat = torch.logsumexp(flat, dim=1) - torch.log(torch.tensor(float(flat.size(1))))
        else:
            k = min(self.cfg["topk"], flat.size(1))
            max_hat = flat.topk(k, dim=1).values.mean(1)
        mean_hat = flat.mean(1)
        return max_hat, mean_hat

    def forward(self, batch: dict) -> dict:
        x        = batch["diag"]        # (B, C, H, W)
        climate  = batch["climate"]     # (B, 4)
        time_f   = batch["time"]        # (B, 4)
        sat      = batch["sat"]         # (B, 5)
        sat_mask = batch["sat_mask"]    # (B, 1)

        cond = torch.cat([climate, time_f], dim=1)   # (B, cond_dim)

        # Encoder
        skips = []
        for block, film, pool in zip(self.enc_blocks, self.enc_films, self.pools):
            x = block(x)
            x = film(x, cond) if self.cfg["film"] else x
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.bot_film(x, cond) if self.cfg["film"] else x

        # Satellite fusion at bottleneck
        if self.sat_mlp is not None:
            sat_in  = torch.cat([sat, sat_mask], dim=1)   # (B, 6)
            sat_vec = self.sat_mlp(sat_in)                 # (B, sat_out)
            H, W    = x.shape[2], x.shape[3]
            sat_map = sat_vec[:, :, None, None].expand(-1, -1, H, W)
            x = torch.cat([x, sat_map], dim=1)

        # Decoder
        for up, block, skip in zip(self.dec_ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            # handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        field    = self.head(x)                    # (B, 1, H, W)
        max_hat, mean_hat = self.aggregate(field)

        return {"field": field, "max_hat": max_hat, "mean_hat": mean_hat}


def build_model(cfg: dict, in_channels: int, climate_dim: int = 4,
                time_dim: int = 4, sat_dim: int = 5) -> CatCNNTorch:
    return CatCNNTorch(in_channels, climate_dim, time_dim, sat_dim, cfg)
