"""
CatCNN — Pure NumPy 2-D CNN for CAT turbulence intensity prediction.

Architecture
------------
Input : (N, C, H, W)  multi-channel ERA5 spatial field
        C = 6 variables × 3 jet-stream levels = 18 channels
        H = W = 24 (resized to fixed grid)

Conv1  : 18 → 16 filters, 3×3, pad=1  → (N,16,24,24) → ReLU
MaxPool: 2×2                           → (N,16,12,12)
Conv2  : 16 → 32 filters, 3×3, pad=1  → (N,32,12,12) → ReLU
MaxPool: 2×2                           → (N,32, 6, 6)
GAP    : Global Average Pool           → (N,32)
FC1    : 32 → 32                       → ReLU
FC2    : 32 →  1                       → Sigmoid

Output : scalar ∈ [0, 1] representing normalised EDR (turbulence intensity).
         0 = no turbulence, 1 = maximum observed severity.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# im2col / col2im
# ─────────────────────────────────────────────────────────────────────────────

def _im2col(x: np.ndarray, kH: int, kW: int, pad: int = 0, stride: int = 1):
    """Return (col, outH, outW) where col has shape (N, outH*outW, C*kH*kW)."""
    N, C, H, W = x.shape
    outH = (H + 2 * pad - kH) // stride + 1
    outW = (W + 2 * pad - kW) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    s = xp.strides
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, outH, outW, kH, kW),
        strides=(s[0], s[1], stride * s[2], stride * s[3], s[2], s[3]),
    )
    col = np.ascontiguousarray(patches.transpose(0, 2, 3, 1, 4, 5)).reshape(
        N, outH * outW, C * kH * kW
    )
    return col, outH, outW


def _col2im(
    dcol: np.ndarray, x_shape: tuple, kH: int, kW: int, pad: int = 0, stride: int = 1
) -> np.ndarray:
    """Scatter-add col gradients back into spatial domain."""
    N, C, H, W = x_shape
    outH = (H + 2 * pad - kH) // stride + 1
    outW = (W + 2 * pad - kW) // stride + 1
    d = dcol.reshape(N, outH, outW, C, kH, kW)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=dcol.dtype)
    for h in range(outH):
        for w in range(outW):
            xp[:, :, h * stride : h * stride + kH, w * stride : w * stride + kW] += d[:, h, w]
    return xp[:, :, pad : H + pad, pad : W + pad] if pad > 0 else xp


# ─────────────────────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────────────────────

class Conv2D:
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, pad: int = 1):
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = np.random.randn(out_ch, in_ch, k, k).astype(np.float32) * scale
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.pad = pad
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._cache: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        F, _, kH, kW = self.W.shape
        col, outH, outW = _im2col(x, kH, kW, self.pad)
        out = (col @ self.W.reshape(F, -1).T).transpose(0, 2, 1).reshape(N, F, outH, outW)
        out += self.b[None, :, None, None]
        self._cache = (x, col)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, col = self._cache
        N, F, outH, outW = dout.shape
        _, _, kH, kW = self.W.shape
        dl = dout.reshape(N, F, -1).transpose(0, 2, 1)          # (N, L, F)
        self.dW = np.einsum("nlf,nlc->fc", dl, col).reshape(self.W.shape)
        self.db = dout.sum(axis=(0, 2, 3))
        dcol = np.einsum("nlf,fc->nlc", dl, self.W.reshape(F, -1))
        return _col2im(dcol, x.shape, kH, kW, self.pad)


class MaxPool2D:
    def __init__(self, size: int = 2):
        self.size = size
        self._cache: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        s = self.size
        oH, oW = H // s, W // s
        out = np.zeros((N, C, oH, oW), dtype=x.dtype)
        for i in range(oH):
            for j in range(oW):
                out[:, :, i, j] = x[:, :, i * s : (i + 1) * s, j * s : (j + 1) * s].max(
                    axis=(2, 3)
                )
        self._cache = (x, oH, oW)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, oH, oW = self._cache
        s = self.size
        dx = np.zeros_like(x)
        for i in range(oH):
            for j in range(oW):
                patch = x[:, :, i * s : (i + 1) * s, j * s : (j + 1) * s]
                pm = patch.max(axis=(2, 3), keepdims=True)
                mask = (patch == pm).astype(float)
                cnt = mask.sum(axis=(2, 3), keepdims=True).clip(min=1)
                dx[:, :, i * s : (i + 1) * s, j * s : (j + 1) * s] += (
                    mask / cnt * dout[:, :, i : i + 1, j : j + 1]
                )
        return dx


class GlobalAvgPool:
    """Collapses H×W to a scalar per channel: (N,C,H,W) → (N,C)."""

    def __init__(self):
        self._shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = self._shape
        return np.broadcast_to(dout[:, :, None, None] / (H * W), (N, C, H, W)).copy()


class ReLU:
    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class Linear:
    def __init__(self, in_f: int, out_f: int):
        self.W = (np.random.randn(in_f, out_f) * np.sqrt(2.0 / in_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


class Sigmoid:
    def __init__(self):
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-x.clip(-500, 500)))
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        s = self._out
        return dout * s * (1.0 - s)


# ─────────────────────────────────────────────────────────────────────────────
# CatCNN model
# ─────────────────────────────────────────────────────────────────────────────

class CatCNN:
    """
    2-D convolutional network for CAT turbulence intensity regression.
    Pure NumPy — no external ML frameworks required.
    """

    def __init__(self, in_channels: int = 18):
        self.layers = [
            Conv2D(in_channels, 16, k=3, pad=1),   # → (N,16,H,W)
            ReLU(),
            MaxPool2D(2),                            # → (N,16,H/2,W/2)
            Conv2D(16, 32, k=3, pad=1),             # → (N,32,H/2,W/2)
            ReLU(),
            MaxPool2D(2),                            # → (N,32,H/4,W/4)
            GlobalAvgPool(),                         # → (N,32)
            Linear(32, 32),
            ReLU(),
            Linear(32, 1),
            Sigmoid(),
        ]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x  # (N, 1)

    def backward(self, dout: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    # ------------------------------------------------------------------
    # Parameter iteration (for Adam updates)
    # ------------------------------------------------------------------

    def _learnable(self):
        return [l for l in self.layers if isinstance(l, (Conv2D, Linear))]

    def update(
        self,
        lr: float,
        m: dict,
        v: dict,
        t: int,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Adam update with optional L2 regularisation."""
        for i, layer in enumerate(self._learnable()):
            for name in ("W", "b"):
                g = getattr(layer, "d" + name).astype(np.float64)
                if name == "W":
                    g = g + weight_decay * getattr(layer, name).astype(np.float64)
                key = (i, name)
                m[key] = beta1 * m.get(key, 0.0) + (1 - beta1) * g
                v[key] = beta2 * v.get(key, 0.0) + (1 - beta2) * g ** 2
                m_hat = m[key] / (1 - beta1 ** t)
                v_hat = v[key] / (1 - beta2 ** t)
                param = getattr(layer, name)
                param -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(param.dtype)

    # ------------------------------------------------------------------
    # Save / load weights
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        weights = {}
        for i, layer in enumerate(self._learnable()):
            weights[f"l{i}_W"] = layer.W
            weights[f"l{i}_b"] = layer.b
        np.savez(path, **weights)

    def load(self, path: str) -> None:
        data = np.load(path)
        for i, layer in enumerate(self._learnable()):
            layer.W = data[f"l{i}_W"].astype(np.float32)
            layer.b = data[f"l{i}_b"].astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns (N,) array of turbulence intensities in [0, 1]."""
        return self.forward(x).ravel()
