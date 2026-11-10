import math

import einx
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2 / (out_features + in_features))
        wts = torch.empty((self.out_features, self.in_features), dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(wts, 0, std, -3 * std, 3 * std)
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(wts)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        wts = torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype)
        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(torch.nn.init.trunc_normal_(wts, 0, 1, -3, 3))

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        cur_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x / torch.sqrt(x.square().sum(-1, keepdim=True) / self.d_model + self.eps)) * self.weight
        return result.to(cur_dtype)


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryPositionalEmbedding._init_cache(theta, d_k, max_seq_len).to(device),
            persistent=False,
        )
        self._freq_cis_cache: Float[Tensor, "2 max_seq_len half_dim"]

    @staticmethod
    def _init_cache(theta: float, d_k: int, max_seq_len: int) -> Float[Tensor, " 2 max_seq_len half_dim"]:
        assert d_k % 2 == 0

        d = torch.arange(0, d_k, 2) / d_k
        freqs = torch.tensor(theta) ** -d
        t = torch.arange(max_seq_len)

        angles = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(angles), torch.sin(angles)
        return torch.stack((cos, sin))

    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)

        if pos_ids is not None:
            cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids)
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)

        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2

        result = rearrange(torch.stack((x1_rot, x2_rot)), "xy ... half_d -> ... (half_d xy)")
        return result
