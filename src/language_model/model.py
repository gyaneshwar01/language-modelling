import math

import einx
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
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
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

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
        return result.to(x.dtype)


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    m = x.max(dim, keepdim=True)[0]
    counts = torch.exp(x - m)
    return counts / counts.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))

    att_wts = softmax(scores, dim=-1)
    return einsum(att_wts, V, " ... queries keys, ... keys d_v -> ... queries d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v, device=device, dtype=dtype)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model, device=device, dtype=dtype)

        self.positional_encoder = positional_encoder

    def forward(
        self, x: Float[Tensor, " ... seq d_model"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_model"]:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q, K, V = (rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads) for X in (Q, K, V))

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = rearrange(attn_out, "... heads seq d_v -> ... seq (heads d_v)")

        output = self.output_proj(attn_out)

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, dtype=dtype, device=device)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, positional_encoder, dtype=dtype, device=device)
        self.ln2 = RMSNorm(d_model, dtype=dtype, device=device)
        self.ffn = SwiGLU(d_model, d_ff, dtype=dtype, device=device)

    def forward(
        self, x: Float[Tensor, " ... seq d_model"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_model"]:
        x_ln1 = self.ln1(x)
        x_attn = self.attn(x_ln1, token_positions)
        attn_out = x + x_attn

        x_ln2 = self.ln2(attn_out)
        x_ffn = self.ffn(x_ln2)
        return attn_out + x_ffn


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        d_k = d_model // num_heads
        self.positional_encoder = (
            RotaryPositionalEmbedding(theta, d_k, context_length, device) if theta is not None else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    self.positional_encoder,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, x: Int[Tensor, "... seq"], token_positions: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq_len vocab"]:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x, token_positions)  # (B, seq, d_model)

        x = self.ln_final(x)  # (B, seq, d_model)
        logits = self.lm_head(x)  # (B, seq, vocab)

        return logits
