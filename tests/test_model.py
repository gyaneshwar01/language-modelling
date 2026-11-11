import numpy
import torch
import torch.nn.functional as F
from einops import rearrange

from language_model.model import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    scaled_dot_product_attention,
    silu,
    softmax,
)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    weights = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    linear = Linear(d_model, d_ff, device=weights.device, dtype=weights.dtype)
    linear.load_state_dict({"weight": weights})
    numpy_snapshot.assert_match(linear(in_embeddings))


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    weights = ts_state_dict[0]["token_embeddings.weight"]
    embedding = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    embedding.load_state_dict({"weight": weights})
    numpy_snapshot.assert_match(embedding(in_indices))


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    swiglu = SwiGLU(d_model, d_ff)
    for k in ["w1", "w2", "w3"]:
        getattr(swiglu, k).load_state_dict({"weight": ts_state_dict[0][f"layers.0.ffn.{k}.weight"]})
    numpy_snapshot.assert_match(swiglu(in_embeddings), atol=1e-5)


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    weights = ts_state_dict[0]["layers.1.ln1.weight"]
    rms_norm = RMSNorm(weights.shape[0], eps=1e-5, device=weights.device, dtype=weights.dtype)
    rms_norm.load_state_dict({"weight": weights})
    numpy_snapshot.assert_match(rms_norm(in_embeddings), atol=1e-4)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    rope = RotaryPositionalEmbedding(theta, d_model, n_queries, device=in_embeddings.device)
    numpy_snapshot.assert_match(rope(in_embeddings, pos_ids), atol=1e-5)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-5)


def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-5)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        softmax(x + 100, dim=-1).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-5,
    )


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    numpy_snapshot.assert_match(scaled_dot_product_attention(q, k, v, mask), atol=1e-5)


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)
    numpy_snapshot.assert_match(scaled_dot_product_attention(q, k, v, mask), atol=1e-5)


def load_attention_weights(attn: CausalMultiHeadSelfAttention, state_dict: dict):
    for k in ["q", "k", "v", "output"]:
        getattr(attn, f"{k}_proj").load_state_dict({"weight": state_dict[f"layers.0.attn.{k}_proj.weight"]})


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    attn = CausalMultiHeadSelfAttention(d_model, n_heads)
    load_attention_weights(attn, ts_state_dict[0])
    numpy_snapshot.assert_match(attn(in_embeddings), atol=1e-5)


def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    rope = RotaryPositionalEmbedding(theta, d_model // n_heads, n_keys)
    attn = CausalMultiHeadSelfAttention(d_model, n_heads, rope)
    load_attention_weights(attn, ts_state_dict[0])
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    numpy_snapshot.assert_match(attn(in_embeddings, pos_ids), atol=1e-5)
