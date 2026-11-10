import numpy
import torch
import torch.nn.functional as F

from language_model.model import Embedding, Linear, RMSNorm, RotaryPositionalEmbedding, SwiGLU, silu


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
