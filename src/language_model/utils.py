from collections.abc import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(logits: Float[Tensor, " ... vocab"], targets: Int[Tensor, " ..."]) -> Float[Tensor, ""]:
    shifted = logits - logits.max(dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    nl_softmax_logits = log_sum_exp - torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1))
    return torch.mean(nl_softmax_logits)


def clip_gradient(parameters: Iterable[torch.nn.parameter.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        total_norm += (g**2).sum()
    total_norm = torch.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
