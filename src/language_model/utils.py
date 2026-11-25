import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(logits: Float[Tensor, " ... vocab"], targets: Int[Tensor, " ..."]) -> Float[Tensor, ""]:
    shifted = logits - logits.max(dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    nl_softmax_logits = log_sum_exp - torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1))
    return torch.mean(nl_softmax_logits)
