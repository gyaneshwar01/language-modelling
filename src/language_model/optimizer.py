import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if betas[0] < 0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # Decoupled weight array
                p.mul_(1 - lr * decay)

                # Moment updates (in place)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)  # beta1 * m + (1 - beta1) * grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # beta2 * v + (1 - beta2) * grad * grad

                # Bias-corrected step size
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Parameter update:
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)  # p = p - alpha_t * m/(sqrt(v) + eps)

        return loss
