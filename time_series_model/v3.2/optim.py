"""当前精简环境使用的标准 AdamW 实现。"""

from __future__ import annotations

import torch


class LocalAdamW:
    def __init__(
        self,
        parameters,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_number = 0
        self.first_moment = [torch.zeros_like(p) for p in self.parameters]
        self.second_moment = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.step_number += 1
        correction1 = 1.0 - self.beta1 ** self.step_number
        correction2 = (1.0 - self.beta2 ** self.step_number) ** 0.5
        for parameter, first, second in zip(self.parameters, self.first_moment, self.second_moment):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            denominator = second.sqrt().div_(correction2).add_(self.eps)
            parameter.addcdiv_(first, denominator, value=-self.lr / correction1)
