from __future__ import annotations
import math
import torch
import torch.nn as nn
from dataclasses import dataclass

class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int, act: str = "relu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        assert act in acts, f"Unknown act={act}"
        layers = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), acts[act]]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

@dataclass
class TSData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

def make_teacher_student_regression(
    n_train: int = 512,
    n_test: int = 2048,
    d_in: int = 20,
    teacher_width: int = 256,
    teacher_depth: int = 2,
    noise_std: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> TSData:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x_train = torch.randn(n_train, d_in, generator=g)
    x_test  = torch.randn(n_test, d_in, generator=g)

    teacher = MLP(d_in, teacher_width, teacher_depth, act="tanh")
    # freeze teacher weights deterministically
    torch.manual_seed(seed + 12345)
    for p in teacher.parameters():
        nn.init.normal_(p, mean=0.0, std=0.5)

    with torch.no_grad():
        y_train_clean = teacher(x_train)
        y_test_clean  = teacher(x_test)

        eps = noise_std * torch.randn(n_train, generator=g)
        y_train = y_train_clean + eps
        y_test  = y_test_clean  # test is noiseless by default (clean target)

    return TSData(
        x_train.to(device),
        y_train.to(device),
        x_test.to(device),
        y_test.to(device),
    )

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)