from __future__ import annotations
import time
import torch
import torch.nn as nn

@torch.no_grad()
def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> float:
    model.eval()
    n = x.shape[0]
    total = 0.0
    for i in range(0, n, batch_size):
        xb = x[i:i+batch_size]
        yb = y[i:i+batch_size]
        pred = model(xb)
        total += torch.mean((pred - yb) ** 2).item() * xb.shape[0]
    return total / n

def train_regression(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    batch_size: int = 128,
    steps: int = 4000,
    log_every: int = 200,
    device: str = "cpu",
):
    model.to(device)
    model.train()

    if optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer={optimizer}")

    loss_fn = nn.MSELoss()
    n = x_train.shape[0]
    t0 = time.time()
    history = {"step": [], "train_mse": [], "test_mse": []}

    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb = x_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == 1 or step == steps:
            tr = mse(model, x_train, y_train)
            te = mse(model, x_test, y_test)
            history["step"].append(step)
            history["train_mse"].append(tr)
            history["test_mse"].append(te)

    return history, time.time() - t0