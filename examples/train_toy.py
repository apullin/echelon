"""Minimal end-to-end example: train a tiny model with Echelon + AdamW.

Run from the repo root in the uv env (see README):

    uv run --no-sync python examples/train_toy.py

On CPU this takes a few seconds and the loss should fall. The model is
deliberately tiny; this is a usability smoke test, not a benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from echelon import echelon_adamw_pair


class ToyModel(nn.Module):
    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.block1 = nn.Linear(d, 4 * d)
        self.block2 = nn.Linear(4 * d, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        h = F.gelu(self.block1(h))
        h = self.block2(h)
        return self.head(self.norm(h))


def main() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    opt_echelon, opt_adamw = echelon_adamw_pair(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_echelon, T_max=50)
    data = torch.randint(0, 64, (32, 8))
    for step in range(20):
        opt_echelon.zero_grad()
        opt_adamw.zero_grad()
        logits = model(data[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_echelon.step()
        opt_adamw.step()
        sched.step()
        if step % 5 == 0 or step == 19:
            print(f"step {step:2d}  loss {loss.item():.4f}")
    print("done")


if __name__ == "__main__":
    main()
