"""Ecosystem test: use Echelon the way a downstream user would.

A small transformer-ish model trained for a few steps with the
Echelon + AdamW pair, an LR scheduler, grad clipping, autocast, and a
checkpoint/resume round-trip. This is the test that would have caught
a package that is importable but not painlessly usable (wrong param
partition, a scheduler that cannot read the param groups, a state
dict that does not resume, ...). CPU-only; GPU behaviour of the same
code path is covered by the foreach/for-loop parity test in
test_echelon.py only insofar as the code is device-agnostic.
"""

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from echelon import Echelon, echelon_adamw_pair, partition_parameters


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


def _train(opt_ech, opt_adam, model, steps=6, seed=0, schedulers=None):
    torch.manual_seed(seed)
    data = torch.randint(0, 64, (16, 8))
    if schedulers is None:
        schedulers = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_ech, T_max=100),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=100),
        )
    sched_ech, sched_adam = schedulers
    losses = []
    for _ in range(steps):
        opt_ech.zero_grad()
        opt_adam.zero_grad()
        logits = model(data[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_ech.step()
        opt_adam.step()
        sched_ech.step()
        sched_adam.step()
        losses.append(loss.item())
    return losses, schedulers


def test_partition_parameters():
    model = ToyModel()
    hidden, rest = partition_parameters(model)
    hidden_ids = {id(p) for p in hidden}
    hidden_names = {n for n, p in model.named_parameters() if id(p) in hidden_ids}
    rest_names = {n for n, p in model.named_parameters() if id(p) not in hidden_ids}
    assert hidden_names == {"block1.weight", "block2.weight"}
    assert rest_names == {
        "embed.weight", "block1.bias", "block2.bias",
        "norm.weight", "norm.bias", "head.weight",
    }
    # Also accepts an explicit (name, param) iterable.
    hidden2, _ = partition_parameters(model.named_parameters())
    assert len(hidden2) == len(hidden)


def test_echelon_adamw_pair_trains_and_loss_falls():
    torch.manual_seed(0)
    model = ToyModel()
    opt_ech, opt_adam = echelon_adamw_pair(model)
    assert isinstance(opt_ech, Echelon)
    losses, _ = _train(opt_ech, opt_adam, model, steps=8)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0], losses


@pytest.mark.parametrize("foreach", [False, True])
def test_pair_trains_with_foreach(foreach):
    torch.manual_seed(0)
    model = ToyModel()
    opt_ech, opt_adam = echelon_adamw_pair(
        model, echelon_kwargs={"foreach": foreach}
    )
    losses, _ = _train(opt_ech, opt_adam, model, steps=4)
    assert losses[-1] < losses[0]


def test_checkpoint_resume_matches_uninterrupted_run():
    # Same seed + same data order: checkpointing mid-run must reproduce
    # the uninterrupted trajectory, including scheduler state (the
    # schedulers are part of the checkpoint; a fresh scheduler would
    # restart the LR schedule and mask a bad optimizer resume).
    torch.manual_seed(0)
    model_a = ToyModel()
    opt_a_e, opt_a_a = echelon_adamw_pair(model_a)
    losses_a, sched_a = _train(opt_a_e, opt_a_a, model_a, steps=4, seed=1)
    buf = io.BytesIO()
    torch.save(
        {
            "model": model_a.state_dict(),
            "echelon": opt_a_e.state_dict(),
            "adamw": opt_a_a.state_dict(),
            "sched_ech": sched_a[0].state_dict(),
            "sched_adam": sched_a[1].state_dict(),
        },
        buf,
    )
    losses_a2, _ = _train(opt_a_e, opt_a_a, model_a, steps=4, seed=1, schedulers=sched_a)
    losses_a += losses_a2

    torch.manual_seed(0)
    model_b = ToyModel()
    opt_b_e, opt_b_a = echelon_adamw_pair(model_b)
    _, sched_b = _train(opt_b_e, opt_b_a, model_b, steps=4, seed=1)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)
    model_b.load_state_dict(ckpt["model"])
    opt_b_e.load_state_dict(ckpt["echelon"])
    opt_b_a.load_state_dict(ckpt["adamw"])
    sched_b[0].load_state_dict(ckpt["sched_ech"])
    sched_b[1].load_state_dict(ckpt["sched_adam"])
    losses_b, _ = _train(opt_b_e, opt_b_a, model_b, steps=4, seed=1, schedulers=sched_b)

    assert losses_a[4:] == pytest.approx(losses_b, abs=1e-6)
    # The continuation must actually move the parameters (a zero LR
    # after a finished cosine schedule would make this pass vacuously).
    assert losses_a[-1] != losses_a[3]
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(pa, pb)


def test_model_parameters_directly_gives_actionable_error():
    model = ToyModel()
    with pytest.raises(ValueError, match="partition_parameters"):
        Echelon(model.parameters())


def test_autocast_training_loop():
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("autocast is not available on cpu")
    torch.manual_seed(0)
    model = ToyModel()
    opt_ech, opt_adam = echelon_adamw_pair(model)
    data = torch.randint(0, 64, (8, 8))
    for _ in range(3):
        opt_ech.zero_grad()
        opt_adam.zero_grad()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(data[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1)
            )
        loss.backward()
        opt_ech.step()
        opt_adam.step()
    assert all(torch.isfinite(p.float()).all() for p in model.parameters())
