"""Ecosystem tests: use Echelon the way downstream PyTorch applications do.

Covers both low-rank (rank < short_side, with repeated same-shape matrices
for batched foreach Cholesky-QR) and full-rank training, checkpoint/resume
with weights_only=True, real-world parameter partitioning (GPT-2 wte/wpe +
tied lm_head, Llama output.weight, PyTorch multihead_attn, BERT dense,
shared explicit iterables, frozen weights, single-partition models), and
packaging/typing metadata.
"""

import io
import typing
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import echelon
from echelon import Echelon, echelon_adamw_pair, partition_parameters

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


class ToyModel(nn.Module):
    def __init__(self, d=32, vocab=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "proj": nn.Linear(d, d, bias=False),
                        "fc1": nn.Linear(d, 4 * d),
                        "fc2": nn.Linear(4 * d, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.layers:
            h = h + blk["proj"](h)
            h = h + blk["fc2"](F.gelu(blk["fc1"](h)))
        return self.head(self.norm(h))


def _train(opt_ech, opt_adam, model, steps=6, seed=0, schedulers=None):
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    data = torch.randint(0, 64, (16, 8), device=device)
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


def test_partition_parameters_basic_and_real_architectures():
    model = ToyModel(n_layers=2)
    hidden, _rest = partition_parameters(model)
    hidden_ids = {id(p) for p in hidden}
    hidden_names = {n for n, p in model.named_parameters() if id(p) in hidden_ids}
    rest_names = {n for n, p in model.named_parameters() if id(p) not in hidden_ids}
    assert hidden_names == {
        "layers.0.proj.weight",
        "layers.0.fc1.weight",
        "layers.0.fc2.weight",
        "layers.1.proj.weight",
        "layers.1.fc1.weight",
        "layers.1.fc2.weight",
    }
    assert rest_names == {
        "embed.weight",
        "layers.0.fc1.bias",
        "layers.0.fc2.bias",
        "layers.1.fc1.bias",
        "layers.1.fc2.bias",
        "norm.weight",
        "norm.bias",
        "head.weight",
    }
    hidden2, _ = partition_parameters(model.named_parameters())
    assert len(hidden2) == len(hidden)

    # GPT-2 / nanoGPT: wte, wpe, and tied lm_head must all land in rest (deduplicated).
    class GPT2Like(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(100, 32)
            self.wpe = nn.Embedding(16, 32)
            self.c_attn = nn.Linear(32, 96, bias=False)
            self.lm_head = nn.Linear(32, 100, bias=False)
            self.wte.weight = self.lm_head.weight

    gpt = GPT2Like()
    h_gpt, r_gpt = partition_parameters(gpt)
    assert len(h_gpt) == 1 and h_gpt[0] is gpt.c_attn.weight
    assert len(r_gpt) == 2 and set(map(id, r_gpt)) == {
        id(gpt.wte.weight),
        id(gpt.wpe.weight),
    }

    # Llama output.weight -> rest, BERT attention.output.dense.weight -> hidden.
    class LlamaAndBertLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = nn.Embedding(100, 32)
            self.attention = nn.ModuleDict(
                {"output": nn.ModuleDict({"dense": nn.Linear(32, 32, bias=False)})}
            )
            self.output = nn.Linear(32, 100, bias=False)

    lb = LlamaAndBertLike()
    h_lb, r_lb = partition_parameters(lb)
    assert h_lb == [lb.attention["output"]["dense"].weight]
    assert set(map(id, r_lb)) == {id(lb.tok_embeddings.weight), id(lb.output.weight)}

    # PyTorch nn.TransformerDecoderLayer: multihead_attn 2D weights must stay in hidden!
    dec = nn.TransformerDecoderLayer(d_model=32, nhead=4)
    h_dec, r_dec = partition_parameters(dec)
    h_dec_names = {
        n for n, p in dec.named_parameters() if id(p) in {id(x) for x in h_dec}
    }
    assert "multihead_attn.in_proj_weight" in h_dec_names
    assert "multihead_attn.out_proj.weight" in h_dec_names
    assert set(map(id, h_dec)).isdisjoint(set(map(id, r_dec)))


def test_partition_shared_explicit_iterable_and_frozen():
    w_shared = nn.Parameter(torch.zeros(8, 8))
    w_hidden = nn.Parameter(torch.zeros(8, 8))
    w_frozen = nn.Parameter(torch.zeros(8, 8), requires_grad=False)
    hidden, rest = partition_parameters(
        [
            ("block.weight", w_shared),
            ("lm_head.weight", w_shared),
            ("mlp.weight", w_hidden),
            ("frozen.weight", w_frozen),
        ]
    )
    assert hidden == [w_hidden]
    assert rest == [w_shared]
    assert set(map(id, hidden)).isdisjoint(set(map(id, rest)))

    hidden_inc, _ = partition_parameters(
        [("mlp.weight", w_hidden), ("frozen.weight", w_frozen)],
        include_frozen=True,
    )
    assert len(hidden_inc) == 2


def test_echelon_adamw_pair_single_partition_and_empty_model():
    # Bias-free Linear has no AdamW params; LayerNorm has no Echelon params.
    for mod in (nn.Linear(16, 16, bias=False), nn.LayerNorm(16)):
        opt_e, opt_a = echelon_adamw_pair(mod, echelon_kwargs={"rank": 4})
        sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=10)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=10)
        x = torch.randn(4, 16)
        opt_e.zero_grad()
        opt_a.zero_grad()
        mod(x).sum().backward()
        opt_e.step()
        opt_a.step()
        sched_e.step()
        sched_a.step()

    with pytest.raises(ValueError, match="no parameters to optimize"):
        echelon_adamw_pair(nn.Identity())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("rank", [4, 64])
def test_pair_trains_low_rank_and_full_rank(device, foreach, rank):
    torch.manual_seed(0)
    model = ToyModel(d=32, n_layers=2).to(device)
    opt_ech, opt_adam = echelon_adamw_pair(
        model, echelon_kwargs={"foreach": foreach, "rank": rank}
    )
    assert isinstance(opt_ech, Echelon)
    losses, _ = _train(opt_ech, opt_adam, model, steps=6)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0], losses
    for p in opt_ech.param_groups[0]["params"]:
        if rank < min(p.shape):
            assert "subspace_frame" in opt_ech.state[p]
        else:
            assert "subspace_frame" not in opt_ech.state[p]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_resume_matches_uninterrupted_run(device, dtype):
    torch.manual_seed(0)
    model_a = ToyModel(d=32, n_layers=2).to(device=device, dtype=dtype)
    opt_a_e, opt_a_a = echelon_adamw_pair(model_a, echelon_kwargs={"rank": 4})
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
    losses_a2, _ = _train(
        opt_a_e, opt_a_a, model_a, steps=4, seed=2, schedulers=sched_a
    )
    losses_a += losses_a2

    # Build model_b fresh (different initial weights, never pre-stepped) and load with weights_only=True.
    torch.manual_seed(99)
    model_b = ToyModel(d=32, n_layers=2).to(device=device, dtype=dtype)
    opt_b_e, opt_b_a = echelon_adamw_pair(model_b, echelon_kwargs={"rank": 4})
    sched_b = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_b_e, T_max=100),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_b_a, T_max=100),
    )
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=True)
    model_b.load_state_dict(ckpt["model"])
    opt_b_e.load_state_dict(ckpt["echelon"])
    opt_b_a.load_state_dict(ckpt["adamw"])
    sched_b[0].load_state_dict(ckpt["sched_ech"])
    sched_b[1].load_state_dict(ckpt["sched_adam"])
    losses_b, _ = _train(
        opt_b_e, opt_b_a, model_b, steps=4, seed=2, schedulers=sched_b
    )

    tol = 1e-5 if dtype == torch.float32 else 1e-3
    assert losses_a[4:] == pytest.approx(losses_b, abs=tol)
    assert losses_a[-1] != losses_a[3]
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(pa, pb, atol=tol, rtol=tol)


def test_model_parameters_directly_gives_actionable_error():
    model = ToyModel()
    with pytest.raises(ValueError, match="partition_parameters"):
        Echelon(model.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_training_loop(device):
    if not torch.amp.autocast_mode.is_autocast_available(device):
        pytest.skip(f"autocast is not available on {device}")
    torch.manual_seed(0)
    model = ToyModel(d=32, n_layers=2).to(device)
    opt_ech, opt_adam = echelon_adamw_pair(model, echelon_kwargs={"rank": 4})
    data = torch.randint(0, 64, (8, 8), device=device)
    for _ in range(3):
        opt_ech.zero_grad()
        opt_adam.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(data[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1)
            )
        loss.backward()
        opt_ech.step()
        opt_adam.step()
    assert all(torch.isfinite(p.float()).all() for p in model.parameters())


def test_typing_metadata_and_type_hints():
    hints = typing.get_type_hints(echelon_adamw_pair)
    assert "return" in hints
    py_typed = Path(echelon.__file__).with_name("py.typed")
    assert py_typed.is_file()
