"""Tests ported from PyTorch PR #205345 (torch/test/test_optim.py).

Run in the uv env: ``uv run --no-sync pytest`` (torch is installed with
``uv pip`` from the CPU wheel index; see README).
"""

from copy import deepcopy

import pytest
import torch
from torch.nn import Parameter

from echelon import Echelon, echelon

DEVICE = "cpu"
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def assert_close(a, b, atol=1e-5, rtol=1e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


class TestEchelon:
    @pytest.mark.parametrize("nesterov", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_orthogonalization_does_not_alias_momentum_buffer(self, dtype, nesterov):
        momentum = 0.9
        param = Parameter(torch.zeros(8, 5, dtype=dtype))
        param.grad = torch.ones_like(param)
        optimizer = Echelon(
            [param], lr=0, weight_decay=0, momentum=momentum, nesterov=nesterov, rank=2
        )
        expected = torch.zeros_like(param)
        expected.lerp_(param.grad, 1 - momentum)
        optimizer.step()
        assert_close(optimizer.state[param]["momentum_buffer"], expected)

    def test_exact_frame_matches_spectral_clip(self):
        def scalar_ns5(val: float) -> float:
            a, b, c = NS_COEFFS
            for _ in range(5):
                val = a * val + b * (val**3) + c * (val**5)
            return val

        sig = torch.tensor([4.0, 2.0, 0.5, 0.2], dtype=torch.float32)
        g = torch.zeros(4, 6, dtype=torch.float32)
        g[:, :4] = torch.diag(sig)
        p = Parameter(torch.zeros_like(g))
        buf = torch.zeros_like(g)
        u = torch.eye(4, 2, dtype=torch.float32)
        z = torch.zeros(6, 1, dtype=torch.float32)
        z[2, 0] = 1.0
        with torch.no_grad():
            echelon(
                [p], [g], [buf], [u], [z],
                lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False,
                rank=2, tau_iters=2, margin=1.05, ns_coefficients=NS_COEFFS,
                ns_steps=5, eps=1e-7, adjust_lr_fn="original", has_complex=False,
            )
        head_norm = (4.0**2 + 2.0**2) ** 0.5
        expected_diag = torch.tensor(
            [-scalar_ns5(4.0 / head_norm), -scalar_ns5(2.0 / head_norm),
             -1.0 / 1.05, -(0.2 / 0.5) / 1.05], dtype=torch.float32)
        assert_close(p[:, :4], torch.diag(expected_diag))
        assert_close(u.T @ u, torch.eye(2, dtype=torch.float32))

    def test_full_rank_and_sub_eps_consistency(self):
        torch.manual_seed(0)
        g = torch.randn(16, 32, dtype=torch.float32)
        p_ech = Parameter(torch.zeros_like(g))
        p_ech.grad = g.clone()
        Echelon([p_ech], lr=0.02, weight_decay=0.0, rank=16).step()
        a, b, c = NS_COEFFS
        g64 = g.double()
        x = g64 / g64.norm()
        for _ in range(5):
            A = x @ x.T
            x = a * x + (b * A + c * A @ A) @ x
        assert_close(p_ech, -0.02 * x.float(), atol=1e-6, rtol=1e-4)

        g_sub = torch.diag(torch.tensor([2e-11, 0.0], dtype=torch.float32))
        p_low = Parameter(torch.zeros_like(g_sub))
        p_full = Parameter(torch.zeros_like(g_sub))
        with torch.no_grad():
            echelon([p_low], [g_sub], [torch.zeros_like(g_sub)],
                    [torch.tensor([[1.0], [0.0]])], [torch.tensor([[0.0], [1.0]])],
                    lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False,
                    rank=1, tau_iters=2, margin=1.05, ns_coefficients=NS_COEFFS,
                    ns_steps=5, eps=1e-7, adjust_lr_fn="original", has_complex=False)
            echelon([p_full], [g_sub], [torch.zeros_like(g_sub)], [None], [None],
                    lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False,
                    rank=2, tau_iters=2, margin=1.05, ns_coefficients=NS_COEFFS,
                    ns_steps=5, eps=1e-7, adjust_lr_fn="original", has_complex=False)
        assert_close(p_low, p_full, atol=1e-7, rtol=1e-5)

    def test_transpose_symmetry_and_functional_parity(self):
        torch.manual_seed(1)
        p_wide = Parameter(torch.zeros(12, 20, dtype=torch.float32))
        p_tall = Parameter(torch.zeros(20, 12, dtype=torch.float32))
        g_wide = torch.randn(12, 20, dtype=torch.float32)
        p_wide.grad = g_wide.clone()
        p_tall.grad = g_wide.T.clone()
        opt_wide = Echelon([p_wide], lr=0.05, weight_decay=0.0, rank=3,
                           adjust_lr_fn="match_rms_adamw")
        opt_tall = Echelon([p_tall], lr=0.05, weight_decay=0.0, rank=3,
                           adjust_lr_fn="match_rms_adamw")
        opt_wide.step()
        opt_tall.step()
        assert_close(p_tall, p_wide.T)

        p_fn = Parameter(p_wide.detach().clone())
        buf_fn = opt_wide.state[p_wide]["momentum_buffer"].clone()
        u_fn = opt_wide.state[p_wide]["subspace_frame"].clone()
        z_fn = opt_wide.state[p_wide]["tail_vec"].clone()
        g_next = torch.randn_like(p_wide)
        p_wide.grad = g_next.clone()
        opt_wide.step()
        with torch.no_grad():
            echelon([p_fn], [g_next.clone()], [buf_fn], [u_fn], [z_fn],
                    lr=0.05, weight_decay=0.0, momentum=0.95, nesterov=True,
                    rank=3, tau_iters=2, margin=1.05, ns_coefficients=NS_COEFFS,
                    ns_steps=5, eps=1e-7, adjust_lr_fn="match_rms_adamw",
                    has_complex=False)
        assert_close(p_fn, p_wide)

    def test_rank_change_reinitializes_frame(self):
        torch.manual_seed(3)
        p = Parameter(torch.randn(16, 24, dtype=torch.float32))
        q = Parameter(torch.randn(16, 24, dtype=torch.float32))
        opt = Echelon([p, q], rank=4)
        p.grad = torch.randn_like(p)
        q.grad = torch.randn_like(q)
        opt.step()
        state = opt.state[p]
        frame = state["subspace_frame"].clone()
        for bad_rank in (0, -1):
            opt.param_groups[0]["rank"] = bad_rank
            with pytest.raises(ValueError, match="rank must be an integer >= 1"):
                opt.step()
        assert_close(state["subspace_frame"], frame)

        q.grad = None
        opt.param_groups[0]["rank"] = 16
        opt.step()
        assert "subspace_frame" not in state
        assert "subspace_frame" not in opt.state[q]
        opt.param_groups[0]["rank"] = 6
        opt.step()
        u = state["subspace_frame"]
        assert_close(u.T @ u, torch.eye(6, dtype=torch.float32), rtol=0)
        assert torch.isfinite(p).all()

    def test_invalid_group_option_steps_no_group(self):
        p0 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        p1 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        opt = Echelon([{"params": [p0]}, {"params": [p1]}], rank=2)
        p0.grad, p1.grad = torch.randn_like(p0), torch.randn_like(p1)
        before = p0.detach().clone()
        opt.param_groups[1]["rank"] = 0
        with pytest.raises(ValueError, match="rank must be an integer >= 1"):
            opt.step()
        assert_close(p0, before)
        assert len(opt.state) == 0

    def test_state_returns_to_float32_after_load(self):
        torch.manual_seed(5)
        p = Parameter(torch.randn(16, 24, dtype=torch.bfloat16))
        opt = Echelon([p], rank=4)
        p.grad = torch.randn_like(p)
        opt.step()
        p_loaded = Parameter(p.detach().clone())
        opt_loaded = Echelon([p_loaded], rank=4)
        opt_loaded.load_state_dict(deepcopy(opt.state_dict()))
        # The loaded run continues from the saved state, rounded once to the
        # parameter dtype (torch <2.16 has no preserve_state_dtype_on_load).
        for key in ("subspace_frame", "tail_vec"):
            opt.state[p][key] = opt.state[p][key].to(torch.bfloat16).float()
        p.grad = torch.randn_like(p)
        p_loaded.grad = p.grad.clone()
        opt.step()
        opt_loaded.step()
        assert_close(p_loaded, p)
        for key in ("subspace_frame", "tail_vec"):
            loaded = opt_loaded.state[p_loaded][key]
            assert loaded.dtype == torch.float32
            assert_close(loaded, opt.state[p][key])

    def test_zero_and_empty_params_are_noops(self):
        p_zero = Parameter(torch.ones(8, 12, dtype=torch.float32))
        p_empty = Parameter(torch.empty(0, 4, dtype=torch.float32))
        p_zero.grad = torch.zeros_like(p_zero)
        p_empty.grad = torch.empty_like(p_empty)
        opt = Echelon([p_zero, p_empty], lr=0.01, weight_decay=0.0, rank=2)
        opt.step()
        assert_close(p_zero, torch.ones_like(p_zero))
        assert "subspace_frame" not in opt.state[p_empty]

    def test_zero_first_update_keeps_cold_frame(self):
        torch.manual_seed(7)
        p = Parameter(torch.randn(8, 12, dtype=torch.float32))
        opt = Echelon([p], weight_decay=0.0, rank=2)
        p.grad = torch.zeros_like(p)
        opt.step()
        u = opt.state[p]["subspace_frame"]
        assert_close(u.T @ u, torch.eye(2, dtype=torch.float32), rtol=0)
        p.grad = torch.randn_like(p)
        opt.step()
        assert torch.isfinite(p).all()

    def test_init_under_default_device(self):
        with torch.device(DEVICE):
            p = Parameter(torch.randn(8, 12, dtype=torch.float32))
            p.grad = torch.randn_like(p)
            opt = Echelon([p], rank=2)
            opt.step()
        assert opt.state[p]["subspace_frame"].device == p.device

    def test_state_is_contiguous(self):
        p = Parameter(torch.randn(16, 24, dtype=torch.float32))
        opt = Echelon([p], rank=4)
        for _ in range(2):
            p.grad = torch.randn_like(p)
            opt.step()
            for key in ("subspace_frame", "tail_vec"):
                assert opt.state[p][key].is_contiguous()

    def test_deepcopy_before_first_step_matches_fresh_optimizer(self):
        torch.manual_seed(4)
        p = Parameter(torch.randn(8, 12, dtype=torch.float32))
        p_fresh = Parameter(p.detach().clone())
        p_copy, opt_copy = deepcopy((p, Echelon([p], lr=0.01, rank=2)))
        opt_fresh = Echelon([p_fresh], lr=0.01, rank=2)
        p_copy.grad = torch.randn_like(p)
        p_fresh.grad = p_copy.grad.clone()
        opt_copy.step()
        opt_fresh.step()
        assert_close(p_copy, p_fresh)
        assert_close(opt_copy.state[p_copy]["subspace_frame"],
                     opt_fresh.state[p_fresh]["subspace_frame"])

    def test_rank_deficient_momentum_damps_roundoff_tail(self):
        torch.manual_seed(4)
        q_l, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float32))
        q_r, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float32))
        sig = torch.diag(torch.tensor([4.0, 1.2], dtype=torch.float32))
        p = Parameter(torch.zeros(12, 16, dtype=torch.float32))
        p.grad = q_l[:, :2] @ sig @ q_r[:, :2].T
        Echelon([p], lr=1.0, weight_decay=0.0, rank=4,
                adjust_lr_fn="spectral_unclamped").step()
        # Without the roundoff floor this singular value is about 0.82.
        assert torch.linalg.svdvals(p)[2].item() <= 0.1

    def test_ignores_ambient_autocast(self):
        if not torch.amp.autocast_mode.is_autocast_available("cpu"):
            pytest.skip("autocast is not available on cpu")
        torch.manual_seed(6)
        g = torch.randn(12, 16, dtype=torch.float32)
        p_ref = Parameter(torch.zeros_like(g))
        p_ac = Parameter(torch.zeros_like(g))
        p_ref.grad = g.clone()
        p_ac.grad = g.clone()
        Echelon([p_ref], lr=1.0, rank=4).step()
        opt_ac = Echelon([p_ac], lr=1.0, rank=4)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            opt_ac.step()
        assert_close(p_ac, p_ref)

    def test_foreach_matches_forloop_on_batched_shapes(self):
        # Same-shape params are stepped as one batch with Cholesky-QR, including a
        # zero-gradient member, an NS-only (rank >= short side) group and an empty param.
        torch.manual_seed(2)
        shapes = [(24, 16)] * 3 + [(16, 24)] * 2 + [(4, 6)] * 2 + [(0, 5)]
        base = [torch.randn(s, dtype=torch.float32) for s in shapes]
        p_loop = [Parameter(b.clone()) for b in base]
        p_each = [Parameter(b.clone()) for b in base]
        kwargs = {"lr": 0.02, "weight_decay": 0.01, "rank": 4}
        opt_loop = Echelon(p_loop, foreach=False, **kwargs)
        opt_each = Echelon(p_each, foreach=True, **kwargs)
        for _ in range(5):
            for i, (a, b) in enumerate(zip(p_loop, p_each)):
                a.grad = torch.zeros_like(a) if i == 2 else torch.randn_like(a)
                b.grad = a.grad.clone()
            opt_loop.step()
            opt_each.step()
        assert_close(p_each[2], base[2] * (1 - 0.02 * 0.01) ** 5)
        eye = torch.eye(4, dtype=torch.float32)
        for a, b in zip(p_loop, p_each):
            assert_close(a, b)
            if "subspace_frame" in opt_loop.state[a]:
                u_a = opt_loop.state[a]["subspace_frame"]
                u_b = opt_each.state[b]["subspace_frame"]
                assert_close(u_b.T @ u_b, eye)
                assert_close(u_a @ u_a.T, u_b @ u_b.T, atol=1e-4, rtol=1e-4)

    def test_foreach_rank_deficient_momentum_stays_finite(self):
        # Rank-4 momenta make the batched Cholesky-QR fail on CUDA; those params keep their frame.
        torch.manual_seed(0)
        params = [Parameter(torch.randn(512, 2048, dtype=torch.float32)) for _ in range(2)]
        before = [p.detach().clone() for p in params]
        opt = Echelon(params, rank=64, foreach=True)
        for _ in range(4):
            for p in params:
                p.grad = torch.randn(512, 4) @ torch.randn(4, 2048)
            opt.step()
        for p, p0 in zip(params, before):
            assert torch.isfinite(p).all()
            assert torch.isfinite(opt.state[p]["subspace_frame"]).all()
            assert (p - p0).abs().max().item() > 0.0


class TestEchelonArgumentValidation:
    """Generic (device-independent) argument validation."""

    def test_add_param_group_rejects_invalid_groups(self):
        opt = Echelon([Parameter(torch.zeros(4, 4))])
        bad_groups = {
            "Echelon only supports 2D parameters": {},
            "match_rms_adam is not supported": {"adjust_lr_fn": "match_rms_adam"},
            "rank must be an integer >= 1": {"rank": 0},
        }
        for regex, options in bad_groups.items():
            w = Parameter(torch.zeros((4,) if not options else (4, 4)))
            with pytest.raises(ValueError, match=regex):
                opt.add_param_group({"params": [w], **options})
            assert len(opt.param_groups) == 1

    def test_functional_rejects_unknown_adjust_lr_fn(self):
        p = torch.zeros(4, 4)
        with pytest.raises(ValueError, match="match_rms_adam is not supported"):
            echelon([p], [p.clone()], [torch.zeros_like(p)], [None], [None],
                    lr=0.01, weight_decay=0.0, momentum=0.9, nesterov=True,
                    ns_coefficients=NS_COEFFS, ns_steps=5, eps=1e-7,
                    adjust_lr_fn="match_rms_adam", has_complex=False,
                    rank=4, tau_iters=2, margin=1.05)

    def test_constructor_rejects_non_2d_and_bad_options(self):
        with pytest.raises(ValueError, match="Echelon only supports 2D"):
            Echelon([Parameter(torch.zeros(4, 4, 4))])
        with pytest.raises(ValueError, match="rank must be an integer"):
            Echelon([Parameter(torch.zeros(4, 4))], rank=0)
        with pytest.raises(ValueError, match="is not supported"):
            Echelon([Parameter(torch.zeros(4, 4))], adjust_lr_fn="bogus")
