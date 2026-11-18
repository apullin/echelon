"""Unit tests for Echelon (ported from PyTorch PR #205345 and expanded).

Runs on CPU and, when available, on CUDA with the same tight tolerances.
"""

from copy import deepcopy

import pytest
import torch
from torch.nn import Parameter

from echelon import Echelon, echelon

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def assert_close(a, b, atol=1e-5, rtol=1e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
class TestEchelon:
    @pytest.mark.parametrize("nesterov", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_orthogonalization_does_not_alias_momentum_buffer(
        self, device, dtype, nesterov
    ):
        momentum = 0.9
        param = Parameter(torch.zeros(8, 5, device=device, dtype=dtype))
        param.grad = torch.ones_like(param)
        optimizer = Echelon(
            [param], lr=0, weight_decay=0, momentum=momentum, nesterov=nesterov, rank=2
        )
        expected = torch.zeros_like(param)
        expected.lerp_(param.grad, 1 - momentum)
        optimizer.step()
        assert_close(optimizer.state[param]["momentum_buffer"], expected)

    def test_exact_frame_matches_spectral_clip(self, device):
        def scalar_ns5(val: float) -> float:
            a, b, c = NS_COEFFS
            for _ in range(5):
                val = a * val + b * (val**3) + c * (val**5)
            return val

        sig = torch.tensor([4.0, 2.0, 0.5, 0.2], device=device, dtype=torch.float32)
        g = torch.zeros(4, 6, device=device, dtype=torch.float32)
        g[:, :4] = torch.diag(sig)
        p = Parameter(torch.zeros_like(g))
        buf = torch.zeros_like(g)
        u = torch.eye(4, 2, device=device, dtype=torch.float32)
        z = torch.zeros(6, 1, device=device, dtype=torch.float32)
        z[2, 0] = 1.0
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                echelon(
                    [p],
                    [g],
                    [buf],
                    [u],
                    [z],
                    lr=1.0,
                    weight_decay=0.0,
                    momentum=0.0,
                    nesterov=False,
                    rank=2,
                    tau_iters=2,
                    margin=1.05,
                    ns_coefficients=NS_COEFFS,
                    ns_steps=5,
                    eps=1e-7,
                    adjust_lr_fn="original",
                    has_complex=False,
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        head_norm = (4.0**2 + 2.0**2) ** 0.5
        expected_diag = torch.tensor(
            [
                -scalar_ns5(4.0 / head_norm),
                -scalar_ns5(2.0 / head_norm),
                -1.0 / 1.05,
                -(0.2 / 0.5) / 1.05,
            ],
            device=device,
            dtype=torch.float32,
        )
        assert_close(p[:, :4], torch.diag(expected_diag))
        assert_close(u.T @ u, torch.eye(2, device=device, dtype=torch.float32))

    def test_full_rank_and_sub_eps_consistency(self, device):
        torch.manual_seed(0)
        g = torch.randn(16, 32, device=device, dtype=torch.float32)
        p_ech = Parameter(torch.zeros_like(g))
        p_ech.grad = g.clone()
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            Echelon([p_ech], lr=0.02, weight_decay=0.0, rank=16).step()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        a, b, c = NS_COEFFS
        g64 = g.double()
        x = g64 / g64.norm()
        for _ in range(5):
            A = x @ x.T
            x = a * x + (b * A + c * A @ A) @ x
        assert_close(p_ech, -0.02 * x.float(), atol=1e-6, rtol=1e-4)

        g_sub = torch.diag(torch.tensor([2e-11, 0.0], device=device, dtype=torch.float32))
        p_low = Parameter(torch.zeros_like(g_sub))
        p_full = Parameter(torch.zeros_like(g_sub))
        with torch.no_grad():
            echelon(
                [p_low],
                [g_sub],
                [torch.zeros_like(g_sub)],
                [torch.tensor([[1.0], [0.0]], device=device)],
                [torch.tensor([[0.0], [1.0]], device=device)],
                lr=1.0,
                weight_decay=0.0,
                momentum=0.0,
                nesterov=False,
                rank=1,
                tau_iters=2,
                margin=1.05,
                ns_coefficients=NS_COEFFS,
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn="original",
                has_complex=False,
            )
            echelon(
                [p_full],
                [g_sub],
                [torch.zeros_like(g_sub)],
                [None],
                [None],
                lr=1.0,
                weight_decay=0.0,
                momentum=0.0,
                nesterov=False,
                rank=2,
                tau_iters=2,
                margin=1.05,
                ns_coefficients=NS_COEFFS,
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn="original",
                has_complex=False,
            )
        assert_close(p_low, p_full, atol=1e-7, rtol=1e-5)

    def test_transpose_symmetry_and_functional_parity(self, device):
        torch.manual_seed(1)
        p_wide = Parameter(torch.zeros(12, 20, device=device, dtype=torch.float32))
        p_tall = Parameter(torch.zeros(20, 12, device=device, dtype=torch.float32))
        g_wide = torch.randn(12, 20, device=device, dtype=torch.float32)
        p_wide.grad = g_wide.clone()
        p_tall.grad = g_wide.T.clone()
        opt_wide = Echelon(
            [p_wide], lr=0.05, weight_decay=0.0, rank=3, adjust_lr_fn="match_rms_adamw"
        )
        opt_tall = Echelon(
            [p_tall], lr=0.05, weight_decay=0.0, rank=3, adjust_lr_fn="match_rms_adamw"
        )
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
            echelon(
                [p_fn],
                [g_next.clone()],
                [buf_fn],
                [u_fn],
                [z_fn],
                lr=0.05,
                weight_decay=0.0,
                momentum=0.95,
                nesterov=True,
                rank=3,
                tau_iters=2,
                margin=1.05,
                ns_coefficients=NS_COEFFS,
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn="match_rms_adamw",
                has_complex=False,
            )
        assert_close(p_fn, p_wide)

    def test_rank_change_reinitializes_frame(self, device):
        torch.manual_seed(3)
        p = Parameter(torch.randn(16, 24, device=device, dtype=torch.float32))
        q = Parameter(torch.randn(16, 24, device=device, dtype=torch.float32))
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
        assert_close(u.T @ u, torch.eye(6, device=device, dtype=torch.float32), rtol=0)
        assert torch.isfinite(p).all()

    def test_state_preserves_float32_on_load_and_upcasts_precast_state(self, device):
        torch.manual_seed(5)
        p = Parameter(torch.randn(16, 24, device=device, dtype=torch.bfloat16))
        opt = Echelon([p], rank=4)
        p.grad = torch.randn_like(p)
        opt.step()

        # 1. Standard state_dict round-trip preserves exact float32 frame/tail precision.
        p_loaded = Parameter(p.detach().clone())
        opt_loaded = Echelon([p_loaded], rank=4)
        opt_loaded.load_state_dict(deepcopy(opt.state_dict()))
        assert opt_loaded.state[p_loaded]["subspace_frame"].dtype == torch.float32
        assert_close(
            opt_loaded.state[p_loaded]["subspace_frame"],
            opt.state[p]["subspace_frame"],
        )

        p.grad = torch.randn_like(p)
        p_loaded.grad = p.grad.clone()
        opt.step()
        opt_loaded.step()
        assert_close(p_loaded, p)
        for key in ("subspace_frame", "tail_vec"):
            assert opt_loaded.state[p_loaded][key].dtype == torch.float32
            assert_close(opt_loaded.state[p_loaded][key], opt.state[p][key])

        # 2. If an external checkpoint manually saved bfloat16 state, next step upcasts it.
        sd_bf16 = deepcopy(opt.state_dict())
        for st in sd_bf16["state"].values():
            st["subspace_frame"] = st["subspace_frame"].to(torch.bfloat16)
            st["tail_vec"] = st["tail_vec"].to(torch.bfloat16)
        p_bf16 = Parameter(p.detach().clone())
        opt_bf16 = Echelon([p_bf16], rank=4)
        opt_bf16.load_state_dict(sd_bf16)
        p_bf16.grad = torch.randn_like(p_bf16)
        opt_bf16.step()
        for key in ("subspace_frame", "tail_vec"):
            assert opt_bf16.state[p_bf16][key].dtype == torch.float32

    def test_zero_and_empty_params_are_noops(self, device):
        p_zero = Parameter(torch.ones(8, 12, device=device, dtype=torch.float32))
        p_empty = Parameter(torch.empty(0, 4, device=device, dtype=torch.float32))
        p_zero.grad = torch.zeros_like(p_zero)
        p_empty.grad = torch.empty_like(p_empty)
        opt = Echelon([p_zero, p_empty], lr=0.01, weight_decay=0.0, rank=2)
        opt.step()
        assert_close(p_zero, torch.ones_like(p_zero))
        assert "subspace_frame" not in opt.state[p_empty]

    def test_zero_first_update_keeps_cold_frame(self, device):
        torch.manual_seed(7)
        p = Parameter(torch.randn(8, 12, device=device, dtype=torch.float32))
        opt = Echelon([p], weight_decay=0.0, rank=2)
        p.grad = torch.zeros_like(p)
        opt.step()
        u = opt.state[p]["subspace_frame"]
        assert opt.state[p].get("warmed") is False
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        u0 = torch.randn((8, 2), generator=gen, dtype=torch.float32)
        torch.randn((12, 1), generator=gen, dtype=torch.float32)
        expected, _ = torch.linalg.qr(u0.to(device=device), mode="reduced")
        assert_close(u, expected.contiguous(), atol=1e-6, rtol=0)
        p.grad = torch.randn_like(p)
        opt.step()
        assert opt.state[p].get("warmed") is True
        assert not torch.equal(opt.state[p]["subspace_frame"], expected.contiguous())
        assert torch.isfinite(p).all()

    def test_init_under_default_device(self, device):
        with torch.device(device):
            p = Parameter(torch.randn(8, 12, dtype=torch.float32))
            p.grad = torch.randn_like(p)
            opt = Echelon([p], rank=2)
            opt.step()
        assert opt.state[p]["subspace_frame"].device.type == torch.device(device).type

    def test_state_is_contiguous(self, device):
        p = Parameter(torch.randn(16, 24, device=device, dtype=torch.float32))
        opt = Echelon([p], rank=4)
        p.grad = torch.randn_like(p)
        opt.step()
        frame = opt.state[p]["subspace_frame"]
        assert frame.is_contiguous(), f"subspace_frame strides {frame.stride()}"

    def test_deepcopy_before_first_step_matches_fresh_optimizer(self, device):
        torch.manual_seed(4)
        p = Parameter(torch.randn(8, 12, device=device, dtype=torch.float32))
        p_fresh = Parameter(p.detach().clone())
        p_copy, opt_copy = deepcopy((p, Echelon([p], lr=0.01, rank=2)))
        opt_fresh = Echelon([p_fresh], lr=0.01, rank=2)
        p_copy.grad = torch.randn_like(p)
        p_fresh.grad = p_copy.grad.clone()
        opt_copy.step()
        opt_fresh.step()
        assert_close(p_copy, p_fresh)
        assert_close(
            opt_copy.state[p_copy]["subspace_frame"],
            opt_fresh.state[p_fresh]["subspace_frame"],
        )

    def test_rank_deficient_momentum_damps_roundoff_tail(self, device):
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.manual_seed(4)
            q_l, _ = torch.linalg.qr(torch.randn(12, 12, device=device, dtype=torch.float32))
            q_r, _ = torch.linalg.qr(torch.randn(16, 16, device=device, dtype=torch.float32))
            sig = torch.diag(torch.tensor([4.0, 1.2], device=device, dtype=torch.float32))
            p = Parameter(torch.zeros(12, 16, device=device, dtype=torch.float32))
            p.grad = q_l[:, :2] @ sig @ q_r[:, :2].T
            Echelon(
                [p], lr=1.0, weight_decay=0.0, rank=4, adjust_lr_fn="spectral_unclamped"
            ).step()
            assert torch.linalg.svdvals(p)[2].item() <= 0.1
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    def test_ignores_ambient_autocast(self, device):
        if not torch.amp.autocast_mode.is_autocast_available(device):
            pytest.skip(f"autocast is not available on {device}")
        torch.manual_seed(6)
        g = torch.randn(12, 16, device=device, dtype=torch.float32)
        p_ref = Parameter(torch.zeros_like(g))
        p_ac = Parameter(torch.zeros_like(g))
        p_ref.grad = g.clone()
        p_ac.grad = g.clone()
        Echelon([p_ref], lr=1.0, rank=4).step()
        opt_ac = Echelon([p_ac], lr=1.0, rank=4)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            opt_ac.step()
        assert_close(p_ac, p_ref)

    def test_foreach_matches_forloop_on_batched_shapes(self, device):
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.manual_seed(2)
            shapes = [(24, 16)] * 3 + [(16, 24)] * 2 + [(12, 20)] + [(4, 6)] * 2 + [(0, 5)]
            base = [torch.randn(s, device=device, dtype=torch.float32) for s in shapes]
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
            eye = torch.eye(4, device=device, dtype=torch.float32)
            for a, b in zip(p_loop, p_each):
                assert_close(a, b)
                if "subspace_frame" in opt_loop.state[a]:
                    u_a = opt_loop.state[a]["subspace_frame"]
                    u_b = opt_each.state[b]["subspace_frame"]
                    assert_close(u_b.T @ u_b, eye, atol=1e-4, rtol=1e-4)
                    assert_close(u_a @ u_a.T, u_b @ u_b.T, atol=1e-4, rtol=1e-4)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    def test_foreach_rank_deficient_preserves_orthonormality_and_recovers(self, device):
        # Rank-1 / rank-4 momenta must NOT collapse subspace_frame columns to zero
        # on foreach=True; the frame stays orthonormal and rotates normally once
        # full-rank gradients return.
        torch.manual_seed(0)
        params = [
            Parameter(torch.randn(64, 128, device=device, dtype=torch.float32))
            for _ in range(2)
        ]
        before = [p.detach().clone() for p in params]
        opt = Echelon(
            params, lr=0.02, weight_decay=0.0, momentum=0.0, nesterov=False, rank=8, foreach=True
        )
        eye = torch.eye(8, device=device, dtype=torch.float32)
        # Step 1: full-rank warmup & step
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
        # Steps 2-3: exact rank-1 gradients (singular Gram matrix)
        for _ in range(2):
            for p in params:
                p.grad = torch.randn(64, 1, device=device) @ torch.randn(1, 128, device=device)
            opt.step()
        for p in params:
            u = opt.state[p]["subspace_frame"]
            assert torch.isfinite(p).all()
            assert_close(u.T @ u, eye, atol=1e-2, rtol=1e-2)
        # Step 4: full-rank gradient with distinct top-8 subspace -> frame must rotate
        u_prev = opt.state[params[0]]["subspace_frame"].clone()
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
        u_after = opt.state[params[0]]["subspace_frame"]
        assert_close(u_after.T @ u_after, eye, atol=1e-3, rtol=1e-3)
        assert not torch.allclose(u_after, u_prev)
        for p, p0 in zip(params, before):
            assert (p - p0).abs().max().item() > 0.0

    @pytest.mark.parametrize("foreach", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
    def test_dtypes_fp16_and_fp64(self, device, dtype, foreach):
        torch.manual_seed(11)
        ps = [Parameter(torch.randn(16, 24, device=device, dtype=dtype)) for _ in range(2)]
        opt = Echelon(ps, lr=0.02, weight_decay=0.01, rank=4, foreach=foreach)
        scale = 1e100 if dtype == torch.float64 else 1.0
        for p in ps:
            p.grad = torch.randn_like(p) * scale
        opt.step()
        for p in ps:
            assert p.dtype == dtype
            assert torch.isfinite(p.double()).all()
            assert opt.state[p]["subspace_frame"].dtype == torch.float32

    @pytest.mark.parametrize("foreach", [False, True])
    @pytest.mark.parametrize("rank", [4, 16])
    @pytest.mark.parametrize("scale_exp", [-30, 25])
    def test_extreme_gradient_scales_overflow_and_underflow(
        self, device, foreach, rank, scale_exp
    ):
        torch.manual_seed(21)
        ps = [
            Parameter(torch.zeros(16, 24, device=device, dtype=torch.float32))
            for _ in range(2)
        ]
        opt = Echelon(
            ps,
            lr=1.0,
            weight_decay=0.0,
            rank=rank,
            eps=0.0,
            foreach=foreach,
            adjust_lr_fn="spectral_unclamped",
        )
        for p in ps:
            p.grad = torch.randn_like(p) * (10.0**scale_exp)
        opt.step()
        for p in ps:
            assert torch.isfinite(p).all()
            sv0 = torch.linalg.svdvals(p)[0].item()
            assert 0.5 < sv0 < 1.5, f"expected O(1) spectral norm, got {sv0}"

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf")])
    @pytest.mark.parametrize("rank", [4, 16])
    def test_nonfinite_gradient_leaves_param_and_siblings_finite(
        self, device, bad_val, rank
    ):
        torch.manual_seed(31)
        ps = [
            Parameter(torch.randn(16, 24, device=device, dtype=torch.float32))
            for _ in range(2)
        ]
        opt = Echelon(ps, lr=0.02, weight_decay=0.0, rank=rank, foreach=True)
        for p in ps:
            p.grad = torch.randn_like(p)
        opt.step()
        p0_before = ps[0].detach().clone()
        p1_before = ps[1].detach().clone()
        ps[0].grad = torch.full_like(ps[0], bad_val)
        ps[1].grad = torch.randn_like(ps[1])
        opt.step()
        # Non-finite member receives zero update (no NaN written to param), sibling updates cleanly.
        assert_close(ps[0], p0_before)
        assert torch.isfinite(ps[1]).all()
        assert not torch.equal(ps[1], p1_before)

    @pytest.mark.parametrize("foreach", [False, True])
    def test_tensor_lr_and_cross_device_tensor_lr(self, device, foreach):
        torch.manual_seed(41)
        for lr_dev in ({"cpu", device}):
            for shape in ((), (1,)):
                ps = [
                    Parameter(torch.randn(16, 24, device=device, dtype=torch.float32))
                    for _ in range(2)
                ]
                ps_ref = [Parameter(p.detach().clone()) for p in ps]
                lr_t = torch.full(shape, 0.02, device=lr_dev, dtype=torch.float32)
                opt = Echelon(ps, lr=lr_t, weight_decay=0.01, rank=4, foreach=foreach)
                opt_ref = Echelon(
                    ps_ref, lr=0.02, weight_decay=0.01, rank=4, foreach=foreach
                )
                for p, pr in zip(ps, ps_ref):
                    g = torch.randn_like(p)
                    p.grad = g.clone()
                    pr.grad = g.clone()
                opt.step()
                opt_ref.step()
                for p, pr in zip(ps, ps_ref):
                    assert_close(p, pr)

    @pytest.mark.parametrize("foreach", [False, True])
    def test_noncontiguous_params_and_grads(self, device, foreach):
        torch.manual_seed(51)
        ps = [
            Parameter(torch.randn(24, 16, device=device, dtype=torch.float32).T)
            for _ in range(2)
        ]
        assert not ps[0].is_contiguous()
        opt = Echelon(ps, lr=0.02, rank=4, foreach=foreach)
        for _ in range(2):
            for p in ps:
                p.grad = torch.randn(24, 16, device=device, dtype=torch.float32).T
            opt.step()
        for p in ps:
            assert torch.isfinite(p).all()

    @pytest.mark.parametrize("foreach", [False, True])
    def test_torch_compile_parity(self, device, foreach):
        torch.manual_seed(61)
        shapes = [(16, 24), (16, 24), (24, 16)]
        ps_e = [torch.randn(s, device=device) for s in shapes]
        ps_c = [p.clone() for p in ps_e]
        gs = [torch.randn(s, device=device) for s in shapes]
        buf_e = [torch.zeros_like(p) for p in ps_e]
        buf_c = [torch.zeros_like(p) for p in ps_c]
        u_e = [
            torch.linalg.qr(torch.randn(16, 4, device=device))[0].contiguous()
            for _ in shapes
        ]
        u_c = [u.clone() for u in u_e]
        z_e = [torch.randn(24, 1, device=device) for _ in shapes]
        z_c = [z.clone() for z in z_e]
        kw = {
            "foreach": foreach,
            "lr": 0.02,
            "weight_decay": 0.01,
            "momentum": 0.95,
            "nesterov": True,
            "rank": 4,
            "tau_iters": 2,
            "margin": 1.05,
            "ns_coefficients": NS_COEFFS,
            "ns_steps": 5,
            "eps": 1e-7,
            "adjust_lr_fn": "original",
            "has_complex": False,
        }
        compiled_fn = torch.compile(echelon)
        with torch.no_grad():
            echelon(ps_e, gs, buf_e, u_e, z_e, **kw)
            compiled_fn(ps_c, [g.clone() for g in gs], buf_c, u_c, z_c, **kw)
        for a, b in zip(ps_e, ps_c):
            assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_cuda_steady_state_sync_free_and_cudagraph_safe(foreach, tensor_lr):
    device = torch.device("cuda")
    torch.manual_seed(71)
    ps = [Parameter(torch.randn(32, 64, device=device)) for _ in range(3)]
    grads = [torch.randn_like(p) for p in ps]
    for p, g in zip(ps, grads):
        p.grad = g
    lr = torch.tensor(0.02, device=device) if tensor_lr else 0.02
    opt = Echelon(ps, lr=lr, weight_decay=0.01, rank=8, foreach=foreach)
    opt.step()  # warmup step (frame init)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        opt.step()
    finally:
        torch.cuda.set_sync_debug_mode("default")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        opt.step()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()
    if isinstance(lr, torch.Tensor):
        lr.fill_(0.01)
    for g in grads:
        g.normal_()
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.isfinite(p).all() for p in ps)


class TestEchelonArgumentValidation:
    """Generic (device-independent) argument and input validation."""

    @pytest.mark.parametrize(
        "mutation, regex",
        [
            ({"rank": 0}, "rank must be an integer >= 1"),
            ({"adjust_lr_fn": "bogus"}, "bogus is not supported"),
            ({"momentum": 1.0}, "momentum must be a finite number in \\[0, 1\\)"),
            ({"margin": float("inf")}, "margin must be finite"),
            ({"ns_coefficients": ("3.4", "-4.7", "2.0")}, "exactly 3 finite numbers"),
        ],
    )
    def test_invalid_group_option_steps_no_group(self, mutation, regex):
        p0 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        p1 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        opt = Echelon([{"params": [p0]}, {"params": [p1]}], rank=2)
        p0.grad, p1.grad = torch.randn_like(p0), torch.randn_like(p1)
        before = p0.detach().clone()
        opt.param_groups[1].update(mutation)
        with pytest.raises(ValueError, match=regex):
            opt.step()
        assert_close(p0, before)
        assert len(opt.state) == 0

    def test_add_param_group_rejects_invalid_groups(self):
        opt = Echelon([Parameter(torch.zeros(4, 4))])
        bad_groups = {
            "Echelon only supports 2D parameters": {},
            "match_rms_adam is not supported": {"adjust_lr_fn": "match_rms_adam"},
            "rank must be an integer >= 1": {"rank": 0},
            "momentum must be a finite number": {"momentum": 1.0},
        }
        for regex, options in bad_groups.items():
            w = Parameter(torch.zeros((4,) if not options else (4, 4)))
            with pytest.raises(ValueError, match=regex):
                opt.add_param_group({"params": [w], **options})
            assert len(opt.param_groups) == 1

    def test_functional_rejects_unknown_adjust_lr_fn(self):
        p = torch.zeros(4, 4)
        with pytest.raises(ValueError, match="match_rms_adam is not supported"):
            echelon(
                [p],
                [p.clone()],
                [torch.zeros_like(p)],
                [None],
                [None],
                lr=0.01,
                weight_decay=0.0,
                momentum=0.9,
                nesterov=True,
                ns_coefficients=NS_COEFFS,
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn="match_rms_adam",
                has_complex=False,
                rank=4,
                tau_iters=2,
                margin=1.05,
            )

    @pytest.mark.parametrize("foreach", [False, True])
    def test_functional_shape_and_length_validation_parity(self, foreach):
        p = torch.zeros(4, 4)
        # Empty params with non-empty grads must fail on both foreach=False and True.
        with pytest.raises(ValueError, match="matching lengths"):
            echelon(
                [],
                [p.clone()],
                [],
                [],
                [],
                foreach=foreach,
                lr=0.01,
                weight_decay=0.0,
                momentum=0.9,
                nesterov=True,
                ns_coefficients=NS_COEFFS,
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn=None,
                has_complex=False,
                rank=4,
                tau_iters=2,
                margin=1.05,
            )
        # Broadcastable (1, 4) grad must be rejected rather than silently broadcast.
        with pytest.raises(ValueError, match="shape"):
            echelon(
                [p],
                [torch.zeros(1, 4)],
                [torch.zeros_like(p)],
                [None],
                [None],
                foreach=foreach,
                lr=0.01,
                weight_decay=0.0,
                momentum=0.9,
                nesterov=True,
                ns_coefficients=NS_COEFFS,
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn=None,
                has_complex=False,
                rank=4,
                tau_iters=2,
                margin=1.05,
            )

    def test_constructor_rejects_nonfinite_or_string_ns_coefficients(self):
        for bad in (
            (float("nan"), 0.0, 0.0),
            ("a", 0.0, 0.0),
            ("3.4", "-4.7", "2.0"),
            (True, 0.0, 0.0),
            (1.0, 2.0),
        ):
            with pytest.raises(ValueError, match="exactly 3 finite numbers"):
                Echelon([Parameter(torch.zeros(4, 4))], ns_coefficients=bad)

    def test_constructor_rejects_non_bool_foreach(self):
        for bad in ("yes", 1):
            with pytest.raises(ValueError, match="foreach must be"):
                Echelon([Parameter(torch.zeros(4, 4))], foreach=bad)

    def test_sparse_grad_in_later_group_steps_no_group(self):
        p0 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        p1 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        opt = Echelon([{"params": [p0]}, {"params": [p1]}], rank=2)
        p0.grad = torch.randn_like(p0)
        p1.grad = torch.randn_like(p1).to_sparse()
        before = p0.detach().clone()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()
        assert_close(p0, before)
        assert len(opt.state) == 0

    def test_closure_invalid_grad_in_later_group_steps_no_group(self):
        p0 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        p1 = Parameter(torch.randn(8, 12, dtype=torch.float32))
        opt = Echelon([{"params": [p0]}, {"params": [p1]}], rank=2)
        before = p0.detach().clone()

        def closure():
            p0.grad = torch.randn_like(p0)
            p1.grad = torch.randn_like(p1).to_sparse()
            return torch.tensor(1.0)

        with pytest.raises(RuntimeError, match="sparse"):
            opt.step(closure)
        assert_close(p0, before)
        assert len(opt.state) == 0

    def test_constructor_rejects_non_2d_and_bad_options(self):
        w = Parameter(torch.zeros(4, 4))
        with pytest.raises(ValueError, match="Echelon only supports 2D"):
            Echelon([Parameter(torch.zeros(4, 4, 4))])
        with pytest.raises(ValueError, match="rank must be an integer"):
            Echelon([w], rank=0)
        with pytest.raises(ValueError, match="is not supported"):
            Echelon([w], adjust_lr_fn="bogus")
        with pytest.raises(ValueError, match="momentum"):
            Echelon([w], momentum=1.0)
        with pytest.raises(ValueError, match="momentum"):
            Echelon([w], momentum=-0.1)
        with pytest.raises(ValueError, match="weight decay"):
            Echelon([w], weight_decay=float("inf"))
        with pytest.raises(ValueError, match="margin"):
            Echelon([w], margin=float("inf"))
        with pytest.raises(ValueError, match="eps"):
            Echelon([w], eps=float("inf"))
        with pytest.raises(ValueError, match="Learning rate"):
            Echelon([w], lr=-1e-3)
        with pytest.raises(ValueError, match="1-element"):
            Echelon([w], lr=torch.tensor([0.01, 0.02]))


def test_dtensor_params_rejected_early():
    """DTensor params raise NotImplementedError at construction, not deep in step()."""
    import os

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    try:
        dist.init_process_group("gloo", rank=0, world_size=1)
    except (RuntimeError, OSError):
        pytest.skip("could not start a gloo process group")
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Replicate

        mesh = init_device_mesh("cpu", (1,))
        dt = DTensor.from_local(torch.randn(4, 4), mesh, [Replicate()])
        with pytest.raises(NotImplementedError, match="DTensor"):
            Echelon([Parameter(dt)], rank=2)
    finally:
        dist.destroy_process_group()
