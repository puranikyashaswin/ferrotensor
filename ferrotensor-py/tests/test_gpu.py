"""GPU parity tests; run with FERROTENSOR_RUN_GPU_TESTS=1 (the old FERROTENSOR_GPU_TESTS=1 also works).

They need a GPU adapter; CI runs them on software Vulkan (Mesa lavapipe).
"""

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F
from conftest import close
from ferrotensor import nn, optim

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gpu():
    return ft.Device("gpu")


def test_device(gpu):
    assert gpu.is_gpu and gpu.type == "gpu" and gpu == ft.Device.gpu()
    assert gpu.name and gpu.backend != "Cpu"
    assert ft.zeros(2, device=gpu).device == gpu


def test_ops_match_cpu(gpu):
    a = ft.randn(4, 5)
    b = ft.randn(5, 3)
    ga, gb = a.to(gpu), b.to(gpu)
    close((ga @ gb).tolist(), (a @ b).tolist(), tol=1e-4)
    close((ga.relu() * 2 + 1).softmax(-1).tolist(), (a.relu() * 2 + 1).softmax(-1).tolist(), tol=1e-5)
    close(ga.sum(1).tolist(), a.sum(1).tolist(), tol=1e-4)
    close(ga[1:, ::2].tolist(), a[1:, ::2].tolist())


def test_grads_match_cpu(gpu):
    w = ft.randn(3, 4)
    wc = w.detach().requires_grad_()
    wg = w.to(gpu).requires_grad_()
    (wc.tanh() @ wc.T).sum().backward()
    (wg.tanh() @ wg.T).sum().backward()
    assert wg.grad.device == gpu
    close(wg.grad.tolist(), wc.grad.tolist(), tol=1e-4)


def test_device_mismatch_raises(gpu):
    with pytest.raises(ValueError, match="same device"):
        ft.ones(2) + ft.ones(2, device=gpu)


def test_train_on_gpu(gpu):
    x = ft.randn(64, 4, device=gpu)
    y = (x.sum(1) > 0).tolist()
    model = nn.Sequential(nn.Linear(4, 16, device=gpu), nn.ReLU(), nn.Linear(16, 2, device=gpu))
    opt = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_save_load_gpu(gpu, tmp_path):
    model = nn.Linear(3, 2, device=gpu)
    model.save(tmp_path / "m.safetensors")
    other = nn.Linear(3, 2, device=gpu)
    other.load(tmp_path / "m.safetensors")
    assert other.weight.tolist() == model.weight.tolist()
    assert other.weight.device == gpu


# ---------------------------------------------------------------------------
# CPU-vs-GPU parity. Inputs are fixed (not random) so a failure is repeatable,
# and they avoid ties and the kinks of relu, sign and clamp.
# ---------------------------------------------------------------------------

X = [[0.5, -1.2, 2.0], [1.5, 0.3, -0.7]]
Y = [[0.9, 0.4, -1.3], [-0.6, 1.1, 0.8]]
POS = [[0.5, 1.2, 2.0], [1.5, 0.3, 0.7]]
V = [0.7, -0.4, 1.3]
M = [[1.0, 2.0, 0.5], [0.3, -1.0, 2.0], [1.5, 0.2, -0.8]]


def as_list(out):
    return list(out) if isinstance(out, (list, tuple)) else [out]


def parity(gpu, fn, *inputs, tol=1e-4):
    """Run ``fn`` on CPU and GPU copies of ``inputs`` and compare every output."""
    want = as_list(fn(*[ft.tensor(x) for x in inputs]))
    got = as_list(fn(*[ft.tensor(x, device=gpu) for x in inputs]))
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.device == gpu
        assert g.shape == w.shape
        close(g.tolist(), w.tolist(), tol=tol)


def grad_parity(gpu, fn, *inputs, tol=1e-4):
    """Compare the gradient of ``sum(fn(...) ** 2)`` on CPU and GPU for every input."""

    def run(device):
        leaves = [ft.tensor(x, device=device, requires_grad=True) for x in inputs]
        out = fn(*leaves)
        (out * out).sum().backward()
        return leaves

    for g, c in zip(run(gpu), run(None), strict=True):
        assert g.grad is not None and g.grad.device == gpu
        close(g.grad.tolist(), c.grad.tolist(), tol=tol)


UNARY = ["relu", "gelu", "silu", "sigmoid", "tanh", "exp", "abs", "neg", "sign", "square", "sin", "cos"]
UNARY_POSITIVE = ["log", "sqrt", "rsqrt", "reciprocal"]


@pytest.mark.parametrize("name", UNARY)
def test_unary_matches_cpu(gpu, name):
    parity(gpu, lambda a: getattr(a, name)(), X)


@pytest.mark.parametrize("name", UNARY_POSITIVE)
def test_unary_positive_matches_cpu(gpu, name):
    parity(gpu, lambda a: getattr(a, name)(), POS)


ELEMENTWISE = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "chain": lambda a, b: a.add(b).mul(0.5).sub(1).div(2),
    "maximum": lambda a, b: a.maximum(b),
    "minimum": lambda a, b: a.minimum(b),
    "gt": lambda a, b: a.gt(b),
    "lt": lambda a, b: a.lt(b),
    "ge": lambda a, b: a.ge(b),
    "le": lambda a, b: a.le(b),
    "eq": lambda a, b: a.eq(a.relu()),
    "gt_scalar": lambda a, b: a > 0.4,
    "rsub": lambda a, b: 2 - a,
    "rtruediv": lambda a, b: 2 / b,
    "scalar_math": lambda a, b: a * 0.5 + 1,
    "broadcast_row": lambda a, b: a + b[0],
    "leaky_relu": lambda a, b: a.leaky_relu(0.1),
    "clamp": lambda a, b: a.clamp(-0.5, 1.0),
    "clamp_min": lambda a, b: a.clamp(min=0.2),
    "masked_fill": lambda a, b: a.masked_fill(a > 0.6, -5.0),
    "where": lambda a, b: ft.where(a > 0, a, b),
    "neg": lambda a, b: -a,
    "abs": lambda a, b: abs(b),
}


@pytest.mark.parametrize("name", list(ELEMENTWISE))
def test_elementwise_matches_cpu(gpu, name):
    parity(gpu, ELEMENTWISE[name], X, Y)


def test_pow_matches_cpu(gpu):
    parity(gpu, lambda a: a**2.5, POS)
    parity(gpu, lambda a: a**-1, POS)
    parity(gpu, lambda a: a.pow(3), X)


def test_broadcast_with_a_vector_matches_cpu(gpu):
    parity(gpu, lambda a, v: a + v, X, V)
    parity(gpu, lambda a, v: a * v - v, X, V)
    parity(gpu, lambda a, v: a.maximum(v), X, V)


MATMUL = {
    "2d": lambda a, m, v: a @ m,
    "vec_mat": lambda a, m, v: v @ m,
    "mat_vec": lambda a, m, v: m @ v,
    "dot": lambda a, m, v: v @ v,
    "method": lambda a, m, v: a.matmul(m),
    "batched_by_2d": lambda a, m, v: ft.stack([a, a * 2]) @ m,
    "batched": lambda a, m, v: ft.stack([a, a * 2]) @ ft.stack([m, m.T]),
}


@pytest.mark.parametrize("name", list(MATMUL))
def test_matmul_matches_cpu(gpu, name):
    parity(gpu, MATMUL[name], X, M, V)


REDUCTIONS = {
    "sum": lambda a: a.sum(),
    "sum_dim0": lambda a: a.sum(0),
    "sum_dim1_keepdim": lambda a: a.sum(1, keepdim=True),
    "sum_both_dims": lambda a: a.sum((0, 1)),
    "mean": lambda a: a.mean(),
    "mean_dim1": lambda a: a.mean(1),
    "mean_dim0_keepdim": lambda a: a.mean(0, keepdim=True),
    "max": lambda a: a.max(),
    "max_dim0": lambda a: a.max(0),
    "max_dim1_keepdim": lambda a: a.max(1, keepdim=True),
    "min": lambda a: a.min(),
    "min_dim1": lambda a: a.min(1),
    "var": lambda a: a.var(),
    "var_dim1": lambda a: a.var(1),
    "argmax": lambda a: a.argmax(),
    "argmax_dim1": lambda a: a.argmax(1),
    "argmin_dim0": lambda a: a.argmin(0),
    "argmin_dim1_keepdim": lambda a: a.argmin(1, keepdim=True),
    "softmax": lambda a: a.softmax(-1),
    "softmax_dim0": lambda a: a.softmax(0),
    "log_softmax": lambda a: a.log_softmax(-1),
    "log_softmax_dim0": lambda a: a.log_softmax(0),
}


@pytest.mark.parametrize("name", list(REDUCTIONS))
def test_reductions_match_cpu(gpu, name):
    parity(gpu, REDUCTIONS[name], X)


SHAPE_OPS = {
    "reshape": lambda a, b: a.reshape(3, 2),
    "reshape_infer": lambda a, b: a.reshape(-1),
    "flatten": lambda a, b: a.flatten(),
    "T": lambda a, b: a.T,
    "transpose": lambda a, b: a.transpose(0, 1),
    "permute": lambda a, b: ft.stack([a, b]).permute(2, 0, 1),
    "unsqueeze_squeeze": lambda a, b: a.unsqueeze(1).squeeze(1),
    "expand": lambda a, b: a.narrow(0, 0, 1).expand(4, 3),
    "narrow": lambda a, b: a.narrow(1, 1, 2),
    "chunk": lambda a, b: a.chunk(2, dim=1),
    "split": lambda a, b: a.split(2, dim=1),
    "index_select": lambda a, b: a.index_select(1, [2, 0, 0]),
    "index_select_tensor": lambda a, b: a.index_select(0, ft.tensor([1, 0], device=a.device)),
    "cat": lambda a, b: ft.cat([a, b]),
    "cat_dim1": lambda a, b: ft.cat([a, b], dim=1),
    "stack_last": lambda a, b: ft.stack([a, b], dim=-1),
    "slice_step": lambda a, b: a[1:, ::2],
    "slice_reverse": lambda a, b: a[:, ::-1],
    "int_index": lambda a, b: a[0],
    "int_int_index": lambda a, b: a[-1, 1],
}


@pytest.mark.parametrize("name", list(SHAPE_OPS))
def test_shape_ops_match_cpu(gpu, name):
    parity(gpu, SHAPE_OPS[name], X, Y)


FUNCTIONAL = {
    "mse": lambda a, b: F.mse_loss(a, b),
    "l1": lambda a, b: F.l1_loss(a, b),
    "smooth_l1": lambda a, b: F.smooth_l1_loss(a, b, beta=0.5),
    "bce": lambda a, b: F.binary_cross_entropy(a.sigmoid(), (b > 0) * 1.0),
    "bce_with_logits": lambda a, b: F.binary_cross_entropy_with_logits(a, (b > 0) * 1.0),
    "cross_entropy": lambda a, b: F.cross_entropy(a, [2, 0]),
    "cross_entropy_ignore": lambda a, b: F.cross_entropy(a, [2, -100]),
    "cross_entropy_probs": lambda a, b: F.cross_entropy(a, b.softmax(-1)),
    "nll": lambda a, b: F.nll_loss(a.log_softmax(-1), [1, 2]),
    "cosine": lambda a, b: F.cosine_similarity(a, b),
    "cosine_dim0": lambda a, b: F.cosine_similarity(a, b, dim=0),
    "dropout_eval": lambda a, b: F.dropout(a, 0.5, training=False),
    "activations": lambda a, b: F.relu(a) + F.leaky_relu(b, 0.2) + F.sigmoid(a) + F.tanh(b),
    "activations2": lambda a, b: F.softmax(a, dim=0) + F.log_softmax(b, dim=1) + F.gelu(a) + F.silu(b),
}


@pytest.mark.parametrize("name", list(FUNCTIONAL))
def test_functional_matches_cpu(gpu, name):
    parity(gpu, FUNCTIONAL[name], X, Y)


def test_functional_layers_match_cpu(gpu):
    parity(gpu, lambda a: F.layer_norm(a), X)
    parity(gpu, lambda a, w, b: F.layer_norm(a, w, b, eps=1e-3), X, V, V[::-1])
    parity(gpu, lambda a: F.rms_norm(a), X)
    parity(gpu, lambda a, w: F.rms_norm(a, w), X, V)
    parity(gpu, lambda a, m: F.linear(a, m), X, M)
    parity(gpu, lambda a, m, b: F.linear(a, m, b), X, M, V)
    parity(gpu, lambda a, m: F.linear(ft.stack([a, a * 2]), m), X, M)
    parity(gpu, lambda a, b: F.scaled_dot_product_attention(a, b, a), X, Y)
    parity(gpu, lambda a, b: F.scaled_dot_product_attention(a, b, b, is_causal=True), X, Y)

    def batched(a, b):
        return F.scaled_dot_product_attention(ft.stack([a, b]), ft.stack([b, a]), ft.stack([a, a]), True)

    parity(gpu, batched, X, Y)


def test_one_hot_and_accuracy_match_cpu(gpu):
    on_gpu = F.one_hot([0, 2, 1], 3, device=gpu)
    assert on_gpu.device == gpu
    assert on_gpu.tolist() == F.one_hot([0, 2, 1], 3).tolist()
    logits = ft.tensor([[0.2, 0.7, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    for target in ([1, 0, 2], [1, 1, 2], [1, -100, 2]):
        assert F.accuracy(logits.to(gpu), target) == F.accuracy(logits, target)
    assert F.accuracy(logits.to(gpu), ft.tensor([1, 0, 2], device=gpu)) == 1.0


GRADIENTS = {
    "matmul": (lambda a, m: a @ m, (X, M)),
    "relu_sigmoid": (lambda a: a.relu() * a.sigmoid(), (X,)),
    "gelu_silu": (lambda a: a.gelu() + a.silu(), (X,)),
    "tanh_exp": (lambda a: a.tanh() * a.exp(), (X,)),
    "abs_sin_cos": (lambda a: a.abs() + a.sin() + a.cos(), (X,)),
    "log_sqrt_rsqrt_reciprocal": (lambda a: a.log() + a.sqrt() + a.rsqrt() + a.reciprocal(), (POS,)),
    "pow": (lambda a: a**2.5, (POS,)),
    "div": (lambda a, b: a / b, (X, Y)),
    "maximum_minimum": (lambda a, b: a.maximum(b) + a.minimum(b) * 2, (X, Y)),
    "softmax": (lambda a: a.softmax(-1), (X,)),
    "log_softmax": (lambda a: a.log_softmax(0), (X,)),
    "sum": (lambda a: a.sum(1), (X,)),
    "mean": (lambda a: a.mean(0), (X,)),
    "max": (lambda a: a.max(1), (X,)),
    "min": (lambda a: a.min(0), (X,)),
    "var": (lambda a: a.var(1), (X,)),
    "index_select": (lambda a: a.index_select(1, [2, 0, 0]), (X,)),
    "narrow": (lambda a: a.narrow(1, 1, 2), (X,)),
    "slice": (lambda a: a[1:, ::2], (X,)),
    "transpose_reshape": (lambda a: a.T.reshape(-1), (X,)),
    "expand": (lambda a: a.narrow(0, 0, 1).expand(3, 3), (X,)),
    "cat": (lambda a, b: ft.cat([a, b * 2], dim=1), (X, Y)),
    "stack": (lambda a, b: ft.stack([a, b * 2]), (X, Y)),
    "where": (lambda a, b: ft.where(a > 0, a, b), (X, Y)),
    "mse": (lambda a, b: F.mse_loss(a, b), (X, Y)),
    "smooth_l1": (lambda a, b: F.smooth_l1_loss(a, b), (X, Y)),
    "cross_entropy": (lambda a: F.cross_entropy(a, [2, 0]), (X,)),
    "bce_with_logits": (lambda a, b: F.binary_cross_entropy_with_logits(a, (b > 0) * 1.0), (X, Y)),
    "layer_norm": (lambda a, w, b: F.layer_norm(a, w, b), (X, V, V)),
    "rms_norm": (lambda a, w: F.rms_norm(a, w), (X, V)),
    "linear": (lambda a, m, b: F.linear(a, m, b), (X, M, V)),
    "attention": (lambda a, b: F.scaled_dot_product_attention(a, b, a, is_causal=True), (X, Y)),
    "cosine": (lambda a, b: F.cosine_similarity(a, b), (X, Y)),
}


@pytest.mark.parametrize("name", list(GRADIENTS))
def test_gradients_match_cpu(gpu, name):
    fn, inputs = GRADIENTS[name]
    grad_parity(gpu, fn, *inputs)


def test_clip_grad_matches_cpu(gpu):
    def run(device):
        w = ft.tensor([[3.0, 4.0], [1.0, -2.0]], device=device, requires_grad=True)
        b = ft.tensor([0.5, -1.5], device=device, requires_grad=True)
        loss = (w * w).sum() + (b * 3).sum()
        loss.backward()
        norm = optim.clip_grad_norm([w, b], 2.0)
        clipped = (w.grad.tolist(), b.grad.tolist())
        optim.clip_grad_value([w, b], 0.5)
        return norm, clipped, (w.grad.tolist(), b.grad.tolist())

    norm_g, clipped_g, value_g = run(gpu)
    norm_c, clipped_c, value_c = run(None)
    assert norm_g == pytest.approx(norm_c, rel=1e-4)
    close(clipped_g, clipped_c, tol=1e-4)
    close(value_g, value_c, tol=1e-4)


def test_creation_on_gpu(gpu):
    cases = [
        (ft.zeros(2, 3, device=gpu), ft.zeros(2, 3)),
        (ft.ones((2, 2), device=gpu), ft.ones((2, 2))),
        (ft.full((3,), 2.5, device=gpu), ft.full((3,), 2.5)),
        (ft.arange(1, 2, 0.25, device=gpu), ft.arange(1, 2, 0.25)),
        (ft.linspace(0, 1, 5, device=gpu), ft.linspace(0, 1, 5)),
        (ft.eye(3, device=gpu), ft.eye(3)),
        (ft.tensor(X, device="gpu"), ft.tensor(X)),
    ]
    for on_gpu, on_cpu in cases:
        assert on_gpu.device == gpu
        assert on_gpu.tolist() == on_cpu.tolist()
    base = ft.ones(2, 2, device=gpu)
    for like in (ft.zeros_like(base), ft.ones_like(base), ft.full_like(base, 3.0)):
        assert like.device == gpu
    assert ft.full_like(base, 3.0).tolist() == [[3.0, 3.0], [3.0, 3.0]]
    assert ft.zeros(2, device=gpu, requires_grad=True).requires_grad


def test_random_on_gpu(gpu):
    u = ft.rand(500, device=gpu)
    n = ft.randn(2000, device=gpu)
    assert u.device == gpu and n.device == gpu
    assert 0.0 <= u.min().item() and u.max().item() < 1.0
    assert abs(n.mean().item()) < 0.15 and 0.8 < n.var().item() < 1.2
    ft.manual_seed(7)
    first = ft.randn(6, device=gpu).tolist()
    ft.manual_seed(7)
    assert ft.randn(6, device=gpu).tolist() == first


def test_to_and_cpu_round_trip(gpu):
    c = ft.tensor(X)
    g = c.to(gpu)
    assert g.device == gpu and c.device.is_cpu
    assert g.to(gpu).device == gpu and g.to("gpu").device == gpu
    back = g.cpu()
    assert back.device.is_cpu and back.tolist() == c.tolist()
    assert g.to("cpu").tolist() == c.tolist()
    assert "device='gpu'" in repr(g)
    # Building a tensor from a GPU tensor copies the values to the default device.
    assert ft.tensor(g).device.is_cpu


def test_to_is_differentiable_across_devices(gpu):
    x = ft.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.to(gpu)
    assert y.requires_grad
    (y * y).sum().backward()
    assert x.grad.device.is_cpu
    assert x.grad.tolist() == [2.0, 4.0, 6.0]


def test_device_lookup_and_identity(gpu):
    assert ft.Device.gpu_available()
    assert ft.Device("GPU") == gpu and hash(ft.Device("gpu")) == hash(gpu)
    assert ft.Device.best() == gpu and ft.Device("best") == gpu
    assert gpu != ft.Device("cpu") and ft.Device("cpu") != gpu
    assert gpu.is_gpu and not gpu.is_cpu
    assert repr(gpu).startswith("Device('gpu', name=")
    assert ft.zeros(1, device="best").device == gpu


def test_layers_match_cpu(gpu):
    ids = [[1, 5, 2, 0], [3, 3, 4, 6]]
    x_data = ft.randn(2, 5, 8).tolist()
    layers = [
        ("linear", lambda d: nn.Linear(8, 3, device=d), x_data),
        ("linear_no_bias", lambda d: nn.Linear(8, 3, bias=False, device=d), x_data),
        ("layer_norm", lambda d: nn.LayerNorm(8, device=d), x_data),
        ("layer_norm_2d", lambda d: nn.LayerNorm((5, 8), device=d), x_data),
        ("rms_norm", lambda d: nn.RMSNorm(8, device=d), x_data),
        ("attention", lambda d: nn.MultiHeadAttention(8, 2, device=d), x_data),
        ("attention_causal", lambda d: nn.MultiHeadAttention(8, 4, causal=True, device=d), x_data),
        ("embedding", lambda d: nn.Embedding(7, 8, device=d), ids),
    ]
    for name, make, data in layers:
        cpu_layer, gpu_layer = make(None), make(gpu)
        for p in cpu_layer.parameters():
            p.copy_(ft.randn(*p.shape))
        gpu_layer.load_state_dict(cpu_layer.state_dict())
        # Embedding takes the ids as a plain list on both devices.
        x_cpu = data if name == "embedding" else ft.tensor(data)
        x_gpu = data if name == "embedding" else ft.tensor(data, device=gpu)
        out_cpu, out_gpu = cpu_layer(x_cpu), gpu_layer(x_gpu)
        assert out_gpu.device == gpu, name
        close(out_gpu.tolist(), out_cpu.tolist(), tol=1e-3)
        (out_cpu * out_cpu).sum().backward()
        (out_gpu * out_gpu).sum().backward()
        pairs = zip(cpu_layer.named_parameters(), gpu_layer.named_parameters(), strict=True)
        for (pname, pc), (_, pg) in pairs:
            assert pg.grad is not None and pg.grad.device == gpu, (name, pname)
            close(pg.grad.tolist(), pc.grad.tolist(), tol=1e-3)


def test_dropout_on_gpu(gpu):
    x = ft.ones(1000, device=gpu)
    drop = nn.Dropout(0.5)
    y = drop(x)
    assert y.device == gpu
    assert set(y.tolist()) <= {0.0, 2.0}
    assert 350 < y.tolist().count(0.0) < 650
    drop.eval()
    assert drop(x).tolist() == [1.0] * 1000


def test_embedding_ids_may_be_a_gpu_tensor(gpu):
    emb = nn.Embedding(5, 3, device=gpu)
    out = emb(ft.tensor([[0, 4], [2, 2]], device=gpu))
    assert out.shape == (2, 2, 3) and out.device == gpu
    assert out.tolist()[1][0] == emb.weight[2].tolist()


def make_pair(gpu):
    cpu_model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 2))
    gpu_model = nn.Sequential(nn.Linear(4, 6, device=gpu), nn.Tanh(), nn.Linear(6, 2, device=gpu))
    gpu_model.load_state_dict(cpu_model.state_dict())
    return cpu_model, gpu_model


OPTIMIZERS = {
    "sgd": lambda p: optim.SGD(p, lr=0.1),
    "sgd_nesterov_decay": lambda p: optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
    "adamw": lambda p: optim.AdamW(p, lr=0.01),
    "adamw_clip": lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.1, max_grad_norm=0.5),
}


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_optimizer_steps_match_cpu(gpu, name):
    xs = [[((i * 7 + j * 3) % 10) / 5 - 1 for j in range(4)] for i in range(8)]
    ys = [i % 2 for i in range(8)]
    cpu_model, gpu_model = make_pair(gpu)
    for model, x in ((cpu_model, ft.tensor(xs)), (gpu_model, ft.tensor(xs, device=gpu))):
        opt = OPTIMIZERS[name](model.parameters())
        for _ in range(5):
            opt.zero_grad()
            F.cross_entropy(model(x), ys).backward()
            opt.step()
    for (_, pc), (_, pg) in zip(cpu_model.named_parameters(), gpu_model.named_parameters(), strict=True):
        assert pg.device == gpu
        close(pg.tolist(), pc.tolist(), tol=1e-3)


def test_lr_schedule_drives_a_gpu_optimizer(gpu):
    w = ft.ones(2, device=gpu, requires_grad=True)
    opt = optim.SGD([w], lr=1.0)
    sched = optim.StepLR(opt, step_size=1, gamma=0.5)
    for expected in (0.5, 0.25):
        opt.zero_grad()
        (w * 2).sum().backward()
        opt.step()
        assert sched.step() == pytest.approx(expected)
        assert opt.lr == pytest.approx(expected)
    # w = 1 - 1.0 * 2 - 0.5 * 2
    assert w.tolist() == pytest.approx([-2.0, -2.0])


def test_adamw_step_on_gpu(gpu):
    w = ft.ones(2, device=gpu, requires_grad=True)
    opt = optim.AdamW([w], lr=0.1, weight_decay=0.0)
    (w * w).sum().backward()
    opt.step()
    assert w.device == gpu and w.grad.device == gpu
    assert opt.steps == 1
    assert w.tolist() == pytest.approx([0.9, 0.9], abs=1e-4)


def test_gradients_on_gpu(gpu):
    w = ft.ones(3, device=gpu, requires_grad=True)
    (w * 3).sum().backward()
    (w * 3).sum().backward()
    assert w.grad.device == gpu and w.grad.tolist() == [6.0, 6.0, 6.0]
    w.grad = ft.zeros(3, device=gpu)
    assert w.grad.tolist() == [0.0, 0.0, 0.0]
    w.grad = None
    assert w.grad is None
    y = w * 2
    y.backward(ft.tensor([1.0, 2.0, 3.0], device=gpu))
    assert w.grad.tolist() == [2.0, 4.0, 6.0]
    with ft.no_grad():
        assert not (w * 2).requires_grad


def test_detach_clone_copy_on_gpu(gpu):
    a = ft.tensor([1.0, 2.0], device=gpu)
    d, c = a.detach(), a.clone()
    a.copy_(ft.tensor([5.0, 6.0], device=gpu))
    assert d.tolist() == [5.0, 6.0] and c.tolist() == [1.0, 2.0]
    assert d.device == gpu and c.device == gpu
    # Copying from a CPU tensor keeps the destination on the GPU.
    a.copy_(ft.tensor([7.0, 8.0]))
    assert a.device == gpu and a.tolist() == [7.0, 8.0]


def test_save_and_load_tensors_on_gpu(gpu, tmp_path):
    path = tmp_path / "t.safetensors"
    ft.save({"a": ft.tensor(X, device=gpu), "b": ft.arange(4, device=gpu)}, path)
    on_cpu = ft.load(path)
    on_gpu = ft.load(path, device=gpu)
    assert on_cpu["a"].device.is_cpu and on_gpu["a"].device == gpu
    assert on_cpu["a"].tolist() == X and on_gpu["b"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert ft.load(path, device="gpu")["a"].device == gpu


def test_save_load_whole_model_on_gpu(gpu, tmp_path):
    def make(device):
        return nn.Sequential(
            nn.Linear(4, 8, device=device), nn.LayerNorm(8, device=device), nn.Linear(8, 2, device=device)
        )

    src, dst = make(gpu), make(gpu)
    x = ft.randn(3, 4, device=gpu)
    src.save(tmp_path / "m.safetensors")
    dst.load(tmp_path / "m.safetensors")
    assert dst(x).tolist() == src(x).tolist()
    assert all(p.device == gpu for p in dst.parameters())


def test_mixed_devices_raise_value_error(gpu):
    c, g = ft.ones(2, 2), ft.ones(2, 2, device=gpu)
    g_grad = ft.ones(2, 2, device=gpu, requires_grad=True)
    calls = {
        "add": lambda: c + g,
        "sub": lambda: g - c,
        "mul": lambda: c * g,
        "div": lambda: g / c,
        "matmul": lambda: c @ g,
        "maximum": lambda: g.maximum(c),
        "gt": lambda: g.gt(c),
        "cat": lambda: ft.cat([c, g]),
        "stack": lambda: ft.stack([g, c]),
        "where_condition": lambda: ft.where(g, c, c),
        "where_branch": lambda: ft.where(c, g, 0.0),
        "masked_fill": lambda: g.masked_fill(c, 1.0),
        "mse_loss": lambda: F.mse_loss(g, c),
        "l1_loss": lambda: F.l1_loss(c, g),
        "cross_entropy": lambda: F.cross_entropy(g, ft.tensor([0, 1])),
        "bce_with_logits": lambda: F.binary_cross_entropy_with_logits(g, c),
        "linear_weight": lambda: F.linear(g, c),
        "linear_bias": lambda: F.linear(g, g, c.sum(0)),
        "layer_norm": lambda: F.layer_norm(g, c.sum(0)),
        "rms_norm": lambda: F.rms_norm(c, g.sum(0)),
        "attention": lambda: F.scaled_dot_product_attention(g, c, c),
        "cosine_similarity": lambda: F.cosine_similarity(g, c),
        "accuracy": lambda: F.accuracy(g, ft.tensor([0, 1])),
        "backward_gradient": lambda: g_grad.backward(c),
        "grad_setter": lambda: setattr(g, "grad", c),
    }
    for call in calls.values():
        with pytest.raises(ValueError, match="same device"):
            call()


def test_layer_input_on_the_wrong_device_raises_value_error(gpu):
    with pytest.raises(ValueError, match="same device"):
        nn.Linear(3, 2)(ft.ones(4, 3, device=gpu))
    with pytest.raises(ValueError, match="same device"):
        nn.LayerNorm(3, device=gpu)(ft.ones(4, 3))
    with pytest.raises(ValueError, match="same device"):
        nn.MultiHeadAttention(4, 2, device=gpu)(ft.ones(3, 4))
