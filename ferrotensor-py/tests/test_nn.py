import math

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F
from conftest import close
from ferrotensor import nn


def test_linear():
    layer = nn.Linear(3, 2)
    assert layer.weight.shape == (3, 2) and layer.bias.shape == (2,)
    assert layer(ft.ones(4, 3)).shape == (4, 2)
    assert layer(ft.ones(3)).shape == (2,)
    assert layer(ft.ones(5, 4, 3)).shape == (5, 4, 2)
    assert [n for n, _ in layer.named_parameters()] == ["weight", "bias"]
    assert layer.num_parameters() == 8
    assert nn.Linear(3, 2, bias=False).bias is None
    assert repr(layer) == "Linear(in_features=3, out_features=2, bias=True)"


def test_linear_matches_functional():
    layer = nn.Linear(3, 2)
    x = ft.randn(4, 3)
    expected = (x @ layer.weight + layer.bias).tolist()
    got = layer(x).tolist()
    for row_g, row_e in zip(got, expected, strict=True):
        assert row_g == pytest.approx(row_e, abs=1e-5)
    assert nn.functional.linear(x, layer.weight, layer.bias).shape == (4, 2)


def test_embedding():
    emb = nn.Embedding(10, 4)
    assert emb.weight.shape == (10, 4)
    assert emb([1, 2, 3]).shape == (3, 4)
    assert emb(ft.tensor([[0, 9], [1, 1]])).shape == (2, 2, 4)
    assert emb([2]).tolist()[0] == emb.weight[2].tolist()
    out = emb([1, 1]).sum()
    out.backward()
    assert emb.weight.grad[1].tolist() == [2.0] * 4
    assert emb.weight.grad[0].tolist() == [0.0] * 4


def test_norms():
    ln = nn.LayerNorm(4)
    y = ln(ft.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert abs(y.mean().item()) < 1e-5
    assert y.var().item() == pytest.approx(1.0, abs=1e-3)
    assert [n for n, _ in ln.named_parameters()] == ["weight", "bias"]
    assert nn.LayerNorm((2, 3))(ft.ones(5, 2, 3)).shape == (5, 2, 3)
    rms = nn.RMSNorm(2)
    assert rms(ft.tensor([[3.0, 4.0]])).tolist()[0] == pytest.approx(
        [3 / (12.5**0.5), 4 / (12.5**0.5)], abs=1e-4
    )
    assert rms.weight.shape == (2,)


def test_dropout_train_eval():
    drop = nn.Dropout(0.5)
    x = ft.ones(1000)
    y = drop(x)
    zeros = sum(1 for v in y.tolist() if v == 0.0)
    assert 350 < zeros < 650
    assert set(y.tolist()) <= {0.0, 2.0}
    drop.eval()
    assert not drop.training
    assert drop(x).tolist() == x.tolist()
    drop.train()
    assert drop.training


def test_activations():
    x = ft.tensor([-1.0, 0.0, 2.0])
    assert nn.ReLU()(x).tolist() == [0.0, 0.0, 2.0]
    assert nn.LeakyReLU(0.5)(x).tolist() == [-0.5, 0.0, 2.0]
    assert nn.GELU()(x).shape == (3,)
    assert nn.SiLU()(x).tolist()[1] == 0.0
    assert nn.Sigmoid()(x).tolist()[1] == 0.5
    assert nn.Tanh()(x).tolist()[1] == 0.0
    assert sum(nn.Softmax()(x).tolist()) == pytest.approx(1.0)
    assert nn.Identity()(x) is x
    assert nn.Flatten()(ft.ones(2, 3, 4)).shape == (2, 12)


def test_attention():
    mha = nn.MultiHeadAttention(8, 2, causal=True)
    assert mha(ft.randn(2, 5, 8)).shape == (2, 5, 8)
    assert mha(ft.randn(5, 8)).shape == (5, 8)
    names = [n for n, _ in mha.named_parameters()]
    assert names == ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]


def test_sequential_and_containers():
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.1), nn.Linear(8, 1))
    assert len(model) == 4 and isinstance(model[0], nn.Linear)
    assert isinstance(model[:2], nn.Sequential) and len(model[:2]) == 2
    assert model(ft.ones(3, 2)).shape == (3, 1)
    assert [n for n, _ in model.named_parameters()] == ["0.weight", "0.bias", "3.weight", "3.bias"]
    model.eval()
    assert not model[2].training
    model.append(nn.Sigmoid())
    assert len(list(model)) == 5
    ml = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
    assert len(ml.parameters()) == 4
    with pytest.raises(TypeError):
        ml.append("not a module")


class Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(ft.ones(1))
        self.frozen = ft.ones(1)  # no requires_grad: not a parameter
        self.body = nn.Sequential(nn.Linear(2, 2))
        self.heads = [nn.Linear(2, 1), nn.Linear(2, 1)]
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        return self.heads[0](self.drop(self.body(x))) * self.scale


def test_custom_module_discovery():
    m = Custom()
    names = [n for n, _ in m.named_parameters()]
    assert names == [
        "scale",
        "body.0.weight",
        "body.0.bias",
        "heads.0.weight",
        "heads.0.bias",
        "heads.1.weight",
        "heads.1.bias",
    ]
    assert m(ft.ones(3, 2)).shape == (3, 1)
    m.eval()
    assert not m.drop.training and not m.body.training
    assert len(list(m.modules())) == 6
    m(ft.ones(3, 2)).sum().backward()
    assert m.scale.grad is not None
    m.zero_grad()
    assert all(p.grad is None for p in m.parameters())
    assert "(body): Sequential(" in repr(m)


def test_module_without_forward():
    with pytest.raises(NotImplementedError):
        nn.Module()(ft.ones(1))


def test_linear_with_known_weights():
    layer = nn.Linear(3, 2)
    layer.weight.copy_(ft.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    layer.bias.copy_(ft.tensor([1.0, -1.0]))
    x = ft.tensor([[1.0, 0.0, 1.0]])
    assert layer(x).tolist() == [[7.0, 7.0]]
    assert layer(ft.tensor([1.0, 1.0, 1.0])).tolist() == [10.0, 11.0]
    layer(x).sum().backward()
    assert layer.weight.grad.tolist() == [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
    assert layer.bias.grad.tolist() == [1.0, 1.0]
    assert nn.Linear(3, 2, bias=False)(ft.ones(1, 3)).shape == (1, 2)


def test_embedding_with_known_weights():
    emb = nn.Embedding(4, 2)
    emb.weight.copy_(ft.arange(8).reshape(4, 2))
    assert emb([3, 0]).tolist() == [[6.0, 7.0], [0.0, 1.0]]
    assert emb(ft.tensor([[1, 1]])).tolist() == [[[2.0, 3.0], [2.0, 3.0]]]
    assert emb([2]).shape == (1, 2)


def test_layer_norm_with_scale_and_shift():
    ln = nn.LayerNorm(3)
    ln.weight.copy_(ft.tensor([2.0, 2.0, 2.0]))
    ln.bias.copy_(ft.tensor([1.0, 1.0, 1.0]))
    unit = 1 / math.sqrt(2 / 3)
    out = ln(ft.tensor([[1.0, 2.0, 3.0]]))
    close(out.tolist(), [[1 - 2 * unit, 1.0, 1 + 2 * unit]], tol=1e-4)
    # Rows are normalized independently of each other.
    both = ln(ft.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])).tolist()
    close(both[0], both[1], tol=1e-3)


def test_rms_norm_scale():
    rms = nn.RMSNorm(2)
    rms.weight.copy_(ft.tensor([1.0, 3.0]))
    root = math.sqrt(12.5)
    close(rms(ft.tensor([[3.0, 4.0]])).tolist(), [[3 / root, 12 / root]], tol=1e-4)


def test_causal_attention_ignores_the_future():
    ft.manual_seed(3)
    x = ft.randn(4, 8)
    changed = ft.cat([x[:3], ft.randn(1, 8)])
    causal = nn.MultiHeadAttention(8, 2, causal=True)
    close(causal(x)[:3].tolist(), causal(changed)[:3].tolist(), tol=1e-5)
    full = nn.MultiHeadAttention(8, 2)
    assert (full(x)[:3] - full(changed)[:3]).abs().max().item() > 1e-4
    assert causal.num_parameters() == 4 * 8 * 8


def test_attention_batches_do_not_mix():
    ft.manual_seed(4)
    mha = nn.MultiHeadAttention(8, 2)
    a, b = ft.randn(5, 8), ft.randn(5, 8)
    together = mha(ft.stack([a, b]))
    close(together[0].tolist(), mha(a).tolist(), tol=1e-4)
    close(together[1].tolist(), mha(b).tolist(), tol=1e-4)


def test_dropout_probability_zero_is_the_identity():
    assert nn.Dropout(0.0)(ft.ones(5)).tolist() == [1.0] * 5


def test_parameter_helper():
    source = ft.tensor([1.0, 2.0])
    p = nn.Parameter(source)
    assert p.requires_grad and p.is_leaf and p.tolist() == [1.0, 2.0]
    assert not source.requires_grad


def test_train_and_eval_return_the_module():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    assert model.eval() is model and not model.training and not model[1].training
    assert model.train() is model and model.training and model[1].training
    assert model.train(False) is model and not model[1].training


def test_state_dict_shares_storage_with_the_module():
    layer = nn.Linear(2, 1)
    state = layer.state_dict()
    layer.weight.copy_(ft.tensor([[3.0], [4.0]]))
    assert state["weight"].tolist() == [[3.0], [4.0]]
    assert list(state) == ["weight", "bias"]
    assert not state["weight"].requires_grad


def test_load_state_dict_copies_values_not_references():
    src, dst = nn.Linear(2, 2), nn.Linear(2, 2)
    dst.load_state_dict(src.state_dict())
    src.weight.copy_(ft.zeros(2, 2))
    assert dst.weight.tolist() != [[0.0, 0.0], [0.0, 0.0]]


def test_module_containers():
    ml = nn.ModuleList([nn.Linear(1, 1), nn.ReLU()])
    assert len(ml) == 2 and isinstance(ml[1], nn.ReLU)
    assert isinstance(ml[:1], nn.ModuleList) and len(ml[:1]) == 1
    assert [type(m).__name__ for m in ml] == ["Linear", "ReLU"]
    seq = nn.Sequential()
    assert seq.append(nn.Identity()) is seq
    assert seq(ft.ones(2)).tolist() == [1.0, 1.0]
    assert nn.Sequential().parameters() == []


def test_reprs():
    assert repr(nn.Embedding(10, 4)) == "Embedding(10, 4)"
    assert repr(nn.Dropout(0.1)) == "Dropout(p=0.1)"
    assert repr(nn.ReLU()) == "ReLU()"
    assert repr(nn.LeakyReLU(0.5)) == "LeakyReLU(negative_slope=0.5)"
    assert repr(nn.Softmax(dim=1)) == "Softmax(dim=1)"
    assert repr(nn.RMSNorm(4)) == "RMSNorm(4, eps=1e-05)"
    assert repr(nn.LayerNorm(4)) == "LayerNorm((4,), eps=1e-05)"
    assert repr(nn.MultiHeadAttention(8, 2)) == "MultiHeadAttention(embed_dim=8, num_heads=2, causal=False)"
    assert repr(nn.Sequential(nn.ReLU())) == "Sequential(\n  (0): ReLU()\n)"


def test_activation_modules_match_functional():
    x = ft.tensor([-1.5, 0.5, 2.0])
    pairs = [
        (nn.ReLU(), F.relu),
        (nn.GELU(), F.gelu),
        (nn.SiLU(), F.silu),
        (nn.Sigmoid(), F.sigmoid),
        (nn.Tanh(), F.tanh),
        (nn.Softmax(), F.softmax),
        (nn.LeakyReLU(0.2), lambda t: F.leaky_relu(t, 0.2)),
    ]
    for module, fn in pairs:
        assert module(x).tolist() == fn(x).tolist(), module
    assert nn.Flatten(0)(ft.ones(2, 3)).shape == (6,)
    assert nn.Flatten(1, 2)(ft.ones(2, 3, 4, 5)).shape == (2, 12, 5)
