import pytest

import ferrotensor as ft
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
