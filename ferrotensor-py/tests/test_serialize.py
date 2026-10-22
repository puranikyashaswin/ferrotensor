import pytest

import ferrotensor as ft
from ferrotensor import nn


def make_model():
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.LayerNorm(4), nn.Linear(4, 2))


def test_save_load_tensors(tmp_path):
    path = tmp_path / "t.safetensors"
    ft.save({"a": ft.arange(6).reshape(2, 3), "b": ft.tensor(1.5)}, path)
    loaded = ft.load(path)
    assert loaded["a"].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert loaded["b"].shape == () and loaded["b"].item() == 1.5
    assert ft.load(str(path), device="cpu")["a"].device.is_cpu


def test_state_dict_round_trip(tmp_path):
    src, dst = make_model(), make_model()
    x = ft.randn(5, 3)
    path = tmp_path / "model.safetensors"
    src.save(path)
    assert dst(x).tolist() != src(x).tolist()
    result = dst.load(path)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert dst(x).tolist() == src(x).tolist()
    assert set(src.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias", "3.weight", "3.bias"}


def test_load_state_dict_keeps_optimizer_working():
    model = nn.Linear(2, 1)
    opt = ft.optim.SGD(model.parameters(), lr=0.1)
    model.load_state_dict({"weight": ft.zeros(2, 1), "bias": ft.zeros(1)})
    assert model.weight.tolist() == [[0.0], [0.0]]
    model(ft.ones(1, 2)).sum().backward()
    opt.step()
    assert model.bias.tolist() == pytest.approx([-0.1])


def test_load_state_dict_errors():
    model = nn.Linear(2, 1)
    with pytest.raises(RuntimeError, match="missing keys"):
        model.load_state_dict({"weight": ft.zeros(2, 1)})
    with pytest.raises(RuntimeError, match="unexpected keys"):
        model.load_state_dict({"weight": ft.zeros(2, 1), "bias": ft.zeros(1), "x": ft.zeros(1)})
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict({"weight": ft.zeros(1, 2), "bias": ft.zeros(1)})
    result = model.load_state_dict({"bias": ft.ones(1)}, strict=False)
    assert result.missing_keys == ["weight"] and model.bias.tolist() == [1.0]


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ft.load(tmp_path / "missing.safetensors")
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="safetensors"):
        ft.load(bad)
