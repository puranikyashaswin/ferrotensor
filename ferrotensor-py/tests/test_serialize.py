import json
import struct

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


def write_safetensors(path, entries):
    """Write {name: (dtype, shape, raw bytes)} in the safetensors layout, byte by byte."""
    header, offset = {}, 0
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, raw in entries.values():
            f.write(raw)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(7, 8)
        self.attn = nn.MultiHeadAttention(8, 2, causal=True)
        self.norm = nn.LayerNorm(8)
        self.rms = nn.RMSNorm(8)
        self.out = nn.Linear(8, 3, bias=False)

    def forward(self, ids):
        return self.out(self.rms(self.norm(self.attn(self.emb(ids)))))


def test_round_trip_is_exact_for_many_shapes(tmp_path):
    ft.manual_seed(5)
    tensors = {
        "scalar": ft.tensor(-0.5),
        "vec": ft.randn(7),
        "mat": ft.randn(3, 4),
        "cube": ft.randn(2, 3, 4),
        "ints": ft.arange(5),
        "layer.0/weight": ft.rand(2, 2),
    }
    path = tmp_path / "many.safetensors"
    ft.save(tensors, path)
    loaded = ft.load(path)
    assert set(loaded) == set(tensors)
    for name, t in tensors.items():
        assert loaded[name].shape == t.shape, name
        assert loaded[name].tolist() == t.tolist(), name
        assert loaded[name].dtype == "float32" and not loaded[name].requires_grad


def test_save_writes_the_values_of_views_and_graph_results(tmp_path):
    base = ft.arange(6).reshape(2, 3)
    cube = ft.arange(24).reshape(2, 3, 4)
    w = ft.ones(2, requires_grad=True)
    tensors = {"t": base.T, "step": base[:, ::2], "perm": cube.permute(2, 0, 1), "graph": w * 2}
    path = tmp_path / "views.safetensors"
    ft.save(tensors, path)
    loaded = ft.load(path)
    assert loaded["t"].tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
    assert loaded["step"].tolist() == [[0.0, 2.0], [3.0, 5.0]]
    assert loaded["perm"].shape == (4, 2, 3) and loaded["perm"].tolist() == cube.permute(2, 0, 1).tolist()
    assert loaded["graph"].tolist() == [2.0, 2.0]


def test_saving_twice_keeps_the_latest_values(tmp_path):
    path = tmp_path / "twice.safetensors"
    ft.save({"a": ft.ones(2)}, path)
    ft.save({"a": ft.zeros(3), "b": ft.ones(1)}, str(path))
    loaded = ft.load(path)
    assert loaded["a"].tolist() == [0.0, 0.0, 0.0] and set(loaded) == {"a", "b"}


def test_whole_model_round_trip_gives_equal_outputs(tmp_path):
    ids = [[1, 2, 3], [4, 5, 6]]
    src, dst = Net(), Net()
    assert dst(ids).tolist() != src(ids).tolist()
    src.save(tmp_path / "net.safetensors")
    result = dst.load(tmp_path / "net.safetensors")
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert dst(ids).tolist() == src(ids).tolist()
    keys = set(src.state_dict())
    assert {"emb.weight", "attn.q_proj.weight", "norm.bias", "rms.weight", "out.weight"} <= keys
    assert "out.bias" not in keys


def test_state_dict_round_trip_through_save_and_load_functions(tmp_path):
    src, dst = Net(), Net()
    ft.save(src.state_dict(), tmp_path / "sd.safetensors")
    dst.load_state_dict(ft.load(tmp_path / "sd.safetensors"))
    for (name, a), (_, b) in zip(src.named_parameters(), dst.named_parameters(), strict=True):
        assert a.tolist() == b.tolist(), name


def test_partial_load_reports_the_difference(tmp_path):
    ft.save({"weight": ft.ones(3, 2), "extra": ft.zeros(1)}, tmp_path / "p.safetensors")
    layer = nn.Linear(3, 2)
    result = layer.load(tmp_path / "p.safetensors", strict=False)
    assert result.missing_keys == ["bias"] and result.unexpected_keys == ["extra"]
    assert layer.weight.tolist() == [[1.0, 1.0]] * 3
    with pytest.raises(RuntimeError, match="unexpected keys"):
        layer.load(tmp_path / "p.safetensors")


def test_load_converts_f64_f16_and_bf16_to_float32(tmp_path):
    values = [1.5, -2.0, 0.25]
    bf16 = b"".join(struct.pack("<f", v)[2:] for v in values)
    write_safetensors(
        tmp_path / "dtypes.safetensors",
        {
            "f32": ("F32", [3], struct.pack("<3f", *values)),
            "f64": ("F64", [3], struct.pack("<3d", *values)),
            "f16": ("F16", [3], struct.pack("<3e", *values)),
            "bf16": ("BF16", [3], bf16),
            "matrix": ("F64", [2, 2], struct.pack("<4d", 1.0, 2.0, 3.0, 4.0)),
        },
    )
    loaded = ft.load(tmp_path / "dtypes.safetensors")
    for name in ("f32", "f64", "f16", "bf16"):
        assert loaded[name].tolist() == values, name
        assert loaded[name].dtype == "float32"
    assert loaded["matrix"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_load_rejects_dtypes_it_cannot_convert(tmp_path):
    write_safetensors(tmp_path / "ints.safetensors", {"ids": ("I64", [2], struct.pack("<2q", 1, 2))})
    with pytest.raises(ValueError, match="unsupported dtype"):
        ft.load(tmp_path / "ints.safetensors")


def test_save_rejects_bad_arguments(tmp_path):
    path = tmp_path / "bad.safetensors"
    with pytest.raises(TypeError):
        ft.save({"a": [1.0, 2.0]}, path)
    with pytest.raises(TypeError):
        ft.save([ft.ones(1)], path)
    with pytest.raises(TypeError):
        ft.save({1: ft.ones(1)}, path)
    with pytest.raises(TypeError):
        ft.save({"a": ft.ones(1)}, 5)
    assert not path.exists()


def test_save_to_a_missing_directory_raises(tmp_path):
    # The safetensors writer reports the missing directory as an I/O or a format error.
    with pytest.raises((OSError, ValueError)):
        ft.save({"a": ft.ones(1)}, tmp_path / "no_such_dir" / "x.safetensors")


def test_load_bad_arguments_and_files(tmp_path):
    with pytest.raises(ValueError, match="unknown device"):
        ft.load(tmp_path / "missing.safetensors", device="tpu")
    with pytest.raises(TypeError):
        ft.load(123)
    with pytest.raises(TypeError):
        ft.load(None)
    with pytest.raises(OSError):
        ft.load(tmp_path)  # a directory, not a file
    good = tmp_path / "good.safetensors"
    ft.save({"a": ft.arange(8)}, good)
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(good.read_bytes()[:-8])
    with pytest.raises(ValueError, match="safetensors"):
        ft.load(truncated)
    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="safetensors"):
        ft.load(empty)


def test_loaded_tensors_are_independent_of_the_file_and_each_other(tmp_path):
    path = tmp_path / "t.safetensors"
    ft.save({"a": ft.ones(2)}, path)
    first, second = ft.load(path), ft.load(path)
    first["a"].copy_(ft.zeros(2))
    assert second["a"].tolist() == [1.0, 1.0] and ft.load(path)["a"].tolist() == [1.0, 1.0]
