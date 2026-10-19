"""tools/dashboard/menos_dashboard.py: the benchmark dashboard's runner and
renderer. A tiny sweep checks that MENOS and Qiskit agree on the dashboard's
circuit family, and rendering checks that the page embeds the data, stays
self-contained, and leaves unmeasured values unfilled."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dashboard" / "menos_dashboard.py"


@pytest.fixture(scope="module")
def dashboard():
    spec = importlib.util.spec_from_file_location("menos_dashboard", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _crossover_fixture() -> dict:
    # The shape cuda_crossover_benchmark.cpp writes, with one unmeasured CPU
    # row and one row that does not fit in device memory.
    return {
        "schema_version": 1,
        "device_present": True,
        "gpu": {
            "name": "Test GPU </script>",
            "compute_capability": "8.6",
            "multiprocessors": 16,
            "total_memory_bytes": 4 << 30,
            "free_memory_bytes": 3 << 30,
        },
        "cuda_toolkit": "12.9",
        "circuit": {"layers": 6, "description": "H on all qubits, T on q0, CX"},
        "timing": "end-to-end",
        "statevector": [
            {"n": 10, "host_bytes": 16384, "dev_bytes": 16384, "work_units": 1,
             "cpu_seconds": 0.001, "cuda_seconds": 0.002, "vram_ok": True},
            {"n": 26, "host_bytes": 1 << 30, "dev_bytes": 1 << 30, "work_units": 1,
             "cpu_seconds": None, "cuda_seconds": 0.5, "vram_ok": True},
            {"n": 28, "host_bytes": 1 << 32, "dev_bytes": 1 << 32, "work_units": 1,
             "cpu_seconds": None, "cuda_seconds": None, "vram_ok": False},
        ],
        "expectation": [],
    }


def test_circuit_matches_the_crossover_benchmark(dashboard):
    qc = dashboard.build_circuit(4)
    # 6 layers of (4 H + 1 T + CX on alternating pairs: 2 on even layers, 1 on odd).
    assert len(qc.data) == 6 * 5 + 3 * 2 + 3 * 1
    assert dashboard.LAYERS == 6


def test_other_families_have_the_documented_shape(dashboard):
    # clifford: 6 layers of (4 H + 4 S) + the same CX pattern (9 in all).
    assert len(dashboard.build_clifford(4).data) == 6 * 8 + 3 * 2 + 3 * 1
    # product: 6 layers of (4 H + 4 RZ), and no two-qubit gate at all.
    product = dashboard.build_product(4)
    assert len(product.data) == 6 * 8
    assert all(inst.operation.num_qubits == 1 for inst in product.data)
    assert set(dashboard.FAMILIES) == {"baseline", "clifford", "product"}


@pytest.mark.parametrize("family", ["clifford", "product"])
def test_sweep_of_other_families_agrees_with_qiskit(dashboard, family):
    document = dashboard.sweep([2, 4], repeats=1, family=family)
    assert document["circuit"]["family"] == family
    assert document["circuit"]["description"] == dashboard.FAMILIES[family][1]
    for point in document["menos_vs_qiskit"]:
        assert point["agrees"], point


def test_baseline_is_the_default_family(dashboard):
    assert dashboard.sweep([2], repeats=1)["circuit"]["family"] == "baseline"


def test_cli_refuses_a_crossover_json_with_another_family(dashboard, tmp_path):
    crossover = tmp_path / "crossover.json"
    crossover.write_text(json.dumps(_crossover_fixture()))
    with pytest.raises(SystemExit):
        dashboard.main(["run", "--family", "clifford", "--crossover-json", str(crossover),
                        "--out", str(tmp_path / "results.json")])
    with pytest.raises(SystemExit):
        dashboard.main(["run", "--family", "nope", "--out", str(tmp_path / "results.json")])


def test_sweep_agrees_with_qiskit(dashboard):
    document = dashboard.sweep([2, 4, 6], repeats=3)
    assert document["schema_version"] == dashboard.SCHEMA_VERSION
    assert document["repeats"] == 3
    assert document["crossover"] is None
    # Power is read both before and after the timed runs.
    assert set(document["environment"]["power_after"]) == set(document["environment"]["power"])
    points = document["menos_vs_qiskit"]
    assert [p["n"] for p in points] == [2, 4, 6]
    for point in points:
        assert point["agrees"], point
        assert point["max_abs_deviation"] <= dashboard.ABS_TOL
        assert point["menos_backend"] in ("cpu", "cuda")
        assert point["menos_wall_seconds"] > 0
        assert point["qiskit_wall_seconds"] > 0
        for series in ("menos_wall", "menos_execute", "qiskit_wall"):
            samples = point[f"{series}_samples"]
            s = point[f"{series}_spread"]
            assert len(samples) == 3
            assert s["min"] == min(samples) and s["max"] == max(samples)
            assert s["min"] <= s["q1"] <= s["median"] <= s["q3"] <= s["max"]
            # The legacy key is the median, so older readers stay correct.
            assert point[f"{series}_seconds"] == s["median"]
    json.dumps(document, allow_nan=False)


def test_spread_statistics(dashboard):
    assert dashboard.spread([0.5]) == {
        "median": 0.5, "min": 0.5, "max": 0.5, "q1": 0.5, "q3": 0.5,
    }
    s = dashboard.spread([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s == {"median": 3.0, "min": 1.0, "max": 5.0, "q1": 2.0, "q3": 4.0}


def test_environment_records_power_and_cores_without_guessing(dashboard):
    env = dashboard.environment()
    assert isinstance(env["logical_cpus"], int) and env["logical_cpus"] >= 1
    assert env["physical_cpus"] is None or 1 <= env["physical_cpus"] <= env["logical_cpus"]
    assert isinstance(env["wsl"], bool)
    assert env["cpu_governor"] is None or isinstance(env["cpu_governor"], str)
    power = env["power"]
    assert set(power) == {"ac_online", "battery_percent", "power_scheme"}
    assert power["ac_online"] in (None, True, False)
    assert power["battery_percent"] is None or 0 <= power["battery_percent"] <= 100
    assert power["power_scheme"] is None or isinstance(power["power_scheme"], str)


def test_unreadable_power_state_is_unknown(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "_probe", lambda *command: None)
    assert dashboard._windows_power() == {
        "ac_online": None, "battery_percent": None, "power_scheme": None,
    }


def test_windows_power_output_is_parsed(dashboard, monkeypatch):
    replies = {
        "powershell.exe": "Online 0.87",
        "powercfg.exe": "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)",
    }
    monkeypatch.setattr(dashboard, "_probe", lambda *command: replies[command[0]])
    assert dashboard._windows_power() == {
        "ac_online": True, "battery_percent": 87, "power_scheme": "Balanced",
    }
    # 255 is Windows' "no battery / unknown", not a percentage.
    replies["powershell.exe"] = "Offline 255"
    assert dashboard._windows_power()["battery_percent"] is None
    assert dashboard._windows_power()["ac_online"] is False


def test_render_refuses_an_older_schema(dashboard):
    with pytest.raises(ValueError, match="schema_version"):
        dashboard.render({"schema_version": 1})


def test_crossover_json_is_validated(dashboard, tmp_path):
    good = tmp_path / "crossover.json"
    good.write_text(json.dumps(_crossover_fixture()))
    assert dashboard.load_crossover(good)["statevector"][1]["cpu_seconds"] is None

    wrong_version = dict(_crossover_fixture(), schema_version=2)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(wrong_version))
    with pytest.raises(ValueError, match="schema_version"):
        dashboard.load_crossover(bad)

    missing_field = _crossover_fixture()
    del missing_field["statevector"][0]["cuda_seconds"]
    bad.write_text(json.dumps(missing_field))
    with pytest.raises(ValueError, match="cuda_seconds"):
        dashboard.load_crossover(bad)


def test_render_embeds_data_and_is_self_contained(dashboard, tmp_path):
    crossover_path = tmp_path / "crossover.json"
    crossover_path.write_text(json.dumps(_crossover_fixture()))
    document = dashboard.sweep([2], repeats=1, crossover=dashboard.load_crossover(crossover_path))
    html = dashboard.render(document)

    assert dashboard.DATA_PLACEHOLDER not in html
    match = re.search(r"const DATA = (.*);\n", html)
    assert match is not None
    embedded = json.loads(match.group(1).replace("<\\/", "</"))
    assert embedded == json.loads(json.dumps(document))
    # Unmeasured values stay null; nothing is filled in.
    assert embedded["crossover"]["statevector"][2]["cuda_seconds"] is None
    # A string in the data cannot close the <script> element.
    assert "Test GPU </script>" not in html
    # No network requests: no external scripts, stylesheets or images.
    assert not re.search(r"""(src|href)\s*=\s*["']?https?:""", html)
    assert "@import" not in html


def test_cli_run_and_render(dashboard, tmp_path):
    results = tmp_path / "results.json"
    page = tmp_path / "dashboard.html"
    assert dashboard.main(["run", "--min-qubits", "3", "--max-qubits", "3",
                           "--repeats", "1", "--out", str(results)]) == 0
    assert dashboard.main(["render", "--results", str(results), "--out", str(page)]) == 0
    assert "MENOS benchmarks" in page.read_text(encoding="utf-8")
