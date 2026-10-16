"""The simulated optical bench and the focusing tool, plus the executor calling that tool."""

import csv

import numpy as np
from langchain_core.messages import AIMessage

from agentic_ode.agents import Executor, finish_allowed
from agentic_ode.core.state import initial_state
from agentic_ode.hardware.devices.optical_bench.simulator import Bench, make_regions, transmission_matrix
from agentic_ode.runtime.cli import _auto_ask
from agentic_ode.tools.focus import CSV_NAME, run_focus_search


def test_two_shots_of_one_pattern_differ():
    bench = Bench(macropixel=8, seed=0)
    bench.set_pattern(np.arange(bench.n) % 2 == 0)
    a, b = bench.shot(), bench.shot()
    assert a.shape == (64, 64)
    assert not np.array_equal(a, b)  # shot noise + read noise
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9  # but the same speckle


def test_same_medium_in_every_run():
    t = transmission_matrix()
    assert t.shape == (4096, 4096)
    b1, b2 = Bench(macropixel=16, seed=1), Bench(macropixel=16, seed=2)
    pattern = np.ones(16, dtype=bool)
    b1.set_pattern(pattern)
    b2.set_pattern(pattern)
    assert np.allclose(b1.field, b2.field)  # different seeds change the noise, not the medium


def test_flip_matches_full_recompute():
    bench = Bench(macropixel=4, seed=0)
    bench.set_pattern(np.zeros(bench.n, dtype=bool))
    bench.flip(np.array([3, 7, 100]))
    full = bench.tm @ bench.pattern.astype(np.complex64)
    assert np.allclose(bench.field, full, atol=1e-3)


def test_regions_have_k_sectors_and_no_overlap():
    r = make_regions(focus_size=2, n_sectors=8)
    assert r.focus.sum() == 4 and len(r.sectors) == 8
    assert all(s.sum() > 0 for s in r.sectors)
    assert not np.any(r.focus & np.any(r.sectors, axis=0))


def test_focus_search_grows_enhancement_and_saves_files(tmp_path):
    res = run_focus_search(macropixel=8, focus_size=1, n_background_regions=8, iterations=400, seed=0, workdir=tmp_path)
    assert res["final_enhancement"] > 3 * max(res["initial_enhancement"], 1.0)
    assert res["history"][-1] >= res["history"][0] and len(res["history"]) == 401
    assert all((tmp_path / p).exists() for p in res["images"].values())
    run_focus_search(macropixel=16, iterations=50, workdir=tmp_path)
    rows = list(csv.DictReader((tmp_path / CSV_NAME).open()))
    assert [r["experiment"] for r in rows] == ["focus_01", "focus_02"]


class _FocusLLM:
    """Executor model that calls run_focus_search twice in one reply, then reports."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if not any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="", tool_calls=[
                {"name": "run_focus_search", "id": "a", "args": {"macropixel": 16, "focus_size": 1,
                 "n_background_regions": 8, "iterations": 60, "flips_per_step": 1, "seed": 0}},
                {"name": "run_focus_search", "id": "b", "args": {"macropixel": 8, "focus_size": 2,
                 "n_background_regions": 8, "iterations": 60, "flips_per_step": 1, "seed": 1}},
            ])
        return AIMessage(content="two experiments done")


def test_executor_runs_focus_search_and_finish_is_allowed(tmp_path, settings):
    state = initial_state("Focus the beam.")
    out = Executor(_FocusLLM(), settings, tmp_path)(state)
    turn = out["dialogue"][0]["content"]
    assert len(out["experiments"]) == 2
    for exp in out["experiments"]:
        assert exp["result_line"] in turn  # settings and enhancement are in the dialogue verbatim
    assert "run_focus_search(macropixel=16, focus_size=1" in turn
    state["dialogue"] += out["dialogue"]
    state["experiments"] = out["experiments"]
    assert finish_allowed(state)


def test_auto_approve_answers_yes_only_to_hard_questions(monkeypatch, capsys):
    assert _auto_ask({"question": "Run the focusing tool?", "kind": "hard"}) == "yes"
    assert "auto-approve" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "the test RMSE")
    assert _auto_ask({"question": "Which error?", "kind": "ambiguous"}) == "the test RMSE"
