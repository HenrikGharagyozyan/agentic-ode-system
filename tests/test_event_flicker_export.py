"""Offline Event Lab exporter (DV-EV-EXPORT-01): synthetic recordings saved with store.save() in a temporary folder.

No camera, no SDK, no real recording. Some tests replace analysis.spectrum() with a fixed result, to check that the
exporter copies the analyzer's values and never computes its own.
"""

import ast
import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from agentic_ode.core.config import PROJECT_ROOT
from agentic_ode.dualvision import event_export
from agentic_ode.dualvision.comparison import TOP_LEVEL_FIELDS, assess, compare
from agentic_ode.dualvision.event_export import ExportError, export_event_flicker, main, write_json_create_only
from agentic_ode.dualvision.ingest import compare_json_bytes
from agentic_ode.hardware.devices.evcam import analysis, limits, store
from agentic_ode.hardware.devices.evcam.events import EVENT_DTYPE

SECRET_SERIAL = "SERIAL-SECRET-4242"  # must never appear in an export
REAL_CAMERA = {"serial": SECRET_SERIAL, "sensor": "IMX636", "width": 1280, "height": 720, "biases": {},
               "temperature_c": 35}
FAKE_CAMERA = {"serial": "FAKE", "sensor": "IMX636 (simulated)", "width": 1280, "height": 720, "biases": {},
               "temperature_c": 35}
OPERATOR = {"reference_hz": 10.0, "local_time": "2026-10-16T14:34:49+04:00", "run_id": "RUN-1",
            "source_type": "LED_PWM"}


def square_wave(freq_hz: float, seconds: float, per_edge: int = 40, duty: float = 0.4, seed: int = 0) -> np.ndarray:
    """Events of a light that blinks at freq_hz: per_edge ON events at each rising edge, per_edge OFF events at each
    falling edge, plus a little random noise. t in us, from 0, sorted. An edge is spread over 7 % of the period
    (like PWM_JITTER of the fake camera): a real light does not switch at once, so the high harmonics stay weak."""
    rng = np.random.default_rng(seed)
    period = 1e6 / freq_hz
    rises = np.arange(0, seconds * 1e6, period)
    falls = rises + duty * period
    falls = falls[falls < seconds * 1e6]
    t_on = np.repeat(rises, per_edge) + rng.normal(0, 0.07 * period, rises.size * per_edge)
    t_off = np.repeat(falls, per_edge) + rng.normal(0, 0.07 * period, falls.size * per_edge)
    n_noise = int(500 * seconds)
    t_noise = rng.uniform(0, seconds * 1e6, n_noise)
    t = np.concatenate([t_on, t_off, t_noise])
    p = np.concatenate([np.ones(t_on.size), np.zeros(t_off.size), rng.integers(0, 2, n_noise)])
    ev = np.zeros(t.size, dtype=EVENT_DTYPE)
    ev["x"], ev["y"] = rng.integers(0, limits.SENSOR_W, t.size), rng.integers(0, limits.SENSOR_H, t.size)
    ev["p"], ev["t"] = p, np.clip(t.astype(np.int64), 0, int(seconds * 1e6) - 1)  # inside 0 .. seconds
    return np.sort(ev, order="t")


def save_recording(root, freq_hz=10.0, seconds=2.0, recorded_s=None, fake=False, camera=REAL_CAMERA, drop=()):
    """Save a synthetic recording like EvcamInstrument.record() does and return its recording_id. recorded_s <
    seconds is an early stop: the events then end at recorded_s. drop: meta keys to leave out."""
    recorded_s = seconds if recorded_s is None else recorded_s
    events = square_wave(freq_hz, recorded_s)
    meta = {"created": "2026-10-16T14:34:49+04:00", "note": "", "seconds": seconds, "recorded_s": recorded_s,
            "fake": fake, "camera": dict(camera), "state": {"biases": {}, "mask": [], "roi": None},
            "settings": {"masked_pixels": 0, "biases": {}, "roi": None}}
    for key in drop:
        meta.pop(key)
    return store.save(events, meta, root=root)


def drop_saved_meta_key(root, recording_id, key):
    """Rewrite a saved file without one meta key that store.save() always adds (n_events, recording_id)."""
    events, meta = store.load(recording_id, root=root)
    meta.pop(key)
    np.savez_compressed(root / f"{recording_id}.npz", events=events, meta=np.array(json.dumps(meta)))


def fixed_spectrum(**changes):
    """A spectrum() result as analysis.spectrum() gives it, with fixed numbers."""
    spec = {"signal": "on_minus_off", "bin_us": 100, "fmin_hz": 2.0, "fmax_hz": 2000.0, "min_snr": 8.0,
            "seconds": 2.0, "resolution_hz": 0.5, "noise_floor": 1.234, "dominant_hz": 70.5,
            "fundamental_hz": 10.5,
            "peaks": [{"hz": 70.5, "rel_amp": 1.0, "snr": 47.0}, {"hz": 10.5, "rel_amp": 0.7, "snr": 34.2}],
            "harmonics": [{"k": 2, "hz": 21.0, "rel_amp": 0.05, "snr": 2.1},
                          {"k": 3, "hz": 31.5, "rel_amp": 0.9, "snr": 41.0},
                          {"k": 4, "hz": 42.0, "rel_amp": 0.04, "snr": None},
                          {"k": 5, "hz": 52.5, "rel_amp": 0.95, "snr": 45.0}]}
    spec.update(changes)
    return spec


class FakeSpectrum:
    """Stands in for analysis.spectrum(): returns a copy of .result and remembers each call's kwargs."""

    def __init__(self):
        self.result = fixed_spectrum()
        self.calls = []

    def __call__(self, events, **kwargs):
        self.calls.append(kwargs)
        return copy.deepcopy(self.result)


@pytest.fixture
def fake_spectrum(monkeypatch):
    fake = FakeSpectrum()
    monkeypatch.setattr(analysis, "spectrum", fake)
    return fake


# ---- acquisition ----

def test_valid_completed_recording(tmp_path):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    acq = doc["acquisition"]
    assert acq == {"request_mode": "duration", "requested_s": 2.0, "actual_s": 2.0,
                   "samples": store.load(rid, root=tmp_path)[0].size, "sample_kind": "events", "fps": None,
                   "exposure_us": None, "early_stop": False}
    assert doc["status"] == "ok" and doc["warnings"] == []
    assert abs(doc["estimate"]["f0_hz"] - 10.0) <= doc["estimate"]["resolution_hz"]
    summary = assess(doc, "event")
    assert summary["problems"] == [] and summary["eligibility"] == "valid"


def test_early_stop_at_event_cap(tmp_path, monkeypatch):
    rid = save_recording(tmp_path, seconds=2.0, recorded_s=1.5)
    n = store.load(rid, root=tmp_path)[0].size
    monkeypatch.setattr(limits, "MAX_EVENTS", n)  # the saved events reach the (lowered) cap
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert doc["acquisition"]["early_stop"] is True
    assert doc["warnings"][0] == (f"stopped early: {n} events in 1.500 s of the requested 2 s "
                                  f"(the {n} event cap was reached)")
    assert "EARLY_STOP" in assess(doc, "event")["reasons"]


def test_early_stop_below_cap_does_not_claim_the_cap(tmp_path):
    rid = save_recording(tmp_path, seconds=2.0, recorded_s=1.5)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert doc["acquisition"]["early_stop"] is True
    assert "cap" not in doc["warnings"][0]


def test_requested_and_actual_duration_kept_apart(tmp_path):
    rid = save_recording(tmp_path, seconds=3.0, recorded_s=1.25)
    acq = export_event_flicker(rid, tmp_path, **OPERATOR)["acquisition"]
    assert (acq["requested_s"], acq["actual_s"]) == (3.0, 1.25)


@pytest.mark.parametrize("key", ["recorded_s", "seconds"])
def test_missing_duration_rejected(tmp_path, key, fake_spectrum):
    rid = save_recording(tmp_path, drop=(key,))
    with pytest.raises(ExportError, match=key):
        export_event_flicker(rid, tmp_path, **OPERATOR)
    assert fake_spectrum.calls == []  # refused before any analysis


@pytest.mark.parametrize("key", ["n_events", "recording_id"])
def test_missing_store_field_rejected(tmp_path, key):
    rid = save_recording(tmp_path)
    drop_saved_meta_key(tmp_path, rid, key)
    with pytest.raises(ExportError):
        export_event_flicker(rid, tmp_path, **OPERATOR)


@pytest.mark.parametrize("meta_change", [{"recorded_s": 2.5}, {"recorded_s": 0}, {"recorded_s": None},
                                         {"seconds": "2"}, {"n_events": 1}, {"n_events": True}])
def test_contradictory_metadata_rejected(tmp_path, meta_change):
    rid = save_recording(tmp_path)
    events, meta = store.load(rid, root=tmp_path)
    meta.update(meta_change)
    np.savez_compressed(tmp_path / f"{rid}.npz", events=events, meta=np.array(json.dumps(meta)))
    with pytest.raises(ExportError):
        export_event_flicker(rid, tmp_path, **OPERATOR)


def test_missing_recording_error_has_no_path(tmp_path):
    with pytest.raises(ExportError) as e:
        export_event_flicker("20261016-143449-ed6b74", tmp_path, **OPERATOR)
    assert str(tmp_path) not in str(e.value)
    with pytest.raises(ExportError, match="bad recording_id"):
        export_event_flicker("../x", tmp_path, **OPERATOR)


# ---- operator values ----

@pytest.mark.parametrize("freq", [10.0, 50.0])
def test_explicit_reference_is_kept_and_f0_is_measured(tmp_path, freq):
    rid = save_recording(tmp_path, freq_hz=freq)
    doc = export_event_flicker(rid, tmp_path, **{**OPERATOR, "reference_hz": freq})
    assert doc["source"] == {"type": "LED_PWM", "reference_hz": freq, "duty": None, "distance_cm": None,
                             "set_by": "manual"}
    assert abs(doc["estimate"]["f0_hz"] - freq) <= doc["estimate"]["resolution_hz"]


def test_reference_never_replaces_the_measurement(tmp_path, fake_spectrum):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **{**OPERATOR, "reference_hz": 50.0})
    assert doc["estimate"]["f0_hz"] == 10.5  # the analyzer's value, not 50


@pytest.mark.parametrize("change", [{"reference_hz": 0}, {"reference_hz": float("nan")}, {"reference_hz": True},
                                    {"reference_hz": None}, {"local_time": "2026-10-16T14:34:49"},
                                    {"local_time": "2026-10-16"}, {"local_time": "yesterday"},
                                    {"local_time": None}, {"run_id": ""}, {"source_type": " "},
                                    {"trial_id": ""}, {"duty": 0}, {"duty": 1.0}, {"duty": 50},
                                    {"distance_cm": -3}])
def test_bad_operator_values_rejected(tmp_path, change, fake_spectrum):
    rid = save_recording(tmp_path)
    with pytest.raises(ExportError):
        export_event_flicker(rid, tmp_path, **{**OPERATOR, **change})


def test_optional_operator_values(tmp_path):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR, trial_id="T-07", duty=0.4, distance_cm=30.0)
    assert doc["trial_id"] == "T-07"
    assert (doc["source"]["duty"], doc["source"]["distance_cm"]) == (0.4, 30.0)
    assert doc["local_time"] == OPERATOR["local_time"]


def test_no_trial_id_is_invented(tmp_path):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert "trial_id" not in doc  # not taken from the file name, the recording_id or a time


# ---- analysis ----

def test_analyzer_gets_fmin_2_and_fmax_2000(tmp_path, monkeypatch):
    calls = []
    real_spectrum = analysis.spectrum

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_spectrum(*args, **kwargs)

    monkeypatch.setattr(analysis, "spectrum", spy)
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert calls == [{"fmin_hz": 2.0, "fmax_hz": 2000.0}]
    assert "fmin_hz=2," in doc["analysis"] and "fmax_hz=2000," in doc["analysis"]


def test_spectral_results_are_copied_unchanged(tmp_path, fake_spectrum):
    fake_spectrum.result = fixed_spectrum(warning="the recording is only 0.50 s long: ...")
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert doc["estimate"] == {"f0_hz": 10.5, "resolution_hz": 0.5, "snr": 34.2, "method": "fft",
                               "harmonics_hz": [31.5, 52.5]}  # only harmonics with snr >= min_snr
    assert doc["warnings"] == ["the recording is only 0.50 s long: ..."]


def test_snr_is_null_when_the_fundamental_is_not_a_peak(tmp_path, fake_spectrum):
    fake_spectrum.result = fixed_spectrum(peaks=[{"hz": 70.5, "rel_amp": 1.0, "snr": 47.0}])
    rid = save_recording(tmp_path)
    estimate = export_event_flicker(rid, tmp_path, **OPERATOR)["estimate"]
    assert estimate["f0_hz"] == 10.5 and estimate["snr"] is None  # not computed here


def test_missing_frequency_stays_null(tmp_path, fake_spectrum):
    fake_spectrum.result = fixed_spectrum(dominant_hz=None, fundamental_hz=None, peaks=[], harmonics=[])
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert doc["estimate"]["f0_hz"] is None and doc["estimate"]["snr"] is None
    summary = assess(doc, "event")
    assert summary["eligibility"] == "not_recoverable" and summary["f0_status"] == "not_detected"


def test_analyzer_error_becomes_export_error(tmp_path):
    rid = save_recording(tmp_path, seconds=0.0001)  # far too short for 2..2000 Hz
    with pytest.raises(ExportError, match="analyzer"):
        export_event_flicker(rid, tmp_path, **OPERATOR)


# ---- provenance and privacy ----

@pytest.mark.parametrize("fake, camera, real", [(False, REAL_CAMERA, True), (True, FAKE_CAMERA, False),
                                                (False, FAKE_CAMERA, None), (None, REAL_CAMERA, None)])
def test_hardware_provenance(tmp_path, fake, camera, real):
    rid = save_recording(tmp_path, fake=fake, camera=camera, drop=("fake",) if fake is None else ())
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    assert doc["hardware"] == {"model": None, "sensor": camera["sensor"], "real": real}
    kind = {True: "real", False: "simulated", None: "undeclared"}[real]
    assert doc["analysis"].startswith(f"offline re-analysis of a saved {kind} recording")


def test_no_serial_path_or_events_leak(tmp_path):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    text = json.dumps(doc)
    assert SECRET_SERIAL not in text and "serial" not in text
    assert str(tmp_path) not in text and str(tmp_path.as_posix()) not in text
    assert "events" not in doc and len(text) < 2000  # a summary, not an event array


# ---- contract ----

def test_contract_compatible_with_comparator_and_ingest(tmp_path):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR, trial_id="T-1")
    assert set(TOP_LEVEL_FIELDS) <= set(doc)
    out = tmp_path / "event.json"
    write_json_create_only(doc, out)
    baseline = json.loads((PROJECT_ROOT / "report" / "lumen" / "20261016-143449-ed6b74.json").read_text())
    written = json.loads(out.read_text(encoding="utf-8"))
    assert set(baseline) <= set(written)  # the schema reference only; its numbers are not used
    for part in ("source", "acquisition", "estimate"):
        assert set(baseline[part]) <= set(written[part])
    frame = {**copy.deepcopy(written), "lab": "frame", "recording_id": "F-REC",
             "hardware": {"model": "M", "sensor": "frame", "real": True},
             "acquisition": {"requested_s": 2.0, "actual_s": 2.0, "samples": 400, "sample_kind": "frames",
                             "fps": 200.0, "exposure_us": 1000.0, "early_stop": False},
             "warnings": [], "analysis": "synthetic"}
    result = compare_json_bytes(json.dumps(frame).encode(), out.read_bytes())
    assert result["comparison"]["event"]["problems"] == []
    assert result["inputs"]["event"]["acquisition"]["declared"] == "real_hardware"


def test_no_automatic_pairing(tmp_path):
    rid = save_recording(tmp_path)
    doc = export_event_flicker(rid, tmp_path, **OPERATOR)
    frame = {**doc, "lab": "frame", "trial_id": "T-1",
             "acquisition": {"requested_s": 2.0, "actual_s": 2.0, "samples": 400, "sample_kind": "frames",
                             "fps": 200.0, "exposure_us": 1000.0, "early_stop": False}}
    result = compare(frame, doc)
    assert result["pairing"]["status"] == "unpaired" and result["comparison_valid"] is False
    paired = compare(frame, export_event_flicker(rid, tmp_path, **OPERATOR, trial_id="T-1"))
    assert paired["pairing"]["status"] == "paired"


# ---- output file ----

def test_existing_file_is_not_overwritten(tmp_path):
    out = tmp_path / "event.json"
    out.write_text("old", encoding="utf-8")
    with pytest.raises(ExportError, match="already exists"):
        write_json_create_only({"a": 1}, out)
    assert out.read_text(encoding="utf-8") == "old"


def test_output_must_be_json_in_an_existing_folder(tmp_path):
    with pytest.raises(ExportError):
        write_json_create_only({"a": 1}, tmp_path / "event.txt")
    with pytest.raises(ExportError, match="folder"):
        write_json_create_only({"a": 1}, tmp_path / "missing" / "event.json")
    assert not (tmp_path / "missing").exists()


def cli_args(rid, root, out, *extra):
    return ["--recording-id", rid, "--recordings-dir", str(root), "--out", str(out), "--reference-hz", "10",
            "--local-time", "2026-10-16T14:34:49+04:00", "--run-id", "RUN-1", "--source-type", "LED_PWM", *extra]


def test_cli_writes_once_and_refuses_overwrite(tmp_path, capsys):
    rid = save_recording(tmp_path)
    out = tmp_path / "event.json"
    assert main(cli_args(rid, tmp_path, out, "--trial-id", "T-9")) == 0
    first = out.read_bytes()
    assert json.loads(first)["trial_id"] == "T-9"
    assert main(cli_args(rid, tmp_path, out)) == 2
    assert out.read_bytes() == first
    assert str(tmp_path) not in capsys.readouterr().out


def test_cli_refuses_incomplete_recording(tmp_path):
    rid = save_recording(tmp_path, drop=("recorded_s",))
    out = tmp_path / "event.json"
    assert main(cli_args(rid, tmp_path, out)) == 2
    assert not out.exists()


def test_original_recording_untouched(tmp_path):
    rid = save_recording(tmp_path)
    path = tmp_path / f"{rid}.npz"
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    export_event_flicker(rid, tmp_path, **OPERATOR)
    write_json_create_only(export_event_flicker(rid, tmp_path, **OPERATOR), tmp_path / "e.json")
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([path.name, "e.json"])


# ---- isolation ----

def test_module_imports_only_offline_parts():
    tree = ast.parse(open(event_export.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"__future__", "argparse", "json", "math", "sys", "datetime", "pathlib",
                        "agentic_ode.dualvision.comparison", "agentic_ode.hardware.devices.evcam"}


def test_import_loads_no_camera_sdk_mcp_agents_or_network():
    code = ("import sys, agentic_ode.dualvision.event_export\n"
            "print('\\n'.join(sorted(sys.modules)))")
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            cwd=PROJECT_ROOT).stdout.split()
    forbidden = ("metavision", "mcp", "langgraph", "langchain", "openai", "httpx", "requests", "socket",
                 "agentic_ode.hardware.devices.evcam.driver", "agentic_ode.hardware.devices.evcam.fake",
                 "agentic_ode.hardware.devices.evcam.instrument", "agentic_ode.agents", "agentic_ode.graph")
    assert [m for m in loaded if m.split(".")[0] in forbidden or m in forbidden] == []
