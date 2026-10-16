"""EvcamInstrument (hardware/devices/evcam/instrument.py) on its own, without an MCP server. Fake camera only."""

import pytest

from agentic_ode.core.state import CAMERA_MEASUREMENT_TOOLS, CAMERA_SETTING_TOOLS
from agentic_ode.hardware.devices.evcam import fake
from agentic_ode.hardware.devices.evcam.instrument import EvcamInstrument


@pytest.fixture
def instrument(tmp_path, monkeypatch):
    monkeypatch.delenv("EVCAM_FAKE_SCENE", raising=False)
    inst = EvcamInstrument(recordings_dir=tmp_path)
    monkeypatch.setenv("EVCAM_FAKE", "1")  # set AFTER the instrument is made: the backend is chosen on first use
    yield inst
    inst.close()


def test_action_kinds_match_the_agent_side_lists(instrument):
    kinds = {spec.name: spec.kind for spec in instrument.actions()}
    assert tuple(n for n, k in kinds.items() if k == "setting") == CAMERA_SETTING_TOOLS
    assert sorted(n for n, k in kinds.items() if k == "measurement") == sorted(CAMERA_MEASUREMENT_TOOLS)
    assert {n for n, k in kinds.items() if k == "read"} == {"camera_info", "get_settings"}


def test_opens_lazily_and_closes(instrument):
    assert not instrument.is_open
    info = instrument.call("camera_info", {})
    assert info.ok and info.result["fake"] is True and instrument.is_open
    instrument.close()
    assert not instrument.is_open


def test_record_and_spectrum_without_a_server(instrument, tmp_path):
    instrument.call("mask_pixels", {"pixels": [list(p) for p in fake.HOT_PIXELS]})
    rec = instrument.call("record", {"seconds": 2.0})
    assert rec.ok and rec.warnings == [] and (tmp_path / f"{rec.result['recording_id']}.npz").exists()
    spec = instrument.call("spectrum", {"recording_id": rec.result["recording_id"]})
    assert spec.ok and abs(spec.result["fundamental_hz"] - 240) <= spec.result["resolution_hz"]


def test_warnings_go_into_the_envelope(instrument):
    rec = instrument.call("record", {"seconds": 0.5})  # hot pixels not masked
    assert rec.ok and len(rec.warnings) == 1 and "still active" in rec.warnings[0]
    assert "warning" not in rec.result  # only to_wire() makes the flat "warning" key
    assert rec.to_wire()["warning"] == rec.warnings[0]


def test_errors_become_failures(instrument):
    r = instrument.call("set_bias", {"name": "bias_foo", "value": 0})
    assert r.ok is False and r.error.startswith("ValueError: unknown bias")
