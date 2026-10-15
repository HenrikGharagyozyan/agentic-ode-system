import time

import pytest

from agentic_ode.core.metrics import last_rmse_line, parse_rmse
from agentic_ode.tools import run_file, write_file


def test_run_file_timeout(tmp_path):
    write_file(tmp_path, "slow.py", "import time\ntime.sleep(30)\n")
    t0 = time.time()
    res = run_file(tmp_path, "slow.py", timeout=1)
    assert time.time() - t0 < 10
    assert res.timed_out and res.exit_code is None
    assert "TIMEOUT" in res.stderr


def test_run_file_returns_exit_code_and_streams(tmp_path):
    write_file(tmp_path, "boom.py", "import sys\nprint('out')\nsys.exit(3)\n")
    res = run_file(tmp_path, "boom.py", timeout=20)
    assert res.exit_code == 3 and res.stdout == "out\n" and not res.timed_out


def test_write_file_never_overwrites(tmp_path):
    assert not write_file(tmp_path, "a.py", "x=1").startswith("ERROR")
    assert write_file(tmp_path, "a.py", "x=2").startswith("ERROR")
    assert (tmp_path / "a.py").read_text() == "x=1"


@pytest.mark.parametrize("name", ["../evil.py", "sub/a.py", "a.sh", "a py.py"])
def test_filenames_stay_inside_workdir(tmp_path, name):
    with pytest.raises(ValueError):
        write_file(tmp_path, name, "x=1")


def test_parse_rmse():
    assert parse_rmse("loss 1\nFREE_RUN_TEST_RMSE: 2.5e-3\n") == 2.5e-3
    assert parse_rmse("test rmse is about 0.002") is None
    assert last_rmse_line("FREE_RUN_TEST_RMSE:  0.061056\n") == "FREE_RUN_TEST_RMSE:  0.061056"
