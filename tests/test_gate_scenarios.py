"""'Break it on purpose': every scenario must make the gate fail, without touching the originals."""
import hashlib
import json

import pytest

from golden_fighter.contract import gate
from golden_fighter.contract.breakers import SCENARIOS, make_broken_copy
from golden_fighter.contract.validator import REQUIRED_ARTIFACTS, run_gate
from golden_fighter.errors import DataValidationError

EXPECTED_CHECK = {
    "unit_x100": "unit_or_distribution",
    "rename_column": "schema",
    "drop_column": "schema",
    "invalid_value": "allowed_values",
    "out_of_range": "range",
    "missing_values": "nulls",
    "non_numeric": "dtype",
    "target_not_binary": "allowed_values",
    "truncated_data": "row_count",
    "leaky_contract": "contract",
    "contract_missing_field": "contract",
    "corrupt_contract": "contract",
    "missing_artifact": "artifact",
}


def _hashes(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in REQUIRED_ARTIFACTS}


def test_every_scenario_has_an_expectation():
    assert set(EXPECTED_CHECK) == set(SCENARIOS)


def test_gate_passes_on_untouched_artifacts(analyst_out):
    result = run_gate(analyst_out)
    assert result.passed, result.render()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_fails_gate_with_clear_message(analyst_out, tmp_path, scenario):
    originals = _hashes(analyst_out)
    broken = make_broken_copy(analyst_out, scenario, tmp_path / scenario)
    result = run_gate(broken)

    assert not result.passed
    assert EXPECTED_CHECK[scenario] in {issue.check for issue in result.errors}, result.render()
    assert "will NOT run" in result.render()
    with pytest.raises(DataValidationError):
        result.raise_if_failed()
    assert _hashes(analyst_out) == originals, "the original artifacts must never be modified"


def test_breaker_refuses_to_overwrite_originals(analyst_out):
    with pytest.raises(ValueError, match="Refusing"):
        make_broken_copy(analyst_out, "unit_x100", analyst_out)


def test_cli_exit_codes_and_report(analyst_out, tmp_path, capsys):
    report = tmp_path / "report.json"
    assert gate.main(["--analyst-dir", str(analyst_out), "--report", str(report)]) == 0
    assert json.loads(report.read_text())["passed"] is True

    broken = make_broken_copy(analyst_out, "unit_x100", tmp_path / "broken")
    assert gate.main(["--analyst-dir", str(broken), "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["passed"] is False
    assert "FAILED" in capsys.readouterr().out
