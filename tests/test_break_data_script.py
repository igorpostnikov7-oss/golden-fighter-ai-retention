"""scripts/break_data.py: every run breaks a fresh copy; earlier folders and the originals are never touched."""
import importlib.util
import shutil
from pathlib import Path

import pytest

from golden_fighter import config
from golden_fighter.contract.breakers import COPY_MARKER
from golden_fighter.flow import artifacts

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "break_data.py"


def _script():
    spec = importlib.util.spec_from_file_location("break_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path, monkeypatch, analyst_out) -> Path:
    root = tmp_path / "project"
    shutil.copytree(analyst_out, root / "artifacts" / "analyst")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr(config, "ANALYST_DIR", root / "artifacts" / "analyst")
    return root


def _fingerprints(folder: Path) -> dict:
    return {p: artifacts.sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def _new_copies(demo: Path, old: set) -> list[Path]:
    return sorted(d for d in demo.iterdir() if d not in old)


def test_old_unmarked_folder_is_left_alone(project, capsys):
    demo = project / "artifacts" / "_broken"
    old_dir = demo / "unit_x100"
    old_dir.mkdir(parents=True)
    (old_dir / "clean_data.csv").write_text("old copy\n", encoding="utf-8")
    (old_dir / "notes.txt").write_text("keep me\n", encoding="utf-8")
    old, originals = _fingerprints(demo), _fingerprints(project / "artifacts" / "analyst")

    assert _script().main(["unit_x100"]) == 1  # the gate refuses the broken copy; no ValueError

    assert {p: h for p, h in _fingerprints(demo).items() if p in old} == old  # nothing old changed
    (new,) = _new_copies(demo, {old_dir})
    assert (new / "analyst" / COPY_MARKER).is_file()
    assert "UNIT CHANGE" in capsys.readouterr().out
    assert _fingerprints(project / "artifacts" / "analyst") == originals


def test_repeated_runs_use_fresh_folders_and_keep_earlier_ones(project):
    demo = project / "artifacts" / "_broken"
    originals = _fingerprints(project / "artifacts" / "analyst")
    assert _script().main(["unit_x100"]) == 1
    (first,) = _new_copies(demo, set())
    first_files = _fingerprints(first)

    assert _script().main(["unit_x100"]) == 1
    assert _script().main(["rename_column"]) == 1

    assert len(_new_copies(demo, set())) == 3
    assert _fingerprints(first) == first_files
    assert _fingerprints(project / "artifacts" / "analyst") == originals


def test_missing_artifacts_give_a_message_not_a_traceback(project, capsys):
    (project / "artifacts" / "analyst" / "clean_data.csv").unlink()
    assert _script().main(["unit_x100"]) == 2
    assert "--no-llm" in capsys.readouterr().err
    assert not (project / "artifacts" / "_broken").exists()


def test_list_shows_all_scenarios(capsys):
    assert _script().main(["--list"]) == 0
    assert "unit_x100" in capsys.readouterr().out
