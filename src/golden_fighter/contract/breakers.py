"""'Break it on purpose' scenarios (brief, section 06): corrupt a COPY of Crew 1's artifacts.

Each scenario copies artifacts/analyst into a separate directory and damages the copy, so the
real artifacts are never touched. Used by the tests, scripts/break_data.py and the app demo.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from golden_fighter.contract.validator import REQUIRED_ARTIFACTS

MONEY = "avg_additional_charges_total"
COPY_MARKER = ".golden-fighter-demo"


def _edit_csv(out_dir: Path, change) -> None:
    path = out_dir / "clean_data.csv"
    df = pd.read_csv(path)
    df = change(df)
    df.to_csv(path, index=False, lineterminator="\n")


def _edit_contract(out_dir: Path, change) -> None:
    path = out_dir / "dataset_contract.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def _set(df: pd.DataFrame, rows: int, column: str, value) -> pd.DataFrame:
    if not isinstance(value, (int, float)):
        df[column] = df[column].astype(object)
    df.loc[: rows - 1, column] = value
    return df


SCENARIOS = {
    "unit_x100": ("Money column switched from dollars to cents (x100), the Harbor & Vale incident.",
                  lambda d: _edit_csv(d, lambda df: df.assign(**{MONEY: df[MONEY] * 100}))),
    "rename_column": ("A column was renamed to a 'new naming convention'.",
                      lambda d: _edit_csv(d, lambda df: df.rename(columns={MONEY: "extra_spend"}))),
    "drop_column": ("A column the model needs was dropped.",
                    lambda d: _edit_csv(d, lambda df: df.drop(columns=["lifetime"]))),
    "invalid_value": ("contract_period contains a value the contract does not allow (3 months).",
                      lambda d: _edit_csv(d, lambda df: _set(df, 5, "contract_period", 3))),
    "out_of_range": ("Impossible ages (150 years).",
                     lambda d: _edit_csv(d, lambda df: _set(df, 3, "age", 150))),
    "missing_values": ("Missing values in a column that must never be null.",
                       lambda d: _edit_csv(d, lambda df: _set(df, 10, "age", float("nan")))),
    "non_numeric": ("Text in a numeric column.",
                    lambda d: _edit_csv(d, lambda df: _set(df, 2, "age", "unknown"))),
    "target_not_binary": ("The churn label contains a value other than 0/1.",
                          lambda d: _edit_csv(d, lambda df: _set(df, 4, "churn", 2))),
    "truncated_data": ("Only 10% of the rows arrived.",
                       lambda d: _edit_csv(d, lambda df: df.head(len(df) // 10))),
    "leaky_contract": ("Someone 'unblocked' the leaky current-month column in the contract.",
                       lambda d: _edit_contract(d, lambda c: c.update(excluded_from_primary_model=[]))),
    "contract_missing_field": ("The contract lost its target definition.",
                               lambda d: _edit_contract(d, lambda c: c.pop("target"))),
    "corrupt_contract": ("dataset_contract.json is not valid JSON (truncated file).",
                         lambda d: (d / "dataset_contract.json").write_text('{"contract_version": "1.1", ',
                                                                            encoding="utf-8")),
    "missing_artifact": ("insights.md was never produced.",
                         lambda d: (d / "insights.md").unlink()),
}


def make_broken_copy(source_dir: Path, scenario: str, out_dir: Path) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS)}")
    source = source_dir.resolve()
    destination = out_dir.resolve()
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("Refusing to break the original artifacts; choose a different output directory")
    if out_dir.is_symlink() or destination != out_dir.absolute():
        raise ValueError("Refusing to overwrite a linked output directory")
    if out_dir.exists():
        entries = list(out_dir.iterdir())
        marker = out_dir / COPY_MARKER
        allowed = {*REQUIRED_ARTIFACTS, COPY_MARKER}
        if entries and (not marker.is_file() or marker.is_symlink()
                        or marker.read_text(encoding="utf-8") != "Golden Fighter demo copy\n"):
            raise ValueError("Refusing to overwrite an unowned output directory")
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in entries):
            raise ValueError("Refusing to overwrite unrelated output contents")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COPY_MARKER).write_text("Golden Fighter demo copy\n", encoding="utf-8")
    for name in REQUIRED_ARTIFACTS:
        shutil.copy2(source_dir / name, out_dir / name)
    SCENARIOS[scenario][1](out_dir)
    return out_dir
