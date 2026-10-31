"""Load the raw CSV and turn it into clean_data.csv. The raw file is only ever read."""
import logging
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from golden_fighter.analysis.schema_spec import (
    CLEAN_COLUMN_ORDER,
    COLUMN_SPECS,
    CROSS_COLUMN_RULES,
    ID_COLUMN,
    RAW_TO_CLEAN,
)
from golden_fighter.errors import DataValidationError

log = logging.getLogger(__name__)

# If more than this share of rows breaks domain rules, the data is systematically wrong: stop.
MAX_INVALID_ROW_SHARE = 0.05

OPERATORS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq}


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped_non_numeric_or_null: int = 0
    dropped_duplicates: int = 0
    dropped_by_rule: dict[str, int] = field(default_factory=dict)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV and check it is usable at all. Raises DataValidationError otherwise."""
    if not path.exists():
        raise DataValidationError("load_raw", [f"Raw data file not found: {path}"])
    if path.stat().st_size == 0:
        raise DataValidationError("load_raw", [f"Raw data file is empty: {path}"])
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError("load_raw", [f"Could not parse {path.name} as CSV: {exc}"]) from exc

    missing = sorted(set(RAW_TO_CLEAN) - set(df.columns))
    if missing:
        unexpected = sorted(set(df.columns) - set(RAW_TO_CLEAN))
        issues = [f"Missing required column '{col}'" for col in missing]
        if unexpected:
            issues.append(f"Unexpected columns present: {unexpected} (renamed or wrong file?)")
        raise DataValidationError("load_raw", issues)
    if df.empty:
        raise DataValidationError("load_raw", ["Raw data file has a header but no rows"])

    log.info("Loaded raw data: %d rows x %d columns from %s", len(df), df.shape[1], path.name)
    return df


def _rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks of rows that break a domain rule, keyed by a readable rule name."""
    masks: dict[str, pd.Series] = {}
    for spec in COLUMN_SPECS:
        col = df[spec.name]
        if spec.allowed_values is not None:
            masks[f"{spec.name} not in {list(spec.allowed_values)}"] = ~col.isin(spec.allowed_values)
        if spec.min is not None:
            masks[f"{spec.name} < {spec.min}"] = col < spec.min
        if spec.max is not None:
            masks[f"{spec.name} > {spec.max}"] = col > spec.max
        if spec.dtype == "int":
            masks[f"{spec.name} is not a whole number"] = (col % 1) != 0
    for left, op, right, _ in CROSS_COLUMN_RULES:
        masks[f"not ({left} {op} {right})"] = ~OPERATORS[op](df[left], df[right])
    return masks


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return the cleaned dataset and a report of every change. Does not modify `raw`."""
    report = CleaningReport(rows_in=len(raw), renamed_columns=dict(RAW_TO_CLEAN))
    df = raw[list(RAW_TO_CLEAN)].rename(columns=RAW_TO_CLEAN)
    extra = sorted(set(raw.columns) - set(RAW_TO_CLEAN))
    if extra:
        report.notes.append(f"Ignored extra raw columns: {extra}")

    df.insert(0, ID_COLUMN, range(1, len(df) + 1))
    df = df.apply(pd.to_numeric, errors="coerce")

    bad_values = df.isna().any(axis=1)
    report.dropped_non_numeric_or_null = int(bad_values.sum())
    df = df[~bad_values]

    duplicated = df.drop(columns=ID_COLUMN).duplicated()
    report.dropped_duplicates = int(duplicated.sum())
    df = df[~duplicated]

    masks = _rule_masks(df)
    invalid = pd.Series(False, index=df.index)
    for rule, mask in masks.items():
        count = int(mask.sum())
        if count:
            report.dropped_by_rule[rule] = count
        invalid |= mask

    total_dropped = report.dropped_non_numeric_or_null + report.dropped_duplicates + int(invalid.sum())
    if report.rows_in and total_dropped / report.rows_in > MAX_INVALID_ROW_SHARE:
        issues = [f"{total_dropped} of {report.rows_in} rows ({total_dropped / report.rows_in:.1%}) are invalid, "
                  f"above the {MAX_INVALID_ROW_SHARE:.0%} limit; the source data looks systematically wrong"]
        issues += [f"{rule}: {count} rows" for rule, count in report.dropped_by_rule.items()]
        if report.dropped_non_numeric_or_null:
            issues.append(f"non-numeric or missing values: {report.dropped_non_numeric_or_null} rows")
        raise DataValidationError("clean_data", issues)

    df = df[~invalid]
    for spec in COLUMN_SPECS:
        df[spec.name] = df[spec.name].astype("int64" if spec.dtype == "int" else "float64")
    df = df[CLEAN_COLUMN_ORDER].reset_index(drop=True)

    report.rows_out = len(df)
    log.info("Cleaned data: %d -> %d rows (nulls/non-numeric %d, duplicates %d, rule violations %d)",
             report.rows_in, report.rows_out, report.dropped_non_numeric_or_null, report.dropped_duplicates,
             int(invalid.sum()))
    return df, report


def write_clean_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
