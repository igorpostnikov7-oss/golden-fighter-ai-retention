"""The validation gate: checks Crew 1's artifacts against dataset_contract.json.

Pure, deterministic Python. It collects every problem it can find (not only the first), so the
error message tells the analysts everything that must be fixed. No LLM is involved in pass/fail.
"""
import json
import logging
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from golden_fighter.contract.builder import sha256_of_file
from golden_fighter.contract.schema import DatasetContract
from golden_fighter.errors import DataValidationError

log = logging.getLogger(__name__)

REQUIRED_ARTIFACTS = ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json")
OPERATORS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq}
MAX_EXAMPLES = 3


@dataclass
class Issue:
    check: str
    message: str
    column: str | None = None

    def __str__(self) -> str:
        where = f" {self.column}:" if self.column else ""
        return f"[{self.check}]{where} {self.message}"


@dataclass
class ValidationResult:
    stage: str
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    checks_run: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def error(self, check: str, message: str, column: str | None = None) -> None:
        self.errors.append(Issue(check, message, column))

    def warn(self, check: str, message: str, column: str | None = None) -> None:
        self.warnings.append(Issue(check, message, column))

    def extend(self, other: "ValidationResult") -> None:
        self.errors += other.errors
        self.warnings += other.warnings
        self.checks_run += other.checks_run

    def render(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"VALIDATION GATE ({self.stage}): {status} - {self.checks_run} checks, "
                 f"{len(self.errors)} errors, {len(self.warnings)} warnings"]
        lines += [f"  ERROR {issue}" for issue in self.errors]
        lines += [f"  WARN  {issue}" for issue in self.warnings]
        if not self.passed:
            lines.append("  -> The Data Scientist crew will NOT run. Fix the data or publish a new contract.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "passed": self.passed, "checks_run": self.checks_run,
                "errors": [asdict(i) for i in self.errors], "warnings": [asdict(i) for i in self.warnings]}

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise DataValidationError(self.stage, [str(issue) for issue in self.errors])


def _fmt(value) -> str:
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float):
        return f"{value:,.6g}"
    return repr(value)


def _examples(values: pd.Series) -> str:
    return "e.g. " + ", ".join(_fmt(v) for v in values.unique()[:MAX_EXAMPLES])


def _check_column(df: pd.DataFrame, col, result: ValidationResult) -> bool:
    """Run all per-column checks. Returns True if the column is numeric (later checks can use it)."""
    values = df[col.name]
    result.checks_run += 1
    nulls = int(values.isna().sum())
    if nulls and not col.nullable:
        result.error("nulls", f"{nulls} missing values, but the contract says the column is never null", col.name)

    result.checks_run += 1
    if not pd.api.types.is_numeric_dtype(values):
        bad = values[pd.to_numeric(values, errors="coerce").isna() & values.notna()]
        result.error("dtype", f"expected {col.dtype} but found non-numeric values ({_examples(bad)})", col.name)
        return False
    present = values.dropna()
    if col.dtype == "int":
        fractional = present[present % 1 != 0]
        if len(fractional):
            result.error("dtype", f"expected whole numbers but {len(fractional)} values have decimals "
                                  f"({_examples(fractional)})", col.name)

    if col.allowed_values is not None:
        result.checks_run += 1
        bad = present[~present.isin(col.allowed_values)]
        if len(bad):
            result.error("allowed_values", f"{len(bad)} values outside the allowed set {col.allowed_values} "
                                           f"({_examples(bad)})", col.name)
    if col.min is not None:
        result.checks_run += 1
        low = present[present < col.min]
        if len(low):
            result.error("range", f"{len(low)} values below the minimum {col.min:g} {col.unit} "
                                  f"({_examples(low)})", col.name)
    if col.max is not None:
        result.checks_run += 1
        high = present[present > col.max]
        if len(high):
            result.error("range", f"{len(high)} values above the maximum {col.max:g} {col.unit} "
                                  f"({_examples(high)})", col.name)
    if col.expected_mean_range is not None and len(present):
        result.checks_run += 1
        mean = float(present.mean())
        low, high = col.expected_mean_range
        if not low <= mean <= high:
            message = (f"mean {mean:,.3f} is outside the contract's expected range [{low:g}, {high:g}] "
                       f"(contract mean {col.observed.mean:g} {col.unit})")
            if col.observed.mean:
                ratio = mean / col.observed.mean
                if ratio >= 5 or ratio <= 0.2:
                    message += f"; values are x{ratio:,.2f} the contract's, which looks like a UNIT CHANGE"
            result.error("unit_or_distribution", message, col.name)
    return True


def validate_dataset(df: pd.DataFrame, contract: DatasetContract, clean_path: Path | None = None
                     ) -> ValidationResult:
    """Check a clean dataset against every rule in the contract."""
    result = ValidationResult(stage="dataset vs contract")
    expected = [c.name for c in contract.columns]

    result.checks_run += 1
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    for name in missing:
        result.error("schema", "required column is missing from clean_data.csv", name)
    for name in unexpected:
        hint = " (renamed column?)" if missing else ""
        result.error("schema", f"column is not in the contract{hint}; Crew 2 may only use contract columns", name)

    result.checks_run += 1
    if len(df) < contract.row_count_min:
        result.error("row_count", f"{len(df)} rows, but the contract requires at least {contract.row_count_min} "
                                  f"(contract was written for {contract.row_count_observed})")

    numeric = {col.name for col in contract.columns if col.name in df.columns and _check_column(df, col, result)}

    for rule in contract.cross_column_rules:
        if {rule.left, rule.right} <= numeric:
            result.checks_run += 1
            bad = ~OPERATORS[rule.operator](df[rule.left], df[rule.right])
            if bad.any():
                result.error("cross_column", f"{int(bad.sum())} rows break '{rule.left} {rule.operator} "
                                             f"{rule.right}': {rule.description}")

    target = contract.target
    if target.name in numeric:
        result.checks_run += 1
        rate = float(df[target.name].mean())
        low, high = target.expected_rate_range
        if not low <= rate <= high:
            result.error("target_rate", f"churn rate {rate:.1%} is outside the expected range "
                                        f"[{low:.1%}, {high:.1%}]", target.name)

    ids = [c.name for c in contract.columns if c.role == "identifier" and c.name in df.columns]
    for name in ids:
        result.checks_run += 1
        dupes = int(df[name].duplicated().sum())
        if dupes:
            result.error("identifier", f"{dupes} duplicate ids; every row must be a different member", name)

    if clean_path is not None:
        result.checks_run += 1
        if sha256_of_file(clean_path) != contract.clean_data_sha256:
            result.warn("fingerprint", "clean_data.csv is not the exact file the contract was written for "
                                       "(SHA-256 differs); content checks above decide pass/fail")
    return result


def validate_model_features(features: list[str], contract: DatasetContract, available_columns,
                            mode: str = "primary") -> ValidationResult:
    """Check the feature list a model is about to use. mode='ablation' allows excluded leakage columns."""
    result = ValidationResult(stage=f"model features ({mode})")
    if mode not in ("primary", "ablation"):
        raise ValueError(f"Unknown mode '{mode}'")
    known = {c.name for c in contract.columns}
    available = set(available_columns)

    result.checks_run += 1
    if not features:
        result.error("features", "no features selected")
    for name in features:
        result.checks_run += 1
        if name == contract.target.name:
            result.error("leakage", "the target is used as a model input (label leakage)", name)
        elif name in contract.forbidden_features:
            result.error("forbidden", "the contract forbids this column as a model input", name)
        elif name not in known:
            result.error("unknown_feature", "not in the contract; Crew 2 may not assume it exists", name)
        else:
            col = contract.column(name)
            if not col.available_before_churn:
                message = f"not available at prediction time: {col.availability_note}"
                (result.error if mode == "primary" else result.warn)("temporal", message, name)
            if name in contract.excluded_from_primary_model:
                if mode == "primary":
                    result.error("leakage", "excluded from the primary model by the contract "
                                            f"(leakage risk '{col.leakage_risk}')", name)
                else:
                    result.warn("leakage_ablation", "leakage-risk column used in an ablation; results from this "
                                                    "model must not be deployed", name)
        if name not in available:
            result.error("missing_feature", "required feature is not present in the data", name)
    return result


def _load_contract(path: Path, result: ValidationResult) -> DatasetContract | None:
    result.checks_run += 1
    try:
        return DatasetContract.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"])
            message = err["msg"].removeprefix("Value error, ")
            if err["type"] == "json_invalid":
                result.error("contract", f"dataset_contract.json is not valid JSON: {message}")
            elif not location:
                result.error("contract", f"dataset_contract.json breaks a contract rule: {message}")
            else:
                result.error("contract", f"dataset_contract.json field '{location}': {message}")
    return None


def run_gate(analyst_dir: Path) -> ValidationResult:
    """The full gate between the crews: artifacts exist, contract is valid, data honours it."""
    result = ValidationResult(stage="crew handoff gate")
    for name in REQUIRED_ARTIFACTS:
        result.checks_run += 1
        path = analyst_dir / name
        if not path.exists() or path.stat().st_size == 0:
            result.error("artifact", f"required Crew 1 artifact is missing or empty: {name}")
    if not result.passed:
        return result

    contract = _load_contract(analyst_dir / "dataset_contract.json", result)
    if contract is None:
        return result

    clean_path = analyst_dir / contract.clean_data_file
    result.checks_run += 1
    try:
        df = pd.read_csv(clean_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError, UnicodeDecodeError) as exc:
        result.error("artifact", f"could not read {contract.clean_data_file}: {exc}")
        return result

    result.extend(validate_dataset(df, contract, clean_path))
    # "Confirm all required features exist before modeling" (brief, section 05).
    result.extend(validate_model_features(contract.primary_feature_columns, contract, df.columns))
    if result.passed:
        log.info("Validation gate PASSED (%d checks, %d warnings)", result.checks_run, len(result.warnings))
    else:
        log.error("Validation gate FAILED with %d errors: %s", len(result.errors), "; ".join(map(str, result.errors)))
    return result


def write_report(result: ValidationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n")
