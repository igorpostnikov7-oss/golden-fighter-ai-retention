"""Pydantic model of dataset_contract.json. A contract that doesn't fit this schema is rejected."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservedStats(StrictModel):
    min: float
    max: float
    mean: float


class ColumnContract(StrictModel):
    name: str
    dtype: Literal["int", "float"]
    role: Literal["identifier", "feature", "target"]
    description: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    nullable: bool = False
    allowed_values: list[int] | None = None
    min: float | None = None
    max: float | None = None
    expected_mean_range: tuple[float, float] | None = Field(
        default=None, description="Distribution check: a unit change (e.g. x100) moves the mean out of this range.")
    sensitive: bool = False
    leakage_risk: Literal["none", "low", "medium", "high"] = "none"
    leakage_note: str | None = None
    available_before_churn: bool = Field(
        description="True if the value is fully known at prediction time (before the churn outcome).")
    availability_note: str
    observed: ObservedStats


class CrossColumnRule(StrictModel):
    left: str
    operator: Literal["<=", "<", ">=", ">", "=="]
    right: str
    description: str


class TargetContract(StrictModel):
    name: str
    positive_class: int = 1
    expected_rate_range: tuple[float, float]
    observed_rate: float


class DatasetContract(StrictModel):
    contract_version: str = CONTRACT_VERSION
    dataset_name: str
    produced_by: str
    prediction_time: str
    clean_data_file: str
    clean_data_sha256: str
    row_count_min: int
    row_count_observed: int
    target: TargetContract
    columns: list[ColumnContract]
    cross_column_rules: list[CrossColumnRule]
    forbidden_features: list[str] = Field(description="Columns that must never be model inputs.")
    excluded_from_primary_model: list[str] = Field(
        description="High leakage-risk columns: only allowed in a documented leakage ablation.")
    assumptions: list[str]

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetContract":
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate column names in contract")
        if self.target.name not in names:
            raise ValueError(f"Target '{self.target.name}' is not one of the contract columns")
        if self.target.name not in self.forbidden_features:
            raise ValueError("The target must be listed in forbidden_features")
        for rule in self.cross_column_rules:
            for col in (rule.left, rule.right):
                if col not in names:
                    raise ValueError(f"Cross-column rule refers to unknown column '{col}'")
        blocked = set(self.forbidden_features) | set(self.excluded_from_primary_model)
        for col in self.columns:
            if col.role == "feature" and not col.available_before_churn and col.name not in blocked:
                raise ValueError(f"Feature '{col.name}' is not available before churn but is not excluded "
                                 "from the primary model")
            if col.leakage_risk == "high" and col.name not in blocked:
                raise ValueError(f"High leakage-risk column '{col.name}' must be excluded from the primary model")
        return self

    def column(self, name: str) -> ColumnContract:
        return next(c for c in self.columns if c.name == name)

    @property
    def feature_columns(self) -> list[str]:
        """All columns a model may ever see (includes ablation-only columns)."""
        return [c.name for c in self.columns if c.role == "feature" and c.name not in self.forbidden_features]

    @property
    def primary_feature_columns(self) -> list[str]:
        """Columns the primary model may use: allowed, leakage-safe and known before churn."""
        return [name for name in self.feature_columns
                if name not in self.excluded_from_primary_model and self.column(name).available_before_churn]
