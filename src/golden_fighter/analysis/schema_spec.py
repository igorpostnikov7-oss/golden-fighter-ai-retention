"""Domain knowledge about the gym churn dataset: names, types, units, bounds and leakage notes.

This is the single place where the analysts' understanding of each column lives.
Cleaning uses the bounds to reject impossible rows; the contract builder turns it into
dataset_contract.json for the Data Scientist crew.
"""
from dataclasses import dataclass

TARGET = "churn"
ID_COLUMN = "member_id"

# Raw Kaggle column name -> clean snake_case name
RAW_TO_CLEAN = {
    "gender": "gender",
    "Near_Location": "near_location",
    "Partner": "partner",
    "Promo_friends": "promo_friends",
    "Phone": "phone",
    "Contract_period": "contract_period",
    "Group_visits": "group_visits",
    "Age": "age",
    "Avg_additional_charges_total": "avg_additional_charges_total",
    "Month_to_end_contract": "month_to_end_contract",
    "Lifetime": "lifetime",
    "Avg_class_frequency_total": "avg_class_frequency_total",
    "Avg_class_frequency_current_month": "avg_class_frequency_current_month",
    "Churn": "churn",
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # "int" or "float"
    role: str  # "identifier" | "feature" | "target"
    description: str
    unit: str
    min: float | None = None
    max: float | None = None
    allowed_values: tuple[int, ...] | None = None
    sensitive: bool = False
    leakage_risk: str = "none"  # none | low | medium | high
    leakage_note: str | None = None


BINARY = (0, 1)

COLUMN_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec(ID_COLUMN, "int", "identifier", "Row number in the raw file (1-based), added for traceability.",
               "id", min=1),
    ColumnSpec("gender", "int", "feature", "Member gender, anonymised code (0/1).", "category code",
               allowed_values=BINARY, sensitive=True),
    ColumnSpec("near_location", "int", "feature", "Lives or works near the club (1 = yes).", "flag",
               allowed_values=BINARY),
    ColumnSpec("partner", "int", "feature", "Employee of a partner company with a corporate discount (1 = yes).",
               "flag", allowed_values=BINARY),
    ColumnSpec("promo_friends", "int", "feature", "Joined through a 'bring a friend' referral (1 = yes).", "flag",
               allowed_values=BINARY),
    ColumnSpec("phone", "int", "feature", "Provided a contact phone number (1 = yes).", "flag",
               allowed_values=BINARY),
    ColumnSpec("contract_period", "int", "feature", "Length of the current membership contract.", "months",
               allowed_values=(1, 6, 12)),
    ColumnSpec("group_visits", "int", "feature", "Attends group classes (1 = yes).", "flag", allowed_values=BINARY),
    ColumnSpec("age", "int", "feature", "Member age.", "years", min=14, max=90, sensitive=True),
    ColumnSpec("avg_additional_charges_total", "float", "feature",
               "Average spend on extra services (cafe, shop, massage) over the membership.",
               "currency units as provided by the source (not converted)", min=0, max=1000),
    ColumnSpec("month_to_end_contract", "int", "feature", "Months left until the current contract ends.", "months",
               min=1, max=12, leakage_risk="low",
               leakage_note="Known at snapshot time, but tightly coupled to contract_period."),
    ColumnSpec("lifetime", "int", "feature", "Months since the member's first visit.", "months", min=0, max=120),
    ColumnSpec("avg_class_frequency_total", "float", "feature",
               "Average visits per week over the whole membership.", "visits per week", min=0, max=14),
    ColumnSpec("avg_class_frequency_current_month", "float", "feature",
               "Average visits per week in the current month.", "visits per week", min=0, max=14,
               leakage_risk="high",
               leakage_note=("Measured in the same month the churn label refers to, so it partly records the "
                             "churn itself (churners' visits drop sharply). Excluded from the primary model; "
                             "allowed only in a documented leakage ablation.")),
    ColumnSpec(TARGET, "int", "target", "Member left the club (1) or stayed (0).", "flag", allowed_values=BINARY),
)

SPEC_BY_NAME = {spec.name: spec for spec in COLUMN_SPECS}
CLEAN_COLUMN_ORDER = [spec.name for spec in COLUMN_SPECS]

# Structured cross-column rules: (left column, operator, right column, explanation)
CROSS_COLUMN_RULES = (
    ("month_to_end_contract", "<=", "contract_period",
     "Months left on a contract cannot exceed the contract length."),
)
