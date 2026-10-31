"""Shared fixtures. Tests use a synthetic dataset with the raw Kaggle schema, so they run without
the real (not yet redistributable) CSV and without any API key."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from golden_fighter.analysis.pipeline import run_analyst_pipeline


def make_raw_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice([1, 6, 12], size=n, p=[0.55, 0.2, 0.25])
    lifetime = rng.poisson(3.5, size=n)
    freq_total = np.round(rng.gamma(4, 0.47, size=n), 4)
    risk = 0.9 - 0.06 * contract - 0.12 * np.minimum(lifetime, 6) + rng.normal(0, 0.15, size=n)
    churn = (risk > 0.35).astype(int)
    freq_current = np.round(np.clip(freq_total - churn * rng.uniform(0, 1, size=n), 0, None), 4)
    return pd.DataFrame({
        "gender": rng.integers(0, 2, n),
        "Near_Location": rng.choice([0, 1], n, p=[0.15, 0.85]),
        "Partner": rng.integers(0, 2, n),
        "Promo_friends": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "Phone": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "Contract_period": contract,
        "Group_visits": rng.integers(0, 2, n),
        "Age": rng.integers(18, 42, n),
        "Avg_additional_charges_total": np.round(rng.gamma(2.2, 66, size=n), 4),
        "Month_to_end_contract": np.array([rng.integers(1, c + 1) for c in contract]).astype(float),
        "Lifetime": lifetime,
        "Avg_class_frequency_total": freq_total,
        "Avg_class_frequency_current_month": freq_current,
        "Churn": churn,
    })


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def raw_csv(tmp_path: Path, raw_df: pd.DataFrame) -> Path:
    path = tmp_path / "raw" / "gym_churn.csv"
    path.parent.mkdir()
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture
def analyst_out(tmp_path: Path, raw_csv: Path) -> Path:
    out_dir = tmp_path / "analyst"
    run_analyst_pipeline(raw_csv, out_dir)
    return out_dir
