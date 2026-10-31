# Data source

- **Dataset:** Gym customers features and churn (Model Fitness gym chain)
- **Creator / Kaggle owner:** Adrian Vinueza (`adrianvinueza`), version 1, last updated 2024-06-09
- **File:** `gym_churn.csv` (renamed from the original `gym_churn_us.csv` in Kaggle's `archive.zip`)
- **Shape:** 4,000 rows × 14 columns, no nulls, no duplicate rows
- **Target:** `Churn` (1 = member left), 1,061 churned / 2,939 stayed (26.5% churn rate)
- **Privacy:** public, anonymised dataset. No personal identifiers. Not connected to any production CRM.
  These are **not** real Golden Fighter customers.

| Field | Value |
|---|---|
| URL | https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn |
| License | **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| License checked | 2026-10-28, from Kaggle's own dataset metadata (`/api/v1/datasets/view/...` and the Croissant metadata both state `CC BY-NC-SA 4.0`) |
| Downloaded | 2026-10-28 |
| SHA-256 | `8B6C2C47F178901FDBC7116D65B593AC392C21514DDE4CE7E83E788CBD7BC7B6` |

## How to get the file
1. Sign in to Kaggle, open the URL above and click **Download** (you get `archive.zip`).
2. Extract `gym_churn_us.csv`, rename it to `gym_churn.csv` and place it in this folder (`data/raw/`).
3. Check it is the exact file the project was built on:
   `Get-FileHash data\raw\gym_churn.csv -Algorithm SHA256` must print the SHA-256 above.

## What the license allows and requires
CC BY-NC-SA 4.0 = Attribution, NonCommercial, ShareAlike.

- **Attribution (BY):** credit "Gym customers features and churn" by Adrian Vinueza (Kaggle), link the
  license, and say that changes were made. This project's changes: columns renamed to snake_case, a
  `member_id` row number added, validated and documented (`artifacts/analyst/cleaning_report.json`).
- **NonCommercial (NC):** use is limited to non-commercial purposes. This course project qualifies.
  Using this data (or a model trained on it) in a real, commercial Golden Fighter product would not:
  for real use, retrain on Golden Fighter's own data.
- **ShareAlike (SA):** files adapted from the data that are shared (for example `clean_data.csv`,
  `features.csv`) must carry the same CC BY-NC-SA 4.0 license.

## Publication status in this repository
- Decided by the project owner: the dataset and its derivatives are included in the repository under their
  original CC BY-NC-SA 4.0 terms, with the notice in [`data/LICENSE-DATA.md`](../LICENSE-DATA.md). A fresh clone
  therefore runs without the download step (the steps above remain for anyone who wants the original file).
- `gym_churn.csv` is committed byte for byte (`.gitattributes`: `*.csv -text`), so its SHA-256 above holds
  after cloning on any OS.
- The project's own source code is licensed separately under MIT (`LICENSE`) and is not covered by the data
  license.
- Provenance note: the Kaggle page does not name an upstream source for the data. The license above is the
  one the Kaggle uploader states (verified again on 2026-10-28); we rely on it as published.
