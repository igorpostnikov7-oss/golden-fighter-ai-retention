# Data license notice: CC BY-NC-SA 4.0

The dataset and every file derived from it in this repository are licensed under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

The project's source code is licensed separately under the MIT License (see `LICENSE` in the repository root).

## Attribution
- **Work:** "Gym customers features and churn", version 1
- **Creator:** Adrian Vinueza (Kaggle user `adrianvinueza`)
- **Source:** https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn
- **License:** CC BY-NC-SA 4.0, as stated in Kaggle's dataset metadata (`licenseName` in
  `/api/v1/datasets/view/adrianvinueza/gym-customers-features-and-churn` and `license` in the dataset's
  Croissant metadata). Last verified 2026-10-28. The original file `gym_churn_us.csv` is 321,833 bytes, matching
  Kaggle's `totalBytes` for version 1.

## Changes made (this is an adapted version)
- `data/raw/gym_churn.csv`: the original `gym_churn_us.csv`, renamed only (content unchanged;
  SHA-256 `8b6c2c47f178901fdbc7116d65b593ac392c21514dde4ce7e83e788cbd7bc7b6`).
- `artifacts/analyst/clean_data.csv`: columns renamed to snake_case, a `member_id` row number added, values
  validated against domain rules (no rows removed).
- `artifacts/scientist/features.csv`: the model's input columns, a train/test split marker and the target.
- Other files under `artifacts/` (reports, statistics, the dataset contract, the trained model and run
  summaries) are computed from the data and are shared under the same license.

## What the license means for you
- **Attribution:** credit the creator as above, link the license, and say that changes were made.
- **NonCommercial:** no commercial use of the data, of files derived from it, or of the trained model.
- **ShareAlike:** if you share an adapted version, share it under CC BY-NC-SA 4.0.
- No warranties are given. The data is public and anonymised; it contains no names or contact details and is
  not data about real Golden Fighter customers.
