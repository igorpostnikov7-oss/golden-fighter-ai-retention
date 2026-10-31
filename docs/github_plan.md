# GitHub publication plan (prepared, NOT executed)

Nothing here has been run. Creating the repository, pushing and opening Pull Requests all wait for the project
owner's approval. The course requires a **public** repository and real Pull Requests.

## Decisions (made by the owner, 2026-10-28)
1. **Data:** the dataset and its derivatives are published under their original CC BY-NC-SA 4.0 terms
   (`data/LICENSE-DATA.md`); `data/raw/gym_churn.csv`, `clean_data.csv` and `features.csv` are committed byte for
   byte (`*.csv -text`).
2. **Code:** MIT License (`LICENSE`), scoped to the code and documentation.
3. **Repository:** `igorpostnikov7-oss/golden-fighter-ai-retention`, public.
4. **Commit identity:** every commit uses the GitHub no-reply address of the account (ID-based form, see
   GitHub's "Email addresses reference"), so no personal e-mail is published. Local working notes are kept out of
   the repository.

## Branches and Pull Requests
The milestone branches are stacked, each built on the previous one:

| PR | Branch → base | Commits | Content |
|---|---|---|---|
| 1 | `m1-data-layer` → `main` | 5 | cleaning, EDA, insights, dataset contract |
| 2 | `m2-validation-gate` → `m1-data-layer` | 5 | validation gate + 13 break scenarios |
| 3 | `m3-modeling` → `m2-validation-gate` | 6 | features, models, evaluation, model card |
| 4 | `m4-crews` → `m3-modeling` | 9 | two CrewAI crews, tools, grounding, source check |
| 5 | `m5-flow` → `m4-crews` | 3 | the CrewAI Flow, failure report, run summary |
| 6 | `m6-streamlit` → `m5-flow` | 4 | Streamlit app (Hebrew) |
| 7 | `m7-docs` → `m6-streamlit` | 6 + this update | README, architecture, checklist, licenses + dataset, fresh-clone fix, presentation (M8 included) |

Merge in order 1 → 7. When a base branch is merged and deleted, GitHub moves the next PR's base to `main`
automatically.

## Commands (run only after approval)
```powershell
# 0. last checks
git status                                   # clean
.\.venv\Scripts\python.exe -m pytest -q      # all pass
git log --all -p | Select-String -Pattern "sk-ant|ANTHROPIC_API_KEY=." # must print nothing

# 1. create the public repo and push every branch
gh repo create igorpostnikov7-oss/golden-fighter-ai-retention --public --source . --remote origin `
   --description "CrewAI Flow with a machine-enforced data contract between two crews (gym churn)"
git push -u origin main m1-data-layer m2-validation-gate m3-modeling m4-crews m5-flow m6-streamlit m7-docs

# 2. one PR per milestone (title + body from the table above; the owner reviews and merges)
gh pr create --base main               --head m1-data-layer      --title "M1: data layer and dataset contract" --body-file docs/pr/m1.md
gh pr create --base m1-data-layer      --head m2-validation-gate --title "M2: validation gate and break scenarios" --body-file docs/pr/m2.md
# ... and so on for m3 to m7
```

## Before every push
- No `.env`, no keys, no logs (`logs/` is gitignored except `.gitkeep`).
- `final_project.html` and the course `.docx` are course materials: gitignored, never published.
- Artifacts are committed only from a passing run.
