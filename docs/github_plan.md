# GitHub publication plan (prepared, NOT executed)

Nothing here has been run. Creating the repository, pushing and opening Pull Requests all wait for the project
owner's approval. The course requires a **public** repository and real Pull Requests.

## Decisions needed before publishing
1. **Publish the data files?** The Kaggle license (CC BY-NC-SA 4.0) allows redistribution with attribution, for
   non-commercial use, under the same license. Publishing `data/raw/gym_churn.csv`, `clean_data.csv` and
   `features.csv` lets anyone clone and run everything immediately and puts all 8 deliverables in the repo,
   as the brief asks. If yes: remove the three CSV lines from `.gitignore`, add a `data/LICENSE-DATA.md`
   notice (CC BY-NC-SA 4.0, credit to Adrian Vinueza, "modified: columns renamed, member_id added"), commit.
   If no: the README download step stays, and the checklist marks those two deliverables as "generated on run".
2. **A license for the code?** Without one, others may read the public code but not reuse it. MIT is the common
   choice for course projects. The data keeps its own license either way.
3. **Repository name.** Proposal: `golden-fighter-ai-retention` (account `igorpostnikov7-oss`, public).

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
| 7 | `m7-docs` → `m6-streamlit` | see branch | README, architecture, checklist, license, fresh-clone test, presentation |

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
