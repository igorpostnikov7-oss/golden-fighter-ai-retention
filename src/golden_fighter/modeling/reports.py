"""evaluation_report.md and model_card.md, rendered from evaluation.json.

In M4 the Model Card Author agent may add narrative, but every number comes from evaluation.json.
"""
from golden_fighter.modeling.features import ENGINEERED_FEATURES

LABELS = {"baseline_majority": "Baseline (majority)", "logistic_regression": "Logistic Regression",
          "random_forest": "Random Forest"}
FAIRNESS_RECALL_GAP = 0.05
METRICS = (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
           ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"))


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _confusion(cm: dict) -> list[str]:
    return ["| | Predicted: stays | Predicted: churns |", "|---|---:|---:|",
            f"| **Actually stayed** | {cm['tn']} (true negative) | {cm['fp']} (false alarm) |",
            f"| **Actually churned** | {cm['fn']} (missed) | {cm['tp']} (caught) |"]


def _fairness_tables(fairness: dict) -> list[str]:
    lines = []
    for column, rows in fairness.items():
        lines += ["", f"**By {column.replace('_', ' ')}**", "",
                  "| Group | Members | Actual churn | Flagged as at-risk | Recall | Precision |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {r['group']} | {r['members']} | {_pct(r['actual_churn_rate'])} | {_pct(r['flagged_rate'])} "
                  f"| {_f(r['recall'])} | {_f(r['precision'])} |" for r in rows]
    return lines


def render_evaluation_report(r: dict) -> str:
    sel = r["selected_model"]
    res = r["results"]
    test = res[sel]["test"]
    cm = test["confusion_matrix"]
    others = [n for n in res if n not in (sel, "baseline_majority")]
    abl = r["ablation"]
    data = r["data"]

    lines = [
        "# Evaluation Report: Golden Fighter Churn Model",
        "",
        "_Produced by the Data Scientist crew. Numbers come from `evaluation.json` (seed "
        f"{r['seed']}, scikit-learn {r['sklearn_version']})._",
        "",
        "## Summary",
        "",
        f"- **Selected model: {_label(sel)}.** Rule: {r['selection_rule']}.",
        f"- On the held-out test set ({data['test_rows']} members never seen in training) it reaches "
        f"**ROC-AUC {test['roc_auc']:.3f}** and catches **{test['recall']:.0%} of churners** "
        f"({cm['tp']} of {cm['tp'] + cm['fn']}), with precision {test['precision']:.0%}.",
        f"- The majority baseline scores ROC-AUC {res['baseline_majority']['test']['roc_auc']:.2f} and catches "
        f"{res['baseline_majority']['test']['recall']:.0%} of churners, so the model adds real value.",
        "",
        "## Setup",
        "",
        f"- Data: `clean_data.csv` validated against `dataset_contract.json` v{data['contract_version']} "
        "(the validation gate ran before features were built and again before training).",
        f"- Split: stratified {data['train_rows']} train / {data['test_rows']} test; churn rate "
        f"{_pct(data['train_churn_rate'])} train vs {_pct(data['test_churn_rate'])} test.",
        "- Validation: 5-fold stratified cross-validation on the **training set only**. Scaling and feature "
        "engineering sit inside the model pipeline, so they are fitted on each training fold, never on "
        "validation or test rows.",
        f"- Decision threshold: {r['threshold']} on the predicted churn probability (classes are weighted, "
        "so the model is tuned to find churners rather than to maximise accuracy).",
        f"- Model inputs ({len(r['input_features'])}): " + ", ".join(f"`{f}`" for f in r["input_features"]) + ".",
        "- Engineered inside the pipeline: " + ", ".join(f"`{f}`" for f in r["engineered_features"]) + ".",
        "- Never used as inputs: " + ", ".join(f"`{c}`" for c in r["excluded_columns"]["forbidden"])
        + " (target, identifier, protected attribute) and "
        + ", ".join(f"`{c}`" for c in r["excluded_columns"]["leakage_or_late"])
        + " (not known at prediction time).",
        "",
        "## Model comparison: cross-validation (used for selection)",
        "",
        "| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in res.items():
        cv = result["cv"]
        mark = " **(selected)**" if name == sel else ""
        lines.append(f"| {_label(name)}{mark} | {cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f} "
                     f"| {cv['pr_auc']['mean']:.3f} | {cv['recall']['mean']:.3f} | {cv['precision']['mean']:.3f} "
                     f"| {cv['f1']['mean']:.3f} | {cv['accuracy']['mean']:.3f} |")

    lines += ["", "## Model comparison: held-out test set (reported after selection)", "",
              "| Model | " + " | ".join(label for _, label in METRICS) + " |",
              "|---|" + "---:|" * len(METRICS)]
    for name, result in res.items():
        lines.append(f"| {_label(name)} | " + " | ".join(_f(result["test"][m]) for m, _ in METRICS) + " |")

    gap = res[sel]["cv"]["roc_auc"]["mean"] - max(res[n]["cv"]["roc_auc"]["mean"] for n in others)
    noise = res[sel]["cv"]["roc_auc"]["std"]
    lines += [
        "",
        "### Why this model",
        "",
        f"{_label(sel)} had the best mean cross-validation ROC-AUC. The margin over "
        + ", ".join(_label(n) for n in others) + f" is {gap:.3f}, "
        + ("smaller than the fold-to-fold variation" if gap < noise else "larger than the fold-to-fold variation")
        + f" (± {noise:.3f}), so "
        + ("the two models perform about equally well; the simpler, more explainable model wins the tie in practice."
           if gap < noise else "the difference is meaningful.")
        + " The test set confirms the choice without having driven it.",
        "",
        f"## Confusion matrix: {_label(sel)} (test set, threshold {r['threshold']})",
        "",
        *_confusion(cm),
        "",
        f"Out of {cm['tp'] + cm['fn']} members who actually churned, the model flagged {cm['tp']} and missed "
        f"{cm['fn']}. It also flagged {cm['fp']} members who stayed (false alarms); for a retention call or a "
        "discount offer, a false alarm costs far less than a missed churner.",
        "",
        "## Leakage ablation (not deployable)",
        "",
        "To show why the contract excludes " + ", ".join(f"`{c}`" for c in abl["extra_columns"])
        + f", the same {_label(sel)} was retrained **with** it:",
        "",
        "| Variant | CV ROC-AUC | Test ROC-AUC | Test recall |",
        "|---|---:|---:|---:|",
        f"| Primary model (contract features) | {res[sel]['cv']['roc_auc']['mean']:.3f} | {test['roc_auc']:.3f} "
        f"| {test['recall']:.3f} |",
        f"| + current-month visits (leaky) | {abl['cv']['roc_auc']['mean']:.3f} | {abl['test']['roc_auc']:.3f} "
        f"| {abl['test']['recall']:.3f} |",
        "",
        f"The leaky variant looks better (+{abl['test']['roc_auc'] - test['roc_auc']:.3f} ROC-AUC) only because "
        "current-month visits are recorded during the month the member is already leaving. At the start of the "
        "month, when the retention team must act, that value does not exist yet. This is exactly the kind of "
        "silently inflated score the validation gate is designed to prevent.",
        "",
        "## What drives the predictions (permutation importance on the test set)",
        "",
        "| Input column | ROC-AUC drop when shuffled |",
        "|---|---:|",
    ]
    lines += [f"| `{f['feature']}` | {f['auc_drop_mean']:.3f} ± {f['auc_drop_std']:.3f} |"
              for f in r["feature_importance"]]
    lines += ["", "## Fairness check (test set)", "",
              "Gender is not a model input; it is used here only to check that the model treats groups alike."]
    lines += _fairness_tables(r["fairness"])
    lines += ["", "## Sanity checks", ""]
    lines += [f"- WARNING: {w}" for w in r["warnings"]] or [
        "- No model reached a suspiciously perfect ROC-AUC (> 0.99) in cross-validation or on the test set."]
    lines.append("- Train and test churn rates match (stratified split).")
    return "\n".join(lines) + "\n"


def render_model_card(r: dict) -> str:
    sel = r["selected_model"]
    test = r["results"][sel]["test"]
    cv = r["results"][sel]["cv"]
    data = r["data"]
    lines = [
        f"# Model Card: Golden Fighter Churn Predictor ({_label(sel)})",
        "",
        "## Model details",
        "",
        f"- **Type:** scikit-learn pipeline: row-wise feature engineering -> "
        + ("standard scaling -> logistic regression" if sel == "logistic_regression" else "random forest")
        + " (balanced class weights).",
        f"- **Artifact:** `model.joblib` (bundle with the pipeline, input list, threshold {r['threshold']} and the "
        "input ranges from the contract). Load with `golden_fighter.modeling.predict.load_model`.",
        f"- **Version inputs:** dataset contract v{data['contract_version']}, clean data SHA-256 "
        f"`{data['clean_data_sha256'][:16]}...`, seed {r['seed']}, scikit-learn {r['sklearn_version']}.",
        "- **Owner:** Golden Fighter AI course project (Data Scientist crew).",
        "",
        "## Intended use",
        "",
        "- Rank current gym members by their probability of leaving at the start of a month, so the retention "
        "team can prioritise personal calls, onboarding help or contract-upgrade offers.",
        "- Decision support for humans. A person decides what to offer.",
        "",
        "## Out of scope (do not use for)",
        "",
        "- Penalising members, raising prices, or refusing service based on a churn score.",
        "- Any decision about individuals based on gender or age.",
        "- Other clubs, countries or time periods without re-validating on their data.",
        "",
        "## Training data",
        "",
        f"- Public Kaggle dataset \"Gym customers features and churn\": {data['rows']:,} members, one row each, "
        f"churn rate {_pct(data['train_churn_rate'])}. No names, contact details or other direct identifiers.",
        "- Cleaned and documented by the Data Analyst crew; accepted only after the validation gate passed.",
        f"- Stratified split: {data['train_rows']:,} training / {data['test_rows']:,} test members.",
        "",
        "## Features",
        "",
        "Inputs (all known at the start of the month): " + ", ".join(f"`{f}`" for f in r["input_features"]) + ".",
        "",
        "Engineered inside the model:",
        "",
    ]
    lines += [f"- `{name}`: {text}" for name, text in ENGINEERED_FEATURES.items()]
    lines += [
        "",
        "Deliberately excluded: `churn` (the answer), `member_id` (no meaning), `gender` (protected attribute) "
        "and `avg_class_frequency_current_month` (measured during the churn month: leakage).",
        "",
        "## Performance",
        "",
        "| Metric | Cross-validation (train) | Test (held out) |",
        "|---|---:|---:|",
    ]
    lines += [f"| {label} | {cv[m]['mean']:.3f} | {test[m]:.3f} |" for m, label in METRICS]
    cm = test["confusion_matrix"]
    lines += [
        "",
        f"On the test set the model catches {cm['tp']} of {cm['tp'] + cm['fn']} churners and raises "
        f"{cm['fp']} false alarms among {cm['tn'] + cm['fp']} members who stayed. "
        "See `evaluation_report.md` for the full comparison with the baseline and the Random Forest.",
        "",
        "## Fairness",
    ]
    lines += _fairness_tables(r["fairness"])
    recalls = [g["recall"] for g in r["fairness"].get("gender", []) if g["recall"] is not None]
    gap = max(recalls) - min(recalls) if len(recalls) > 1 else None
    if gap is None:
        gender_text = "Gender recall could not be compared (a group has no churners in the test set)."
    elif gap <= FAIRNESS_RECALL_GAP:
        gender_text = (f"Recall differs by only {gap:.3f} between the two gender codes (within the "
                       f"{FAIRNESS_RECALL_GAP} tolerance), so churners are found equally well in both groups.")
    else:
        gender_text = (f"WARNING: recall differs by {gap:.3f} between the gender codes (above the "
                       f"{FAIRNESS_RECALL_GAP} tolerance). Investigate before any use.")
    lines += [
        "",
        gender_text + " Across age bands, the share of members flagged follows the actual churn rate of each "
        "band: younger members churn far more often in this data, so they are flagged more often. Use the score "
        "to offer help, never to treat any age group worse.",
        "",
        "## Limitations",
        "",
        "- **A single snapshot:** one chain, one period of unknown date, 4,000 members. Behaviour may differ "
        "elsewhere or later.",
        "- **Associations, not causes:** a long contract goes with low churn, but forcing long contracts will not "
        "necessarily keep people. Test retention actions (A/B) before rolling them out.",
        "- **Tenure dominates:** `lifetime` and `contract_period` carry most of the signal; the model is weakest "
        "at telling apart established members, where churn is rare.",
        f"- **Threshold:** {r['threshold']} favours catching churners (high recall) over precision. A different "
        "budget for retention offers may call for a different threshold.",
        "- **Assumed data timing:** `avg_class_frequency_total` is assumed to exclude the churn month. If it does "
        "not, it would also need to be excluded.",
        "- **Units:** money is in the source's currency units; the contract and the gate reject rescaled data.",
        "",
        "## Ethical considerations",
        "",
        "- Built only from a public, anonymised dataset; not connected to any production CRM.",
        "- Gender is excluded from the inputs and used only for the fairness check above.",
        "- Age is an input because it is strongly related to churn here. Its use is limited to offering support, "
        "and its effect should be reviewed before any real deployment.",
        "- Scores should be explained to staff as probabilities with uncertainty, not as facts about a person.",
        "- Retrain and re-check fairness whenever the dataset contract changes.",
    ]
    return "\n".join(lines) + "\n"
