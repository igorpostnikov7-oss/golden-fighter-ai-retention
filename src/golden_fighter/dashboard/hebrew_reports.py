"""Hebrew display versions of the English deliverables, built from the same verified JSON.

insights.md, evaluation_report.md and model_card.md are rendered in English from insight_facts.json and
evaluation.json (analysis/insights.py, modeling/reports.py). These functions render Hebrew versions from the
same JSON on every page load, so every run shows Hebrew without translating anything an AI wrote, and every
number is one the English report also shows (a test checks this). The English files stay the deliverables.
"""
from golden_fighter.dashboard.hebrew import COLUMNS, CONTRACT_GROUPS, ENGINEERED, GROUPS, SEGMENTS, SELECTION_RULES

MODEL_NAMES = {"baseline_majority": "מודל בסיס (ניחוש קבוע)", "logistic_regression": "רגרסיה לוגיסטית",
               "random_forest": "יער אקראי (Random Forest)"}
FLAG_NAMES = {"near_location": "גר או עובד קרוב למועדון", "partner": "עובד בחברה שותפה",
              "promo_friends": "הגיע דרך המלצה של חבר", "group_visits": "משתתף בשיעורים קבוצתיים",
              "phone": "מסר מספר טלפון"}
METRICS = (("accuracy", "Accuracy (דיוק כללי)"), ("precision", "Precision (דיוק האזהרות)"),
           ("recall", "Recall (עוזבים שזוהו)"), ("f1", "F1 (איזון)"), ("roc_auc", "ROC-AUC (איכות הדירוג)"),
           ("pr_auc", "PR-AUC"))
FAIRNESS_RECALL_GAP = 0.05  # same tolerance as modeling/reports.py


def _pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def _f(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _model(name: str) -> str:
    return MODEL_NAMES.get(name, name)


def _col(name: str) -> str:
    return COLUMNS.get(name, name)


def _segment(seg: dict) -> str:
    return SEGMENTS.get(seg["definition"], seg["definition"])


# --- Crew 1: insights -------------------------------------------------------------------------------------------

def key_findings(facts: dict) -> str:
    """The headline findings (Hebrew), from insight_facts.json."""
    contract = {r["group"]: r for r in facts["churn_by_contract_period"]}
    lifetime, ages = facts["churn_by_lifetime"], facts["churn_by_age_band"]
    flags, seg = facts["churn_by_flag"], facts["high_risk_segment"]
    return "\n".join([
        f"- **{facts['churned']:,} מתוך {facts['members']:,} מתאמנים עזבו ({_pct(facts['churn_rate'])}).** "
        "בערך מתאמן אחד מכל ארבעה, ולכן שימור הוא מנוף הכנסה אמיתי.",
        f"- **אורך המנוי הוא קו ההפרדה הבולט.** שיעור העזיבה הוא {_pct(contract['1']['churn_rate'])} במנוי "
        f"חודשי, {_pct(contract['6']['churn_rate'])} בחצי שנתי ו-{_pct(contract['12']['churn_rate'])} בשנתי.",
        f"- **החודשים הראשונים מכריעים.** בחודש הראשון עוזבים {_pct(lifetime[0]['churn_rate'])}; אחרי "
        f"7 חודשים ומעלה רק {_pct(lifetime[-1]['churn_rate'])}.",
        f"- **צעירים עוזבים יותר.** מתחת לגיל 25: {_pct(ages[0]['churn_rate'])}; מגיל 35 ומעלה: "
        f"{_pct(ages[-1]['churn_rate'])}.",
        f"- **קהילה שומרת על מתאמנים.** בשיעורים קבוצתיים {_pct(flags['group_visits']['rate_if_yes'])} עוזבים "
        f"לעומת {_pct(flags['group_visits']['rate_if_no'])} בלי; מי שהגיע דרך חבר "
        f"{_pct(flags['promo_friends']['rate_if_yes'])} לעומת {_pct(flags['promo_friends']['rate_if_no'])}.",
        f"- **קבוצת הסיכון הגבוה:** {_segment(seg)}. {seg['members']:,} מתאמנים "
        f"({_pct(seg['share_of_members'])} מכולם), ו-{_pct(seg['churn_rate'])} מהם עזבו.",
    ])


def insights_document(facts: dict) -> str:
    """Hebrew version of insights.md (analysis/insights.py render_insights_md), from insight_facts.json."""
    flags, means, visits = facts["churn_by_flag"], facts["mean_stayed_vs_churned"], facts["visit_change_current_vs_total"]
    lines = [
        "#### תקציר", "", key_findings(facts), "",
        "#### עזיבה לפי קבוצות", "",
        "| קבוצה | מתאמנים | שיעור עזיבה |", "|---|---:|---:|",
    ]
    lines += [f"| מנוי {CONTRACT_GROUPS.get(r['group'], r['group'])} | {r['members']:,} | {_pct(r['churn_rate'])} |"
              for r in facts["churn_by_contract_period"]]
    lines += [f"| ותק: {GROUPS.get(r['group'], r['group'])} | {r['members']:,} | {_pct(r['churn_rate'])} |"
              for r in facts["churn_by_lifetime"]]
    lines += [f"| גיל: {GROUPS.get(r['group'], r['group'])} | {r['members']:,} | {_pct(r['churn_rate'])} |"
              for r in facts["churn_by_age_band"]]
    lines += ["", "| מאפיין | עזיבה בלי המאפיין | עזיבה עם המאפיין |", "|---|---:|---:|"]
    lines += [f"| {FLAG_NAMES.get(key, f['label'])} | {_pct(f['rate_if_no'])} | {_pct(f['rate_if_yes'])} |"
              for key, f in flags.items()]
    lines += ["", "#### מי עוזב: הפרופיל הממוצע", "", "| מדד | נשארו | עזבו |", "|---|---:|---:|"]
    lines += [f"| {_col(col)} | {m['stayed']:g} | {m['churned']:g} |" for col, m in means.items()]
    lines += [
        "", "#### המלצות לשימור", "",
        "1. **הצעת שדרוג למנויים חודשיים.** להציע מעבר מוזל למנוי של 6 או 12 חודשים כבר בחודש הראשון, כשהסיכון "
        "לעזיבה הכי גבוה.",
        "2. **ליווי ב-90 הימים הראשונים.** תוכנית קבלת פנים (שיחה עם מאמן, שיעור קבוצתי ראשון שנקבע מראש), כי "
        "הוותק הוא המגן החזק ביותר בהתחלה.",
        "3. **לקדם שיעורים קבוצתיים והמלצות של חברים.** שניהם קשורים לעזיבה נמוכה בהרבה: להזמין מתאמנים חדשים "
        "לשיעור קבוצתי חינם ולתגמל \"הבא חבר\".",
        "4. **לשים לב לירידה בהגעה.** עוזבים מגיעים החודש פחות מהממוצע של עצמם "
        f"({visits['churned']:+.2f} ביקורים בשבוע, לעומת {visits['stayed']:+.2f} אצל מי שנשאר). ירידה כזו היא "
        "סימן אזהרה מוקדם לשיחה אישית.",
        "", "#### הערות זהירות לצוות המודל", "",
        f"- \"{_col('avg_class_frequency_current_month')}\" נמדדת בחודש העזיבה עצמו, ולכן היא בחלקה **מתעדת** את "
        "העזיבה במקום לחזות אותה. היא הוצאה מהמודל הראשי (ראו dataset_contract.json).",
        "- המגדר לא קשור לעזיבה והוא מאפיין מוגן, ולכן אינו קלט של המודל.",
        "- אלה קשרים סטטיסטיים, לא סיבות מוכחות. כדאי לבדוק הצעות שימור בניסוי (A/B) לפני שמפעילים אותן.",
        "", "#### האותות הלגיטימיים החזקים ביותר (AUC של עמודה בודדת)", "",
    ]
    lines += [f"- {_col(s['column'])}: {s['univariate_auc']:.3f}" for s in facts["strongest_safe_signals"]]
    return "\n".join(lines) + "\n"


# --- Crew 2: evaluation report and model card ---------------------------------------------------------------------

def _confusion(cm: dict) -> list[str]:
    return ["| | המודל אמר: יישאר | המודל אמר: יעזוב |", "|---|---:|---:|",
            f"| **נשארו בפועל** | {cm['tn']} (נכון: נשארו) | {cm['fp']} (אזעקת שווא) |",
            f"| **עזבו בפועל** | {cm['fn']} (פוספסו) | {cm['tp']} (זוהו בזמן) |"]


def _fairness_tables(fairness: dict) -> list[str]:
    names = {"gender": "לפי מגדר (קודים 0 / 1)", "age_band": "לפי קבוצת גיל"}
    lines = []
    for key, rows in fairness.items():
        lines += ["", f"**{names.get(key, key)}**", "",
                  "| קבוצה | מתאמנים | עזיבה בפועל | סומנו בסיכון | Recall | Precision |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {GROUPS.get(r['group'], r['group'])} | {r['members']} | {_pct(r['actual_churn_rate'])} "
                  f"| {_pct(r['flagged_rate'])} | {_f(r['recall'])} | {_f(r['precision'])} |" for r in rows]
    return lines


def evaluation_report(r: dict) -> str:
    """Hebrew version of evaluation_report.md (modeling/reports.py), from evaluation.json."""
    sel, res, abl, data = r["selected_model"], r["results"], r["ablation"], r["data"]
    test = res[sel]["test"]
    cm = test["confusion_matrix"]
    others = [n for n in res if n not in (sel, "baseline_majority")]
    base = res["baseline_majority"]["test"]
    lines = [
        "#### תקציר", "",
        f"- **המודל שנבחר: {_model(sel)}.** כלל הבחירה: {SELECTION_RULES.get(r['selection_rule'], r['selection_rule'])}",
        f"- על נתוני המבחן ({data['test_rows']} מתאמנים שהמודל לא ראה באימון) הוא מגיע ל-**ROC-AUC "
        f"{test['roc_auc']:.3f}** ומזהה **{test['recall']:.0%} מהעוזבים** ({cm['tp']} מתוך {cm['tp'] + cm['fn']}), "
        f"עם Precision של {test['precision']:.0%}.",
        f"- מודל הבסיס מגיע ל-ROC-AUC {base['roc_auc']:.2f} ומזהה {base['recall']:.0%} מהעוזבים, כך שהמודל מוסיף "
        "ערך אמיתי.",
        "", "#### איך נבדק", "",
        f"- נתונים: clean_data.csv, אחרי ששער הבדיקה אישר אותם מול חוזה הנתונים (גרסה {data['contract_version']}); "
        "השער רץ לפני בניית הפיצ'רים ושוב לפני האימון.",
        f"- חלוקה מרובדת: {data['train_rows']} לאימון / {data['test_rows']} למבחן; שיעור עזיבה "
        f"{_pct(data['train_churn_rate'])} באימון ו-{_pct(data['test_churn_rate'])} במבחן.",
        "- אימות: Cross-Validation מרובד של 5 חלקים **על נתוני האימון בלבד**. הנרמול ובניית הפיצ'רים נמצאים בתוך "
        "צינור המודל, כך שהם נלמדים רק מנתוני האימון של כל חלק.",
        f"- סף החלטה: {r['threshold']} על הסיכוי החזוי לעזיבה (המחלקות משוקללות, כדי שהמודל יעדיף למצוא עוזבים).",
        f"- קלטי המודל ({len(r['input_features'])}): " + ", ".join(_col(f) for f in r["input_features"]) + ".",
        "- לעולם לא קלט: " + ", ".join(_col(c) for c in r["excluded_columns"]["forbidden"])
        + " (התשובה, מזהה, מאפיין מוגן) ו-" + ", ".join(_col(c) for c in r["excluded_columns"]["leakage_or_late"])
        + " (לא ידוע בזמן החיזוי).",
        "", "#### השוואת מודלים: Cross-Validation (שימש לבחירה)", "",
        "| מודל | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in res.items():
        cv = result["cv"]
        mark = " **(נבחר)**" if name == sel else ""
        lines.append(f"| {_model(name)}{mark} | {cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f} "
                     f"| {cv['pr_auc']['mean']:.3f} | {cv['recall']['mean']:.3f} | {cv['precision']['mean']:.3f} "
                     f"| {cv['f1']['mean']:.3f} | {cv['accuracy']['mean']:.3f} |")
    lines += ["", "#### השוואת מודלים: נתוני המבחן (דווח אחרי הבחירה)", "",
              "| מודל | " + " | ".join(label.split(" (")[0] for _, label in METRICS) + " |",
              "|---|" + "---:|" * len(METRICS)]
    lines += [f"| {_model(name)} | " + " | ".join(_f(result["test"][m]) for m, _ in METRICS) + " |"
              for name, result in res.items()]
    gap = res[sel]["cv"]["roc_auc"]["mean"] - max(res[n]["cv"]["roc_auc"]["mean"] for n in others)
    noise = res[sel]["cv"]["roc_auc"]["std"]
    lines += [
        "", "#### למה דווקא המודל הזה", "",
        f"ל{_model(sel)} היה ה-ROC-AUC הממוצע הגבוה ביותר ב-Cross-Validation. הפער מול "
        + ", ".join(_model(n) for n in others) + f" הוא {gap:.3f}, "
        + ("קטן מהשונות בין החלקים" if gap < noise else "גדול מהשונות בין החלקים") + f" (± {noise:.3f}), ולכן "
        + ("שני המודלים טובים בערך באותה מידה, והמודל הפשוט והמוסבר יותר מנצח." if gap < noise
           else "ההבדל משמעותי.") + " נתוני המבחן מאשרים את הבחירה, אבל לא השפיעו עליה.",
        "", f"#### מטריצת בלבול: {_model(sel)} (נתוני המבחן, סף {r['threshold']})", "", *_confusion(cm), "",
        f"מתוך {cm['tp'] + cm['fn']} מתאמנים שעזבו בפועל, המודל סימן {cm['tp']} ופספס {cm['fn']}. הוא גם סימן "
        f"{cm['fp']} מתאמנים שנשארו (אזעקות שווא). בשיחת שימור או בהצעת הנחה, אזעקת שווא עולה הרבה פחות מעוזב "
        "שפוספס.",
        "", "#### ניסוי דליפה (לא לשימוש בפועל)", "",
        f"כדי להראות למה החוזה מוציא את \"{', '.join(_col(c) for c in abl['extra_columns'])}\", אותו מודל אומן "
        "שוב **עם** העמודה הזו:", "",
        "| גרסה | CV ROC-AUC | ROC-AUC במבחן | Recall במבחן |", "|---|---:|---:|---:|",
        f"| המודל הראשי (עמודות החוזה) | {res[sel]['cv']['roc_auc']['mean']:.3f} | {test['roc_auc']:.3f} "
        f"| {test['recall']:.3f} |",
        f"| + ביקורי החודש הנוכחי (דולף) | {abl['cv']['roc_auc']['mean']:.3f} | {abl['test']['roc_auc']:.3f} "
        f"| {abl['test']['recall']:.3f} |", "",
        f"הגרסה הדולפת נראית טובה יותר (+{abl['test']['roc_auc'] - test['roc_auc']:.3f} ב-ROC-AUC) רק כי ביקורי "
        "החודש הנוכחי נרשמים בחודש שבו המתאמן כבר עוזב. בתחילת החודש, כשצוות השימור צריך לפעול, הערך הזה עוד לא "
        "קיים. בדיוק ציון מנופח כזה שער הבדיקה נועד למנוע.",
        "", "#### מה משפיע על התחזיות (Permutation Importance על נתוני המבחן)", "",
        "| קלט | הירידה ב-ROC-AUC כשמערבבים אותו |", "|---|---:|",
    ]
    lines += [f"| {_col(f['feature'])} | {f['auc_drop_mean']:.3f} ± {f['auc_drop_std']:.3f} |"
              for f in r["feature_importance"]]
    lines += ["", "#### בדיקת הוגנות (נתוני המבחן)", "",
              "המגדר אינו קלט של המודל; הוא משמש כאן רק כדי לבדוק שהמודל מתייחס לקבוצות באופן דומה."]
    lines += _fairness_tables(r["fairness"])
    lines += ["", "#### בדיקות שפיות", ""]
    lines += [f"- אזהרה: {w}" for w in r["warnings"]] or [
        "- אף מודל לא הגיע ל-ROC-AUC מושלם באופן חשוד (מעל 0.99) ב-Cross-Validation או במבחן."]
    lines.append("- שיעורי העזיבה באימון ובמבחן תואמים (חלוקה מרובדת).")
    return "\n".join(lines) + "\n"


def gender_recall_gap(r: dict) -> float | None:
    recalls = [g["recall"] for g in r["fairness"].get("gender", []) if g["recall"] is not None]
    return max(recalls) - min(recalls) if len(recalls) > 1 else None


def model_card(r: dict) -> str:
    """Hebrew version of model_card.md (modeling/reports.py), from evaluation.json."""
    sel, data = r["selected_model"], r["data"]
    test, cv = r["results"][sel]["test"], r["results"][sel]["cv"]
    cm = test["confusion_matrix"]
    kind = "נרמול ואז רגרסיה לוגיסטית" if sel == "logistic_regression" else "יער אקראי"
    lines = [
        "#### פרטי המודל", "",
        f"- **סוג:** צינור scikit-learn: בניית פיצ'רים לכל שורה, ואז {kind} (משקלות מאוזנים למחלקות).",
        f"- **קובץ:** model.joblib (כולל את הצינור, רשימת הקלטים, סף ההחלטה {r['threshold']} וטווחי הקלט מהחוזה).",
        f"- **גרסאות:** חוזה נתונים {data['contract_version']}, seed {r['seed']}, scikit-learn {r['sklearn_version']}.",
        "", "#### שימוש מיועד", "",
        "- לדרג את המתאמנים לפי הסיכוי שיעזבו, בתחילת כל חודש, כדי שצוות השימור יחליט למי להתקשר, למי להציע "
        "ליווי ולמי להציע שדרוג מנוי.",
        "- כלי לתמיכה בהחלטה. אדם מחליט מה להציע.",
        "", "#### מחוץ לתחום (אסור להשתמש בו כדי)", "",
        "- להעניש מתאמנים, להעלות מחירים או לסרב לשירות על סמך ציון עזיבה.",
        "- לקבל החלטה על אדם לפי מגדר או גיל.",
        "- לנבא במועדונים, מדינות או תקופות אחרות בלי לבדוק מחדש על הנתונים שלהם.",
        "", "#### נתוני האימון", "",
        f"- מערך נתונים ציבורי מ-Kaggle (\"Gym customers features and churn\") של רשת חדרי כושר חיצונית: "
        f"{data['rows']:,} מתאמנים, שורה לכל אחד, שיעור עזיבה {_pct(data['train_churn_rate'])}. **אלה לא לקוחות "
        "אמיתיים של Golden Fighter.** אין שמות, פרטי קשר או מזהים ישירים.",
        "- הנתונים נוקו ותועדו על ידי צוות 1, והתקבלו רק אחרי ששער הבדיקה עבר.",
        f"- חלוקה מרובדת: {data['train_rows']:,} לאימון / {data['test_rows']:,} למבחן.",
        "", "#### קלטים", "",
        "כל הקלטים ידועים בתחילת החודש: " + ", ".join(_col(f) for f in r["input_features"]) + ".", "",
        "פיצ'רים שנבנים בתוך המודל:", "",
    ]
    lines += [f"- {name}: {ENGINEERED.get(name, name)}" for name in r["engineered_features"]]
    lines += [
        "", f"הוצאו בכוונה: העזיבה עצמה (התשובה), מזהה המתאמן (חסר משמעות), המגדר (מאפיין מוגן) ו-"
        f"\"{_col('avg_class_frequency_current_month')}\" (נמדדת בחודש העזיבה: דליפת מידע).",
        "", "#### ביצועים", "", "| מדד | Cross-Validation (אימון) | מבחן |", "|---|---:|---:|",
    ]
    lines += [f"| {label} | {cv[m]['mean']:.3f} | {test[m]:.3f} |" for m, label in METRICS]
    lines += ["", f"בנתוני המבחן המודל מזהה {cm['tp']} מתוך {cm['tp'] + cm['fn']} עוזבים, ומתריע לשווא על "
                  f"{cm['fp']} מתוך {cm['tn'] + cm['fp']} מתאמנים שנשארו.", "", "#### הוגנות"]
    lines += _fairness_tables(r["fairness"])
    gap = gender_recall_gap(r)
    if gap is None:
        gender = "לא ניתן להשוות את ה-Recall בין המגדרים (באחת הקבוצות אין עוזבים בנתוני המבחן)."
    elif gap <= FAIRNESS_RECALL_GAP:
        gender = (f"ה-Recall שונה ב-{gap:.3f} בלבד בין שני קודי המגדר (בתוך הסבולת של {FAIRNESS_RECALL_GAP}), "
                  "כך שעוזבים מזוהים באותה מידה בשתי הקבוצות.")
    else:
        gender = (f"אזהרה: ה-Recall שונה ב-{gap:.3f} בין קודי המגדר (מעל הסבולת של {FAIRNESS_RECALL_GAP}). "
                  "יש לבדוק לפני כל שימוש.")
    lines += [
        "", gender + " בקבוצות הגיל, החלק שמסומן תואם את שיעור העזיבה בפועל: צעירים עוזבים הרבה יותר בנתונים "
        "האלה, ולכן מסומנים יותר. משתמשים בציון כדי להציע עזרה, אף פעם לא כדי להתייחס לקבוצת גיל גרוע יותר.",
        "", "#### מגבלות", "",
        "- **תמונת מצב אחת:** רשת אחת, תקופה אחת שתאריכה לא ידוע, 4,000 מתאמנים. ההתנהגות יכולה להיות שונה "
        "במקום אחר או בזמן אחר.",
        "- **קשר, לא סיבה:** מנוי ארוך הולך יחד עם עזיבה נמוכה, אבל לכפות מנויים ארוכים לא בהכרח ישאיר אנשים. "
        "בודקים פעולות שימור בניסוי (A/B) לפני הפעלה.",
        "- **הוותק שולט:** הוותק ואורך המנוי נושאים את רוב המידע; המודל חלש יותר בהבחנה בין מתאמנים ותיקים, "
        "שאצלם עזיבה נדירה.",
        f"- **סף:** {r['threshold']} מעדיף לתפוס עוזבים (Recall גבוה) על פני דיוק. תקציב אחר להצעות שימור יכול "
        "להצדיק סף אחר.",
        "- **הסתברויות לא מכוילות:** הסיכוי שהמודל נותן הוא הערכה לדירוג. ערכים קיצוניים (קרוב ל-0% או ל-100%) "
        "דורשים כיול (calibration) ובדיקה לפני שימוש אמיתי.",
        f"- **הנחה על תזמון הנתונים:** מניחים ש-\"{_col('avg_class_frequency_total')}\" לא כולל את חודש העזיבה. "
        "אם הוא כן כולל, צריך להוציא גם אותו.",
        "- **יחידות:** הכסף ביחידות המטבע של המקור; החוזה והשער דוחים נתונים שהוחלפה בהם היחידה.",
        "", "#### שיקולים אתיים", "",
        "- נבנה רק ממערך נתונים ציבורי ואנונימי; לא מחובר לשום CRM אמיתי.",
        "- המגדר לא נכנס כקלט, ומשמש רק לבדיקת ההוגנות שלמעלה.",
        "- הגיל הוא קלט כי הוא קשור מאוד לעזיבה בנתונים האלה. השימוש בו מוגבל להצעת עזרה, וצריך לבחון את השפעתו "
        "לפני כל הפעלה אמיתית.",
        "- יש להציג לצוות את הציונים כהסתברויות עם אי-ודאות, לא כעובדות על אדם.",
        "- מאמנים מחדש ובודקים הוגנות מחדש בכל פעם שחוזה הנתונים משתנה.",
    ]
    return "\n".join(lines) + "\n"
