"""Hebrew display text for strings that come from the artifacts, the crews' YAML, the gate and the Flow.

The artifacts, logs and reports stay in English (they are the project's record and the course deliverables).
The app shows these Hebrew versions instead. Every lookup falls back to the original English, so new or changed
text is never hidden, and translated gate messages always keep the exact original line for auditing.
"""
import re

# Units in dataset_contract.json.
UNITS = {"flag": "כן/לא", "category code": "קוד קטגוריה", "currency units (source, unconverted)":
         "יחידות מטבע (כמו במקור, ללא המרה)", "id": "מזהה", "months": "חודשים", "visits per week": "ביקורים בשבוע",
         "years": "שנים"}

# Every column of the clean data: short Hebrew names for tables, charts and gate messages.
COLUMNS = {
    "member_id": "מזהה מתאמן",
    "gender": "מגדר (קוד)",
    "near_location": "גר/עובד קרוב למועדון",
    "partner": "עובד בחברה שותפה",
    "promo_friends": "הגיע דרך חבר",
    "phone": "מסר טלפון",
    "contract_period": "אורך המנוי (חודשים)",
    "group_visits": "שיעורים קבוצתיים",
    "age": "גיל",
    "avg_additional_charges_total": "הוצאה נוספת ממוצעת",
    "month_to_end_contract": "חודשים עד סוף המנוי",
    "lifetime": "ותק במועדון (חודשים)",
    "avg_class_frequency_total": "ביקורים בשבוע (כל התקופה)",
    "avg_class_frequency_current_month": "ביקורים בשבוע (החודש הנוכחי)",
    "churn": "עזב (churn)",
}

# modeling/features.py ENGINEERED_FEATURES
ENGINEERED = {
    "contract_elapsed_share": "החלק מהמנוי הנוכחי שכבר עבר: 0 = מנוי חדש, קרוב ל-1 = המנוי עומד להסתיים.",
    "is_new_member": "1 אם המתאמן הצטרף לפני 0-1 חודשים (התקופה עם שיעור העזיבה הגבוה ביותר).",
    "social_ties": "מספר הקשרים החברתיים למועדון: חברה שותפה + הגעה דרך חבר + שיעורים קבוצתיים (0-3).",
}

# Group names in insight_facts.json / evaluation.json.
GROUPS = {"0-1 months": "עד חודש", "2-3 months": "2-3 חודשים", "4-6 months": "4-6 חודשים",
          "7+ months": "7 חודשים ומעלה", "under 25": "מתחת ל-25", "25-29": "25-29", "30-34": "30-34", "35+": "35+",
          "0": "קוד 0", "1": "קוד 1"}
CONTRACT_GROUPS = {"1": "חודשי (1)", "6": "חצי שנתי (6)", "12": "שנתי (12)"}

SEGMENTS = {"1-month contract, joined in the last 0-1 months, no group classes":
            "מנוי חודשי, ותק של עד חודש, בלי שיעורים קבוצתיים"}

LEAKAGE_REASONS = {
    "avg_class_frequency_current_month":
        "העמודה נמדדת באותו חודש שבו המתאמן עוזב, ולכן היא \"רואה\" חלק מהתשובה: מי שעוזב מפסיק להגיע, "
        "והמספר צונח. בתחילת החודש, כשצריך לפעול, הערך הזה עוד לא קיים. לכן היא לא נכנסת למודל, ומשמשת רק "
        "בניסוי דליפה מתועד.",
    "month_to_end_contract": "ידועה בזמן החיזוי, אבל קשורה מאוד לאורך המנוי.",
}

SELECTION_RULES = {
    "highest mean 5-fold CV roc_auc on the training set (tie-break: CV PR-AUC); the test set was not used for "
    "selection": "המודל עם ה-ROC-AUC הממוצע הגבוה ביותר ב-Cross-Validation של 5 חלקים על נתוני האימון "
                 "(בשוויון: PR-AUC). נתוני המבחן לא שימשו לבחירה.",
}

# contract/breakers.py SCENARIOS: (short title, what happens)
SCENARIOS = {
    "unit_x100": ("החלפת יחידות: דולרים לסנטים",
                  "עמודת הכסף עברה מדולרים לסנטים, וכל ערך גדל פי 100 (התקרית של Harbor & Vale)."),
    "rename_column": ("שינוי שם של עמודה", "עמודה קיבלה שם חדש לפי \"מוסכמת שמות חדשה\"."),
    "drop_column": ("עמודה נמחקה", "עמודה שהמודל צריך נמחקה מהקובץ."),
    "invalid_value": ("ערך לא מותר", "באורך המנוי יש ערך שהחוזה לא מתיר (3 חודשים)."),
    "out_of_range": ("ערכים מחוץ לטווח", "גילים בלתי אפשריים (150 שנה)."),
    "missing_values": ("ערכים חסרים", "חסרים ערכים בעמודה שאסור שתהיה בה ריקה."),
    "non_numeric": ("טקסט בעמודה מספרית", "הופיע טקסט בעמודה שאמורה להכיל רק מספרים."),
    "target_not_binary": ("תווית עזיבה לא תקינה", "עמודת העזיבה (churn) מכילה ערך שאינו 0 או 1."),
    "truncated_data": ("קובץ חלקי", "הגיעו רק 10% מהשורות."),
    "leaky_contract": ("חוזה שמתיר דליפה", "מישהו \"שחרר\" בחוזה את עמודת ביקורי החודש הנוכחי, שדולפת מידע."),
    "contract_missing_field": ("חוזה חסר", "מהחוזה נמחקה ההגדרה של משתנה המטרה."),
    "corrupt_contract": ("קובץ חוזה פגום", "הקובץ dataset_contract.json אינו JSON תקין (הקובץ נקטע)."),
    "missing_artifact": ("קובץ של צוות 1 חסר", "הקובץ insights.md לא נוצר בכלל."),
}

CREWS = {"analyst_crew": "צוות 1 - אנליסט נתונים", "scientist_crew": "צוות 2 - מדען נתונים"}
CREW_KEYS = {"analyst": CREWS["analyst_crew"], "scientist": CREWS["scientist_crew"]}  # run_summary.json keys

# Agent role (crews/*/config/agents.yaml) -> (Hebrew name, Hebrew goal).
AGENTS = {
    "Data Quality Engineer": (
        "מהנדס איכות נתונים",
        "לנקות את קובץ הנתונים הגולמי בעזרת כלי הניקוי הנבדק, ולדווח בדיוק מה הוסר ולמה."),
    "EDA & Insights Analyst": (
        "אנליסט EDA ותובנות",
        "להפיק את דוח ה-EDA ולהסביר למנהל השימור, בשפה עסקית פשוטה, מי עוזב ומה יכול לשמור על מתאמנים."),
    "Data Contract Specialist": (
        "מומחה חוזה הנתונים",
        "לפרסם את חוזה הנתונים (dataset_contract.json), ההבטחה היחידה שצוות 2 רשאי להסתמך עליה, ולוודא "
        "בעזרת השער שהנתונים הנקיים עומדים בה."),
    "Feature Engineer": (
        "מהנדס פיצ'רים",
        "לבנות את טבלת הפיצ'רים רק מעמודות שהחוזה מתיר ושידועות בזמן החיזוי."),
    "Model Trainer & Evaluator": (
        "מאמן ומעריך מודלים",
        "לאמן ולהשוות את המודלים בעזרת כלי האימון הנבדק, ולהסביר בכנות איזה מודל נבחר ולמה, כולל ניסוי "
        "הדליפה."),
    "Model Card Specialist": (
        "מומחה כרטיס המודל",
        "לפרסם את דוח ההערכה ואת כרטיס המודל, ולסכם בשפה פשוטה למה מותר ולמה אסור להשתמש במודל."),
}

# Flow stages (flow/main.py) and their statuses.
STAGES = {"preflight": "1 - בדיקות מקדימות", "analyst": "2 - צוות 1 (אנליסט נתונים)",
          "validation_gate": "3 - שער הבדיקה", "scientist": "4 - צוות 2 (מדען נתונים)"}
STAGE_STATUS = {"ok": "✅ עבר", "skipped": "⏭️ לא הורץ", "validation_failed": "⛔ נעצר בבדיקות הנתונים",
                "error": "❌ שגיאה"}

# contract/validator.py check names -> what kind of rule was broken.
CHECKS = {
    "schema": "מבנה הקובץ", "row_count": "מספר שורות", "nulls": "ערכים חסרים", "dtype": "סוג נתונים",
    "allowed_values": "ערכים מותרים", "range": "טווח ערכים", "unit_or_distribution": "יחידות מידה / התפלגות",
    "cross_column": "התאמה בין עמודות", "target_rate": "שיעור עזיבה", "identifier": "מזהים כפולים",
    "fingerprint": "טביעת אצבע של הקובץ", "features": "רשימת פיצ'רים", "leakage": "דליפת מידע",
    "forbidden": "עמודה אסורה", "unknown_feature": "עמודה לא מוכרת", "missing_feature": "פיצ'ר חסר",
    "leakage_ablation": "ניסוי דליפה", "contract": "קובץ החוזה", "artifact": "קבצים של צוות 1",
}


def text(mapping: dict, key):
    """The Hebrew version, or the original when there is none."""
    return mapping.get(key, key)


def column(name: str | None) -> str:
    return COLUMNS.get(name, name) if name else ""


def _examples(raw: str) -> str:
    raw = raw.removeprefix("e.g. ")
    return f" (דוגמאות: {raw})" if raw else ""


def _unit(unit: str) -> str:
    return "יחידות מטבע" if unit.startswith("currency") else UNITS.get(unit, unit)


_UNITS_RE = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
_GATE_RULES = [
    (r"(\d+) missing values, but the contract says the column is never null",
     lambda m, c: f"בעמודה \"{c}\" יש {m[1]} ערכים חסרים, אבל לפי החוזה אסור שיחסרו בה ערכים."),
    (r"expected (\w+) but found non-numeric values \((.*)\)",
     lambda m, c: f"בעמודה \"{c}\" נמצאו ערכים שאינם מספרים{_examples(m[2])}, אבל לפי החוזה היא מספרית."),
    (r"expected whole numbers but (\d+) values have decimals \((.*)\)",
     lambda m, c: f"בעמודה \"{c}\" יש {m[1]} ערכים עם שבר עשרוני{_examples(m[2])}, אבל מותרים רק מספרים "
                  "שלמים."),
    (r"(\d+) values outside the allowed set (\[.*?\]) \((.*)\)",
     lambda m, c: f"בעמודה \"{c}\" יש {m[1]} ערכים שאינם ברשימת הערכים המותרים {m[2]}{_examples(m[3])}."),
    (rf"(\d+) values below the minimum (\S+) ({_UNITS_RE}) \((.*)\)",
     lambda m, c: f"בעמודה \"{c}\" יש {m[1]} ערכים מתחת למינימום שבחוזה, {m[2]} {_unit(m[3])}"
                  f"{_examples(m[4])}."),
    (rf"(\d+) values above the maximum (\S+) ({_UNITS_RE}) \((.*)\)",
     lambda m, c: f"בעמודה \"{c}\" יש {m[1]} ערכים מעל למקסימום שבחוזה, {m[2]} {_unit(m[3])}"
                  f"{_examples(m[4])}."),
    (r"mean ([\d,.]+) is outside the contract's expected range \[(\S+), (\S+)\] \(contract mean (\S+) .*?\)"
     r"(?:; values are x([\d,.]+) the contract's, which looks like a UNIT CHANGE)?",
     lambda m, c: f"הממוצע של \"{c}\" הוא {m[1]}, מחוץ לטווח הצפוי בחוזה ({m[2]} עד {m[3]}; הממוצע המקורי: "
                  f"{m[4]})." + (f" הערכים גדולים פי {m[5]} מהצפוי, וזה נראה כמו החלפת יחידות מידה (למשל "
                                 "מדולרים לסנטים)." if m[5] else "")),
    (r"required column is missing from clean_data.csv",
     lambda m, c: f"העמודה \"{c}\" חסרה בקובץ הנתונים הנקיים."),
    (r"column is not in the contract( \(renamed column\?\))?; Crew 2 may only use contract columns",
     lambda m, c: f"העמודה \"{c}\" לא מופיעה בחוזה" + (" (אולי שונה שמה של עמודה?)" if m[1] else "")
                  + ". צוות 2 רשאי להשתמש רק בעמודות שבחוזה."),
    (r"(\d+) rows, but the contract requires at least (\d+) \(contract was written for (\d+)\)",
     lambda m, c: f"בקובץ יש רק {m[1]} שורות, אבל החוזה דורש לפחות {m[2]} (החוזה נכתב עבור {m[3]} שורות)."),
    (r"(\d+) rows break '(.+?)': (.*)",
     lambda m, c: f"{m[1]} שורות מפרות את הכלל בין העמודות '{m[2]}'."),
    (r"churn rate ([\d.]+%) is outside the expected range \[(\S+), (\S+)\]",
     lambda m, c: f"שיעור העזיבה {m[1]} מחוץ לטווח הצפוי ({m[2]} עד {m[3]})."),
    (r"(\d+) duplicate ids; every row must be a different member",
     lambda m, c: f"{m[1]} מזהים כפולים: כל שורה חייבת להיות מתאמן אחר."),
    (r"clean_data.csv is not the exact file the contract was written for.*",
     lambda m, c: "הקובץ אינו בדיוק הקובץ שעבורו נכתב החוזה (טביעת האצבע שונה). זו אזהרה בלבד: הבדיקות על "
                  "התוכן הן שקובעות."),
    (r"no features selected", lambda m, c: "לא נבחרו פיצ'רים למודל."),
    (r"the target is used as a model input \(label leakage\)",
     lambda m, c: "משתנה המטרה (העזיבה עצמה) משמש כקלט למודל: המודל היה רואה את התשובה."),
    (r"the contract forbids this column as a model input",
     lambda m, c: f"החוזה אוסר להשתמש ב-\"{c}\" כקלט למודל."),
    (r"not in the contract; Crew 2 may not assume it exists",
     lambda m, c: f"\"{c}\" לא מופיעה בחוזה, ולכן צוות 2 לא רשאי להניח שהיא קיימת."),
    (r"excluded from the primary model by the contract.*",
     lambda m, c: f"החוזה הוציא את \"{c}\" מהמודל הראשי (סיכון דליפת מידע), ולכן אסור להשתמש בה."),
    (r"leakage-risk column used in an ablation.*",
     lambda m, c: f"\"{c}\" (עמודה עם סיכון דליפה) משמשת בניסוי דליפה בלבד; תוצאותיו לא ישמשו בפועל."),
    (r"required feature is not present in the data",
     lambda m, c: f"הפיצ'ר הנדרש \"{c}\" חסר בנתונים."),
    (r"dataset_contract.json is not valid JSON: .*",
     lambda m, c: "קובץ החוזה dataset_contract.json אינו JSON תקין (כנראה שהקובץ נקטע)."),
    (r"dataset_contract.json breaks a contract rule: Feature '(\w+)' is not available before churn but is not "
     r"excluded from the primary model",
     lambda m, c: f"החוזה מתיר למודל הראשי להשתמש ב-\"{column(m[1])}\", אבל הערך הזה לא ידוע לפני העזיבה "
                  "(דליפת מידע)."),
    (r"dataset_contract.json breaks a contract rule: .*",
     lambda m, c: "קובץ החוזה מפר את אחד מכללי המבנה של חוזה."),
    (r"dataset_contract.json field '(.*?)': .*",
     lambda m, c: f"בקובץ החוזה השדה '{m[1]}' חסר או שגוי."),
    (r"required Crew 1 artifact is missing or empty: (.*)",
     lambda m, c: f"קובץ חובה של צוות 1 חסר או ריק: {m[1]}."),
    (r"could not read (.*?): .*", lambda m, c: f"לא ניתן לקרוא את הקובץ {m[1]}."),
]
_ISSUE_RE = re.compile(r"^\[(?P<check>\w+)\](?: (?P<column>[\w.]+):)? (?P<message>.*)$", re.S)


def gate_issue(issue: str) -> dict:
    """One gate issue ("[check] column: message") explained in Hebrew; `original` keeps the exact line."""
    match = _ISSUE_RE.match(issue.strip())
    if not match:
        return {"check": "", "check_he": "", "column": "", "text": issue, "original": issue, "translated": False}
    check, col, message = match["check"], match["column"], match["message"]
    for pattern, render in _GATE_RULES:
        found = re.fullmatch(pattern, message, re.S)
        if found:
            return {"check": check, "check_he": CHECKS.get(check, check), "column": col or "",
                    "text": render(found, column(col)), "original": issue, "translated": True}
    where = f" בעמודה \"{column(col)}\"" if col else ""
    return {"check": check, "check_he": CHECKS.get(check, check), "column": col or "",
            "text": f"הפרה של כלל מסוג \"{CHECKS.get(check, check)}\"{where}.", "original": issue,
            "translated": False}


_STAGE_DETAILS = [
    (r"raw data readable", lambda m: "קובץ הנתונים הגולמי קיים וקריא"),
    (r"using existing analyst artifacts", lambda m: "משתמשים בקבצים הקיימים של צוות 1"),
    (r"skipped: re-using (\S+) .*", lambda m: f"דולג: משתמשים שוב בקבצים שב-{m[1]} (השער בודק אותם מחדש)"),
    (r"not started: an earlier stage failed", lambda m: "לא התחיל: שלב קודם נכשל"),
    (r"not run: an earlier stage failed", lambda m: "לא הורץ: שלב קודם נכשל"),
    (r"not started: the Flow stopped before modelling", lambda m: "לא התחיל: ה-Flow נעצר לפני שלב המודל"),
    (r"VALIDATION GATE \(.*?\): (PASSED|FAILED) - (\d+) checks, (\d+) errors, (\d+) warnings",
     lambda m: f"שער הבדיקה {'עבר' if m[1] == 'PASSED' else 'נכשל'}: {m[2]} בדיקות, {m[3]} שגיאות, "
               f"{m[4]} אזהרות"),
    (r"the validation gate found (\d+) contract violation\(s\)",
     lambda m: f"שער הבדיקה מצא {m[1]} הפרות של החוזה"),
]


def stage_detail(detail: str) -> str:
    """A Flow stage detail or failure reason in Hebrew (unknown text is returned unchanged)."""
    first = (detail or "").splitlines()[0] if detail else ""
    for pattern, render in _STAGE_DETAILS:
        found = re.fullmatch(pattern, first)
        if found:
            return render(found)
    return first


def issue(message: str, labels: dict[str, str]) -> str:
    """One prediction input problem (modeling/predict.py) in Hebrew; unknown messages stay in English."""
    def name(col: str) -> str:
        return labels.get(col, col)

    if message == "'month_to_end_contract' cannot exceed 'contract_period'":
        return "מספר החודשים שנותרו עד סוף המנוי לא יכול להיות גדול מאורך המנוי."
    patterns = [
        (r"missing input column '(\w+)'", lambda m: f"חסר שדה: {name(m[1])}."),
        (r"'(\w+)' must be a number for every member", lambda m: f"{name(m[1])}: חייב להיות מספר."),
        (r"'(\w+)' must be one of (.+)", lambda m: f"{name(m[1])}: הערך חייב להיות אחד מ- {m[2]}."),
        (r"'(\w+)' below the contract minimum (\S+) (.+)",
         lambda m: f"{name(m[1])}: נמוך מהמינימום שבחוזה ({m[2]} {text(UNITS, m[3])})."),
        (r"'(\w+)' above the contract maximum (\S+) (.+)",
         lambda m: f"{name(m[1])}: גבוה מהמקסימום שבחוזה ({m[2]} {text(UNITS, m[3])})."),
    ]
    for pattern, render in patterns:
        match = re.fullmatch(pattern, message)
        if match:
            return render(match)
    return message
