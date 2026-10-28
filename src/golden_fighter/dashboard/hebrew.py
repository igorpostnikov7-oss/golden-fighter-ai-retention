"""Hebrew display text for strings that come from the artifacts, the crews' YAML and other modules.

The artifacts, logs and reports stay in English (they are the project's record). The app shows these Hebrew
versions instead. Every lookup falls back to the original English, so new or changed text is never hidden.
"""
import re

# Units in dataset_contract.json.
UNITS = {"flag": "כן/לא", "category code": "קוד קטגוריה", "currency units (source, unconverted)":
         "יחידות מטבע (כמו במקור, ללא המרה)", "id": "מזהה", "months": "חודשים", "visits per week": "ביקורים בשבוע",
         "years": "שנים"}

# Group names in insight_facts.json / evaluation.json.
GROUPS = {"0-1 months": "0-1 חודשים", "2-3 months": "2-3 חודשים", "4-6 months": "4-6 חודשים",
          "7+ months": "7+ חודשים", "under 25": "מתחת ל-25", "25-29": "25-29", "30-34": "30-34", "35+": "35+",
          "0": "קוד 0", "1": "קוד 1"}
CONTRACT_GROUPS = {"1": "חודש אחד", "6": "6 חודשים", "12": "12 חודשים"}

SEGMENTS = {"1-month contract, joined in the last 0-1 months, no group classes":
            "מנוי לחודש אחד, הצטרפו ב-0-1 החודשים האחרונים ולא משתתפים בשיעורים קבוצתיים"}

LEAKAGE_REASONS = {
    "avg_class_frequency_current_month":
        "העמודה נמדדת באותו חודש שאליו מתייחסת תווית העזיבה, ולכן היא מתעדת בחלקה את העזיבה עצמה (הביקורים של "
        "מתאמנים שעוזבים יורדים בחדות). היא הוצאה מהמודל הראשי ומותרת רק בניסוי דליפה מתועד.",
}

SELECTION_RULES = {
    "highest mean 5-fold CV roc_auc on the training set (tie-break: CV PR-AUC); the test set was not used for "
    "selection": "ה-ROC-AUC הממוצע הגבוה ביותר ב-Cross-Validation של 5 קיפולים על נתוני האימון (בשוויון: PR-AUC). "
                 "נתוני המבחן לא שימשו לבחירה.",
}

# contract/breakers.py SCENARIOS
SCENARIOS = {
    "unit_x100": "עמודת הכסף עברה מדולרים לסנטים (פי 100) - התקרית של Harbor & Vale",
    "rename_column": "שם של עמודה שונה לפי 'מוסכמת שמות חדשה'",
    "drop_column": "עמודה שהמודל צריך נמחקה",
    "invalid_value": "בעמודה contract_period יש ערך שהחוזה לא מתיר (3 חודשים)",
    "out_of_range": "גילים בלתי אפשריים (150 שנה)",
    "missing_values": "ערכים חסרים בעמודה שאסור שתהיה ריקה",
    "non_numeric": "טקסט בעמודה מספרית",
    "target_not_binary": "תווית העזיבה (churn) מכילה ערך שאינו 0 או 1",
    "truncated_data": "הגיעו רק 10% מהשורות",
    "leaky_contract": "מישהו 'שחרר' בחוזה את עמודת החודש הנוכחי הדולפת",
    "contract_missing_field": "מהחוזה נמחקה ההגדרה של משתנה המטרה",
    "corrupt_contract": "הקובץ dataset_contract.json אינו JSON תקין (קובץ קטוע)",
    "missing_artifact": "הקובץ insights.md לא נוצר בכלל",
}

CREWS = {"analyst_crew": "צוות 1 - אנליסט נתונים", "scientist_crew": "צוות 2 - מדען נתונים"}
CREW_KEYS = {"analyst": CREWS["analyst_crew"], "scientist": CREWS["scientist_crew"]}  # run_summary.json keys

# Agent role (crews/*/config/agents.yaml) -> (Hebrew name, Hebrew goal).
AGENTS = {
    "Data Quality Engineer": (
        "מהנדס איכות נתונים",
        "להפוך את קובץ ה-CSV הגולמי של מנויי חדר הכושר למערך נתונים נקי ואמין באמצעות כלי הניקוי הנבדק, "
        "ולדווח בדיוק מה הוסר ולמה."),
    "EDA & Insights Analyst": (
        "אנליסט EDA ותובנות",
        "להפיק את דוח ה-EDA ולהסביר למנהל השימור, בשפה עסקית פשוטה, מי עוזב ואילו צעדים יכולים לשמור על "
        "מתאמנים."),
    "Data Contract Specialist": (
        "מומחה חוזה הנתונים",
        "לפרסם את dataset_contract.json, ההבטחה היחידה שצוות מדעני הנתונים רשאי להסתמך עליה, ולוודא בעזרת "
        "השער הדטרמיניסטי שהנתונים הנקיים עומדים בה."),
    "Feature Engineer": (
        "מהנדס פיצ'רים",
        "לבנות את טבלת הפיצ'רים של המודל רק מעמודות שחוזה הנתונים מתיר ושידועות בזמן החיזוי."),
    "Model Trainer & Evaluator": (
        "מאמן ומעריך מודלים",
        "לאמן ולהשוות את המודלים המועמדים בעזרת כלי האימון הנבדק, ולהסביר בכנות איזה מודל נבחר ולמה, כולל "
        "ניסוי הדליפה."),
    "Model Card Specialist": (
        "מומחה כרטיס המודל",
        "לפרסם את דוח ההערכה ואת כרטיס המודל, ולהוסיף סיכום בשפה פשוטה שמסביר לצוות השימור למה מותר ולמה "
        "אסור להשתמש במודל."),
}

# Flow stages (flow/main.py) and their statuses.
STAGES = {"preflight": "1 - בדיקות מקדימות", "analyst": "2 - צוות 1 (אנליסט נתונים)",
          "validation_gate": "3 - שער הבדיקה", "scientist": "4 - צוות 2 (מדען נתונים)"}
STAGE_STATUS = {"ok": "✅ עבר", "skipped": "⏭️ לא הורץ", "validation_failed": "⛔ נעצר בבדיקות הנתונים",
                "error": "❌ שגיאה"}


def text(mapping: dict, key):
    """The Hebrew version, or the original when there is none."""
    return mapping.get(key, key)


def issue(message: str, labels: dict[str, str]) -> str:
    """One prediction input problem (modeling/predict.py) in Hebrew; unknown messages stay in English."""
    def name(column: str) -> str:
        return labels.get(column, column)

    if message == "'month_to_end_contract' cannot exceed 'contract_period'":
        return "מספר החודשים שנותרו למנוי לא יכול להיות גדול מתקופת המנוי."
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
