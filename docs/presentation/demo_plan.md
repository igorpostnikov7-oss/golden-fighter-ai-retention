# Live demo plan (5 minutes)

Everything in the demo is free and offline: no API key, no paid calls, no internet needed.
Hebrew lines are what to say; English is for preparation.

## Before the presentation (10 minutes earlier)
1. Open PowerShell in the project folder and start the app:
   `.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py` → http://localhost:8501
2. Click through all six screens once, so every page is loaded and cached.
3. Open a second PowerShell window in the same folder for the backup demo (step 6).
4. Close other tabs; zoom the browser to 110–125% so the instructor can read the Hebrew.
5. Have `docs/presentation/` screenshots or the slides ready in case the laptop or app fails.

## The demo
| # | Time | Screen | Do | Say (Hebrew) |
|---|---|---|---|---|
| 1 | 0:30 | סקירה כללית | Point at the 5 numbers and the diagram. | "זה המסך הראשי: 4,000 מתאמנים, 26.5% עזבו, והמודל מזהה 90.1% מהעוזבים. בתרשים: צוות 1, השער, צוות 2." |
| 2 | 1:30 | הדגמת שבירת נתונים | Keep "החלפת יחידות: דולרים לסנטים", click **שברו עותק והריצו את ה-Flow**. | "עכשיו אני שובר עותק של הנתונים: מכפיל את עמודת הכסף פי 100. רואים: ה-Flow נעצר בשער, צוות 2 לא התחיל, 0 קריאות API, והנתונים המקוריים לא השתנו. וכאן ההסבר: איזו עמודה, איזה כלל, ושזה נראה כמו החלפת יחידות." Open "הפרטים הטכניים" for 3 seconds: "וזו ההודעה הטכנית המקורית, לביקורת." |
| 3 | 1:00 | ביצועי המודל | Show the bottom-line box, then the comparison table and the red leakage row. | "הרגרסיה הלוגיסטית נבחרה לפי נתוני האימון בלבד. השורה האדומה היא ניסוי הדליפה: ציון גבוה יותר, אבל הוא רואה את התשובה, ולכן הוא לא בשימוש." |
| 4 | 1:00 | חיזוי למתאמן | Example "מתאמן חדש במנוי חודשי" (high risk), then "מתאמן ותיק במנוי שנתי" (low risk). Optionally set "חודשים עד סוף המנוי" above the contract length and click **חשבו סיכון**. | "מתאמן חדש במנוי חודשי: סיכון גבוה, ההמלצה היא להציע עזרה עכשיו. מתאמן ותיק: סיכון נמוך. ואם אכניס ערך שמפר את החוזה, המודל מסרב לחשב." |
| 5 | 0:45 | ריצות ועלויות | Point at the full run row and the paid-vs-free boxes. | "זו הריצה האמיתית עם הסוכנים: 17 קריאות API, כ-0.007 דולר. כל השאר ריצות חינמיות בלי AI. האפליקציה עצמה לא יכולה להפעיל ריצה בתשלום." |

Total about 4:45. If time is short, do steps 1, 2 and 4 only.

## Backup if the app fails (terminal, 1 minute)
```powershell
.\.venv\Scripts\python.exe -m golden_fighter.flow.main --demo-break unit_x100
echo $LASTEXITCODE        # 1 = the Flow refused to train
```
Say: "גם בלי האפליקציה: ה-Flow נעצר עם קוד יציאה 1 וכותב דוח כשל שמסביר כל בעיה."
Then open the newest `artifacts\_broken\runs\<run_id>\failure_report.md`.

## Do not do during the demo
- Do not run the Flow with the agents (`--no-retries` without `--no-llm`): it costs money and needs the key.
- Do not click "הריצו את ה-Flow בלי AI" unless there is time: it takes 10–30 seconds.
- Do not show `.env` or the terminal history.
