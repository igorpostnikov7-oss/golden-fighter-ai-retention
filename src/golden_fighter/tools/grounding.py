"""Number grounding: an agent may only write numbers that the deterministic code produced.

Agent prose (the insights summary, the model-card summary) is checked against a facts JSON
(`insight_facts.json` / `evaluation.json`) before it is saved. A number in the text is grounded when some
number in the facts, or that number x100 (a percentage), rounds to it at the precision the text uses.
"""
import re

NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
# Small whole numbers ("5-fold", "3 agents", "12-month contract") are always allowed.
ALWAYS_ALLOWED = set(range(0, 13))

SECTION_START = "<!-- agent-summary:start -->"
SECTION_END = "<!-- agent-summary:end -->"


def _numbers_in(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimals) for every number in the text."""
    found = []
    for raw in NUMBER_RE.findall(text):
        plain = raw.replace(",", "")
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        found.append((raw, float(plain), decimals))
    return found


def collect_fact_numbers(facts) -> set[float]:
    """Every number in a JSON-like structure, including numbers inside keys and strings ("30-34")."""
    out: set[float] = set()

    def walk(node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out.add(abs(float(node)))
        elif isinstance(node, str):
            out.update(value for _, value, _ in _numbers_in(node))
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(str(key))
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(facts)
    return out


def ungrounded_numbers(text: str, facts) -> list[str]:
    """Numbers in `text` that cannot be traced to `facts` (empty list = fully grounded)."""
    allowed = collect_fact_numbers(facts)
    bad = []
    for raw, value, decimals in _numbers_in(text):
        if decimals == 0 and value in ALWAYS_ALLOWED:
            continue
        if any(round(a, decimals) == value or round(a * 100, decimals) == value for a in allowed):
            continue
        bad.append(raw)
    return bad


def upsert_agent_section(document: str, heading: str, body: str) -> str:
    """Insert (or replace) the agent-written section just before the document's first '## ' heading."""
    section = f"{SECTION_START}\n## {heading}\n\n{body.strip()}\n{SECTION_END}\n\n"
    if SECTION_START in document and SECTION_END in document:
        before, rest = document.split(SECTION_START, 1)
        after = rest.split(SECTION_END, 1)[1].lstrip("\n")
        return before + section + after
    index = document.find("\n## ")
    if index == -1:
        return document.rstrip("\n") + "\n\n" + section
    return document[: index + 1] + section + document[index + 1:]
