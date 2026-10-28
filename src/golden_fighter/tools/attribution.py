"""Source check: every number in an agent's final answer must be traceable to the tool it is credited to.

The summaries saved into insights.md / model_card.md are grounded by their tools (tools/grounding.py). This
check covers each task's final answer, which is passed to the next agent and logged. It runs as a CrewAI
task guardrail, so it is plain Python: a failing answer is sent back once with the exact problems, and if
the rewrite fails too the crew stops.

Rules, per line of the answer:
  - `[source: tool_name]` credits the line's numbers to that tool. The tool must have run successfully in
    this run; naming any other source is an invented source.
  - A line without a tag is credited to the agent's own tools.
  - A line with numbers that names a report in prose ("the cleaning report", "evaluation.json") must also
    be credited to the tool that produced that report, so prose cannot claim a different source.
  - Every number must appear in the credited tools' results (same rounding rules as grounding.py).
"""
import re
from collections.abc import Callable, Iterable

from golden_fighter.tools.base import ToolRunLog
from golden_fighter.tools.grounding import ungrounded_numbers

SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]*)\]", re.IGNORECASE)

# Report names that claim where a number comes from -> the tools whose results contain those numbers.
# Input files ("built from clean_data.csv") are not listed: naming an input is not a claim about a number.
SOURCE_ALIASES: dict[str, set[str]] = {
    "raw profile": {"profile_raw_data"},
    "cleaning report": {"clean_data"},
    "cleaning_report.json": {"clean_data"},
    "eda facts": {"run_eda"},
    "eda report": {"run_eda"},
    "eda_report.html": {"run_eda"},
    "insight facts": {"run_eda"},
    "insight_facts.json": {"run_eda"},
    "evaluation report": {"train_and_compare_models", "write_model_reports"},
    "evaluation_report.md": {"train_and_compare_models", "write_model_reports"},
    "evaluation.json": {"train_and_compare_models", "write_model_reports"},
    "model card": {"train_and_compare_models", "write_model_reports"},
    "model_card.md": {"train_and_compare_models", "write_model_reports"},
}
MAX_PROBLEMS = 6

SOURCE_RULES = (
    "SOURCE RULES (checked by code; an answer that breaks them is sent back): use only numbers from tool "
    "results, either your own tools or tools credited in your context. When a line uses numbers from a tool "
    "that is not yours, end that line with [source: tool_name]. Name a tool, file or report as the source "
    "of a number only if that tool's result contains the number; never guess a source. If a number is not in "
    "a tool result, leave it out completely: do not write it at all, not even in a sentence saying it is "
    "missing, unknown or not reported. Do not count or compute numbers yourself."
)


def _reports_named(line: str) -> dict[str, set[str]]:
    """Reports the line names in words (outside its [source: ...] tags)."""
    prose = SOURCE_TAG_RE.sub("", line).lower()
    return {alias: tools for alias, tools in SOURCE_ALIASES.items() if alias in prose}


def attribution_problems(text: str, results: dict[str, dict], own_tools: Iterable[str]) -> list[str]:
    """Lines whose numbers are uncredited, credited to a source that did not produce them, or mis-named.

    `results`: successful tool results of this run, by tool name. `own_tools`: the answering agent's tools.
    """
    own = [name for name in own_tools if name in results]
    problems: list[str] = []
    for line in (raw.strip() for raw in text.splitlines()):
        if not line:
            continue
        tags = [name.strip().strip("`") for match in SOURCE_TAG_RE.findall(line) for name in match.split(",")]
        invented = [name for name in tags if name not in results]
        if invented:
            problems.append(f"'{line[:120]}' cites {', '.join(invented)}, which did not produce a result in this "
                            f"run. Valid sources: {', '.join(sorted(results)) or 'none'}.")
            continue
        sources = tags or own
        facts = {name: results[name] for name in sources}
        bad = ungrounded_numbers(SOURCE_TAG_RE.sub("", line), facts)
        if not bad:
            mismatched = [alias for alias, tools in _reports_named(line).items()
                          if not tools & set(sources)]
            if mismatched and ungrounded_numbers(line, {}):  # only lines with real numbers make a claim
                problems.append(f"'{line[:120]}' names {', '.join(mismatched)} but its numbers are credited to "
                                f"{', '.join(sources) or 'no tool'}. Name the source that produced them.")
            continue
        credited = ", ".join(sources) if sources else "no tool"
        found_in = sorted(name for name, result in results.items() if not ungrounded_numbers(" ".join(bad), result))
        hint = f" They appear in: {', '.join(found_in)}." if found_in else " No tool produced them: remove them."
        problems.append(f"'{line[:120]}': {', '.join(bad)} not found in the result of {credited}.{hint}")
    if len(problems) > MAX_PROBLEMS:
        problems = problems[:MAX_PROBLEMS] + [f"... and {len(problems) - MAX_PROBLEMS} more lines"]
    return problems


def source_check(run_log: ToolRunLog, own_tools: list[str], task_name: str) -> Callable:
    """A CrewAI task guardrail: (TaskOutput) -> (passed, answer or feedback for the agent)."""

    def guardrail(output) -> tuple[bool, str]:
        if run_log.blocked_by is not None:  # the run is stopping anyway; the runner reports the validation error
            return True, output.raw
        problems = attribution_problems(output.raw, run_log.results, own_tools)
        if not problems:
            return True, output.raw
        run_log.record("source_check", "rejected", 0.0, f"{task_name}: " + " | ".join(problems))
        return False, ("SOURCE CHECK FAILED. Rewrite your final answer so every number is credited to the tool "
                       "whose result contains it:\n- " + "\n- ".join(problems))

    return guardrail
