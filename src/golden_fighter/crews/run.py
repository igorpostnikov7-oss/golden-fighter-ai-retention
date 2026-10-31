"""Run ONE crew with the real Anthropic API (this costs money; the model comes from GF_LLM_MODEL).

  python -m golden_fighter.crews.run analyst     # Crew 1 -> artifacts/analyst/
  python -m golden_fighter.crews.run scientist   # gate, then Crew 2 -> artifacts/scientist/

Connecting the two crews with the gate in between is the Flow's job (M5).
Exit code: 0 = success, 1 = validation failure, 2 = crew/API/setup failure.
"""
import argparse
import logging

from golden_fighter import config
from golden_fighter.crews.runner import CrewRunError
from golden_fighter.errors import DataValidationError
from golden_fighter.llm import LLMSetupError, build_llm
from golden_fighter.logging_setup import configure_logging
from golden_fighter.tools.base import tool_execution_counts

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("crew", choices=["analyst", "scientist"])
    parser.add_argument("--verbose", action="store_true", help="show the agents' reasoning and tool calls")
    parser.add_argument("--no-retries", action="store_true", help="never repeat a failed API request")
    args = parser.parse_args(argv)
    configure_logging(config.LOGS_DIR / f"{args.crew}_crew.log")

    if args.crew == "analyst":
        from golden_fighter.crews.analyst_crew.crew import run_analyst_crew as run
    else:
        from golden_fighter.crews.scientist_crew.crew import run_scientist_crew as run
    try:
        result = run(llm=build_llm(retries=not args.no_retries), verbose=args.verbose)
    except DataValidationError as err:
        log.error("Stopped by data validation:\n%s", err)
        return 1
    except (CrewRunError, LLMSetupError) as err:
        log.error("%s", err)
        return 2
    log.info("Final answer of the last agent:\n%s", result.final_answer)
    for warning in result.warnings:
        log.warning("%s", warning)
    counts = tool_execution_counts(result.tool_calls)
    log.info("Model %s: %d model API calls, %d agent tool executions, %s input + %s output tokens, "
             "estimated cost %s USD, %.0f s", result.model, result.llm_calls, counts["agent_tool_executions"],
             result.usage.get("prompt_tokens"), result.usage.get("completion_tokens"), result.estimated_cost_usd,
             result.duration_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
