"""Command-line validation gate. Exit code 0 = passed, 1 = failed (the Flow must stop).

    python -m golden_fighter.contract.gate                      # check artifacts/analyst
    python -m golden_fighter.contract.gate --analyst-dir DIR    # check another copy (e.g. a broken one)
"""
import argparse
import logging
from pathlib import Path

from golden_fighter import config
from golden_fighter.contract.validator import run_gate, write_report
from golden_fighter.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Crew 1 artifacts against the dataset contract.")
    parser.add_argument("--analyst-dir", type=Path, default=config.ANALYST_DIR)
    parser.add_argument("--report", type=Path, default=None, help="optional path for a JSON report")
    args = parser.parse_args(argv)

    configure_logging(level=logging.CRITICAL)  # the rendered report below is the CLI output
    result = run_gate(args.analyst_dir)
    print(result.render())
    if args.report:
        write_report(result, args.report)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
