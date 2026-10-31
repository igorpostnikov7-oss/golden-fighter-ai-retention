"""Break a copy of Crew 1's artifacts on purpose, then run the gate on it.

    python scripts/break_data.py unit_x100
    python scripts/break_data.py --list

The original artifacts are never modified. Every run writes its broken copy to a new folder,
artifacts/_broken/<scenario>-<time>-<id>/analyst (gitignored); folders from earlier runs, including ones
older versions wrote, are never reused, changed or deleted.
"""
import argparse
import sys

from golden_fighter import config
from golden_fighter.contract import gate
from golden_fighter.contract.breakers import SCENARIOS, make_broken_copy
from golden_fighter.flow.main import demo_source_dir, new_demo_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", choices=list(SCENARIOS))
    parser.add_argument("--list", action="store_true", help="list the scenarios")
    args = parser.parse_args(argv)

    if args.list or not args.scenario:
        for name, (description, _) in SCENARIOS.items():
            print(f"{name:24s} {description}")
        return 0

    try:
        source = demo_source_dir()
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2
    out_dir = make_broken_copy(source, args.scenario,
                               new_demo_dir(config.ARTIFACTS_DIR / "_broken", args.scenario) / "analyst")
    print(f"Scenario '{args.scenario}': {SCENARIOS[args.scenario][0]}")
    print(f"Broken copy written to {out_dir}\n")
    return gate.main(["--analyst-dir", str(out_dir)])


if __name__ == "__main__":
    sys.exit(main())
