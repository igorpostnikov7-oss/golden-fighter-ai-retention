"""Break a copy of Crew 1's artifacts on purpose, then run the gate on it.

    python scripts/break_data.py unit_x100
    python scripts/break_data.py --list

The original artifacts in artifacts/analyst are never modified; the broken copy goes to
artifacts/_broken/<scenario> (gitignored).
"""
import argparse
import sys

from golden_fighter import config
from golden_fighter.contract import gate
from golden_fighter.contract.breakers import SCENARIOS, make_broken_copy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", choices=list(SCENARIOS))
    parser.add_argument("--list", action="store_true", help="list the scenarios")
    args = parser.parse_args()

    if args.list or not args.scenario:
        for name, (description, _) in SCENARIOS.items():
            print(f"{name:24s} {description}")
        return 0

    out_dir = config.ARTIFACTS_DIR / "_broken" / args.scenario
    make_broken_copy(config.ANALYST_DIR, args.scenario, out_dir)
    print(f"Scenario '{args.scenario}': {SCENARIOS[args.scenario][0]}")
    print(f"Broken copy written to {out_dir}\n")
    return gate.main(["--analyst-dir", str(out_dir)])


if __name__ == "__main__":
    sys.exit(main())
