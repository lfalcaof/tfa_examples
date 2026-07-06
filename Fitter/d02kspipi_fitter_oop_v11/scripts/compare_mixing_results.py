#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid seed range: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_params(text: str) -> list[str]:
    params = [p.strip() for p in text.split(",") if p.strip()]
    if not params:
        raise ValueError("At least one parameter must be requested")
    return params


def result_path(base: Path, seed: int) -> Path:
    return Path(f"{base}_seed{seed}") / "json" / "mixing_fit_result.json"


def load_result(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def print_param_table(old_base: Path, new_base: Path, seeds: Iterable[int], params: list[str]) -> int:
    missing = 0
    for param in params:
        print()
        print(f"Parameter: {param}")
        print(f"{'seed':>4} {'old':>16} {'new':>16} {'delta':>14} {'err_old':>16} {'err_new':>16} {'err_delta':>14}")
        print("-" * 104)

        for seed in seeds:
            old_path = result_path(old_base, seed)
            new_path = result_path(new_base, seed)

            if not old_path.exists():
                print(f"{seed:4d} MISSING_OLD_RESULT: {old_path}")
                missing += 1
                continue
            if not new_path.exists():
                print(f"{seed:4d} MISSING_NEW_RESULT: {new_path}")
                missing += 1
                continue

            old = load_result(old_path)
            new = load_result(new_path)

            if param not in old.get("params", {}):
                print(f"{seed:4d} MISSING_OLD_PARAM: {param}")
                missing += 1
                continue
            if param not in new.get("params", {}):
                print(f"{seed:4d} MISSING_NEW_PARAM: {param}")
                missing += 1
                continue

            old_val, old_err = old["params"][param][:2]
            new_val, new_err = new["params"][param][:2]

            print(
                f"{seed:4d} "
                f"{old_val:16.9g} "
                f"{new_val:16.9g} "
                f"{(new_val - old_val):14.6e} "
                f"{old_err:16.9g} "
                f"{new_err:16.9g} "
                f"{(new_err - old_err):14.6e}"
            )
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mixing_fit_result.json files seed by seed.")
    parser.add_argument("--old-base", required=True, type=Path, help="Reference output base without _seedN suffix")
    parser.add_argument("--new-base", required=True, type=Path, help="New output base without _seedN suffix")
    parser.add_argument("--seeds", default="1-10", help="Seed list or ranges, e.g. 1-10 or 1,3,5")
    parser.add_argument("--params", default="y_mix", help="Comma-separated parameter names, e.g. y_mix or x_mix,y_mix")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    params = parse_params(args.params)
    missing = print_param_table(args.old_base, args.new_base, seeds, params)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
