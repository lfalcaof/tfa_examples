#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from user_config import apply_user_config_to_environment
from d02kspipi_fitcore.config import FitterConfig
from d02kspipi_fitcore.pipeline import FitterPipeline


def main() -> None:
    # User-facing defaults are applied first. Explicit shell variables keep priority.
    apply_user_config_to_environment()

    config = FitterConfig.from_environment(script_path=Path(__file__).resolve())
    pipeline = FitterPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
