from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from d02kspipi_fitcore.config import FitterConfig
from d02kspipi_fitcore.runtime.context import FitContext


@dataclass
class BatchLauncher:
    """Native batch launcher for v11.

    This is the OOP replacement for the validated step_02_batch block. It keeps
    the same environment-variable interface and output naming convention:

      OUTPUT_DIR = f"{OUTPUT_BASE}_seed{seed}"
      DATA_PATH  = DATA_TEMPLATE.format(seed=seed)

    It exits only in the parent batch process, exactly like the monolithic v10
    batch block. Child processes and non-batch runs simply continue through the
    pipeline.
    """

    context: FitContext

    @property
    def config(self) -> FitterConfig:
        return self.context.config

    @staticmethod
    def configured_default_paths() -> dict[str, dict[str, str]]:
        return {
            "NO ACC": {
                "data": "../../output/TEST_DEBUG_XY_6/Toy_EqualSeed1_v8_run1.npy",
                "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v7_Amp_flat_NoOtherVars_NoACC_NoTA_NoTFit_WithFlatTime_NoComp_20kk/Toy_seed1_v7.npy",
            },
            "selD2Kshh_TwoTracks": {
                "data": "../../output/Equal_1toys_v6_Amp_babar_NoOtherVars_WithACC_selD2Kshh_TwoTracks_NoTA_WithTFit_selD2Kshh_TwoTracks_WithComp_1kk/Toy_seed1_v6.npy",
                "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v6_Amp_flat_NoOtherVars_WithACC_selD2Kshh_TwoTracks_NoTA_WithTFit_selD2Kshh_TwoTracks_WithComp_20kk/Toy_seed1_v6.npy",
            },
            "selTwoTracks": {
                "data": "../../output/Equal_1toys_v6_Amp_babar_NoOtherVars_WithACC_selTwoTracks_NoTA_WithTFit_selTwoTracks_WithComp_1kk/Toy_seed1_v6.npy",
                "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v6_Amp_flat_NoOtherVars_WithACC_selTwoTracks_NoTA_WithTFit_selTwoTracks_WithComp_20kk/Toy_seed1_v6.npy",
            },
            "selD2Kshh": {
                "data": "../../output/Equal_1toys_v6_Amp_babar_NoOtherVars_WithACC_selD2Kshh_NoTA_WithTFit_selD2Kshh_WithComp_1kk/Toy_seed1_v6.npy",
                "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v6_Amp_flat_NoOtherVars_WithACC_selD2Kshh_NoTA_WithTFit_selD2Kshh_WithComp_20kk/Toy_seed1_v6.npy",
            },
        }

    @staticmethod
    def infer_seed_template(path: str) -> str:
        replacements = [
            (r"Toy_seed\d+(_v\d+\.npy)", r"Toy_seed{seed}\1"),
            (r"Toy_EqualSeed\d+(_v\d+_run\d+\.npy)", r"Toy_EqualSeed{seed}\1"),
            (r"seed\d+", r"seed{seed}"),
            (r"Seed\d+", r"Seed{seed}"),
        ]
        out = path
        for pattern, repl in replacements:
            new = re.sub(pattern, repl, out)
            if new != out:
                return new
        return out

    def parse_seed_list(self) -> list[int]:
        cfg = self.config
        if cfg.seed_list_env:
            seeds: list[int] = []
            for item in cfg.seed_list_env.split(","):
                item = item.strip()
                if not item:
                    continue
                if "-" in item:
                    a, b = item.split("-", 1)
                    seeds.extend(range(int(a), int(b) + 1))
                else:
                    seeds.append(int(item))
            return seeds
        return list(range(cfg.seed_start, cfg.seed_end + 1))

    def run_if_requested(self) -> None:
        cfg = self.config

        if not cfg.batch or os.environ.get("_BATCH_CHILD", "0") == "1":
            return

        defaults = self.configured_default_paths()
        if cfg.dataset not in defaults and not cfg.data_path_override:
            raise ValueError(
                f"Dataset '{cfg.dataset}' not recognized and DATA_PATH was not supplied."
            )

        base_data_path = cfg.data_path_override or defaults[cfg.dataset]["data"]
        base_integration_path = cfg.integration_path_override or defaults[cfg.dataset]["integration"]

        data_template = cfg.data_template or self.infer_seed_template(base_data_path)
        integration_template = cfg.integration_template or base_integration_path
        seeds = self.parse_seed_list()

        print("[BATCH] Seeds:", seeds)
        print("[BATCH] DATA_TEMPLATE =", data_template)
        print("[BATCH] INTEGRATION_TEMPLATE =", integration_template)

        for seed in seeds:
            env = os.environ.copy()
            env["_BATCH_CHILD"] = "1"
            env["BATCH"] = "0"
            env["DATA_PATH"] = data_template.format(seed=seed)
            env["INTEGRATION_PATH"] = integration_template.format(seed=seed)
            env["OUTPUT_DIR"] = f"{cfg.output_base}_seed{seed}"

            print()
            print(f"[BATCH] Starting seed {seed}")
            print(f"[BATCH] DATA_PATH  = {env['DATA_PATH']}")
            print(f"[BATCH] OUTPUT_DIR = {env['OUTPUT_DIR']}")

            completed = subprocess.run(
                [sys.executable, str(Path(cfg.script_path).resolve())],
                env=env,
            )
            if completed.returncode != 0:
                print(f"[BATCH] Seed {seed}: FAILED with return code {completed.returncode}")
            else:
                print(f"[BATCH] Seed {seed}: OK")

        sys.exit(0)
