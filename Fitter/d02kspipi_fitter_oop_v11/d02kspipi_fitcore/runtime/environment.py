from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import math
import os
import re
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor
from d02kspipi_fitcore.runtime.batch import BatchLauncher


@dataclass
class RuntimeEnvironment:
    """Handles early configuration, CUDA visibility and batch dispatch.

    Native v11 implementation of the old step_01_configuration block.
    This keeps the validated legacy symbol names in FitContext.namespace, but
    the source of truth is now FitterConfig instead of re-reading everything
    from os.environ inside a preserved code block.

    This block must run before TensorFlow imports, matching the monolithic
    order. Batch dispatch is now native via BatchLauncher.
    """

    context: FitContext
    executor: StepExecutor

    def configure(self) -> None:
        """Populate legacy symbols and configure process-level early settings."""
        self.context.sync_config_symbols()
        ns = self.context.namespace

        # Provide standard-library symbols that were historically imported by
        # the monolithic configuration block and are still used by preserved
        # blocks such as the batch launcher.
        ns.update(
            {
                "contextlib": contextlib,
                "_datetime": _datetime,
                "json": json,
                "math": math,
                "os": os,
                "re": re,
                "subprocess": subprocess,
                "sys": sys,
                "time": time,
                "array": array,
                "Path": Path,
                "Any": Any,
                "Dict": Dict,
                "Iterable": Iterable,
                "List": List,
                "Optional": Optional,
            }
        )


        # Matplotlib must not try to connect to a display in batch jobs.
        os.environ.setdefault("MPLBACKEND", "Agg")

        # GPU/CPU selection must happen before TensorFlow is imported.
        use_gpu = bool(ns["USE_GPU"])
        gpu_id = str(ns.get("GPU_ID", "")).strip()

        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # Keep the environment aligned with the typed configuration for child
        # processes launched by the preserved batch block. Explicit shell vars
        # are already reflected in FitterConfig.from_environment().
        os.environ.setdefault("BASE_REPO", str(ns["BASE_REPO"]))
        os.environ.setdefault("TFAEX_ROOT", str(ns["TFAEX_ROOT"]))

    def run_batch_if_requested(self) -> None:
        BatchLauncher(self.context).run_if_requested()
