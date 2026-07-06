#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PY_FILES = [
    p for p in ROOT.rglob("*.py")
    if "__pycache__" not in p.parts
]

ok = True
for path in sorted(PY_FILES):
    rel = path.relative_to(ROOT)
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        print(f"OK syntax: {rel}")
    except Exception as exc:
        ok = False
        print(f"FAILED syntax: {rel}: {exc}")

# Also validate the preserved CODE blocks as executable code strings.
try:
    from d02kspipi_fitcore import FitterConfig, FitterPipeline
    from d02kspipi_fitcore.steps import (
        step_01_configuration,
        step_02_batch,
        step_03_outputs,
        step_04_framework_physics,
        step_05_amplitude_density,
        step_06_data_loading,
        step_07_dalitz_fit,
        step_08_complex_projection,
        step_09_mixing_samples,
        step_10_mixing_cache,
        step_11_diagnostics_functions,
        step_12_mixing_fit,
        step_13_optional_diagnostics,
        step_14_final_outputs,
    )

    steps = [
        step_01_configuration,
        step_02_batch,
        step_03_outputs,
        step_04_framework_physics,
        step_05_amplitude_density,
        step_06_data_loading,
        step_07_dalitz_fit,
        step_08_complex_projection,
        step_09_mixing_samples,
        step_10_mixing_cache,
        step_11_diagnostics_functions,
        step_12_mixing_fit,
        step_13_optional_diagnostics,
        step_14_final_outputs,
    ]
    print("OK import: FitterConfig, FitterPipeline")
    for step in steps:
        compile(step.CODE, f"{step.__name__}::{step.DESCRIPTION}", "exec")
        print(f"OK block: {step.__name__.split('.')[-1]} -- {step.DESCRIPTION}")
except Exception as exc:
    ok = False
    print(f"FAILED import/block validation: {exc}")

if not ok:
    raise SystemExit(1)

print("All v11 OOP fitter modules and preserved blocks compile.")
