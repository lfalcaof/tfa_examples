
#!/usr/bin/env python3
from __future__ import annotations

import importlib

STEPS = [
    "step_01_configuration",
    "step_02_batch",
    "step_03_outputs",
    "step_04_framework_physics",
    "step_05_amplitude_density",
    "step_06_data_loading",
    "step_07_dalitz_fit",
    "step_08_complex_projection",
    "step_09_mixing_samples",
    "step_10_mixing_cache",
    "step_11_diagnostics_functions",
    "step_12_mixing_fit",
    "step_13_optional_diagnostics",
    "step_14_final_outputs",
]

for name in STEPS:
    mod = importlib.import_module(f"d02kspipi_fitcore.steps.{name}")
    compile(mod.CODE, f"{name}::{mod.DESCRIPTION}", "exec")
    print(f"OK: {name} -- {mod.DESCRIPTION}")

print("All modular fitter blocks compile.")
