from __future__ import annotations

from typing import Any

from d02kspipi_fitcore.context_executor import run_code_block
from d02kspipi_fitcore.steps.step_05_amplitude_density import CODE as _DENSITY_CODE


def build_density_model(ns: dict[str, Any]) -> None:
    """Register components, switches and the Dalitz density model.

    Native PhysicsModel entry point replacing the pipeline call to
    steps/step_05_amplitude_density.py.

    It defines the legacy symbols COMPONENTS, ACTIVE_COMPONENT_IDS, SWITCH,
    fit_switches and model(x), which are still consumed by the Dalitz and
    mixing stages.
    """

    run_code_block(
        ns,
        _DENSITY_CODE,
        __file__ + "::build_density_model",
    )
