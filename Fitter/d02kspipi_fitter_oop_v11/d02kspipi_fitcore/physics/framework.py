from __future__ import annotations

from typing import Any

from d02kspipi_fitcore.context_executor import run_code_block
from d02kspipi_fitcore.steps.step_04_framework_physics import CODE as _FRAMEWORK_CODE


def setup_framework_and_constants(ns: dict[str, Any]) -> None:
    """Register framework imports, masses, phase space and line-shape helpers.

    Native PhysicsModel entry point replacing the pipeline call to
    steps/step_04_framework_physics.py.

    This step is deliberately executed in the shared FitContext namespace
    because the downstream model, Dalitz fit and mixing fit still use the
    legacy symbol names: atfi, atfd, phsp, helicity_babar2008,
    LASS_babar2008, K-matrix constants, etc.
    """

    run_code_block(
        ns,
        _FRAMEWORK_CODE,
        __file__ + "::setup_framework_and_constants",
    )
