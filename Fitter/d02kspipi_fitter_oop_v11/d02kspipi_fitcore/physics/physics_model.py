from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor
from d02kspipi_fitcore.physics.framework import setup_framework_and_constants
from d02kspipi_fitcore.physics.density_model import build_density_model
from d02kspipi_fitcore.physics.complex_projection import (
    build_complex_amplitude_model_and_projection,
)


@dataclass
class PhysicsModel:
    """Owns framework setup, constants, components and amplitude/PDF models.

    The v11 class exposes the model entry points through properties while still
    registering the validated v10 physics definitions through native PhysicsModel entry points.
    """

    context: FitContext
    executor: StepExecutor

    def setup_framework_and_constants(self) -> None:
        setup_framework_and_constants(self.context.namespace)

    def build_density_model(self) -> None:
        build_density_model(self.context.namespace)

    def build_complex_amplitude_model_and_projection(self) -> None:
        # Native implementation of the validated step_08 block.  It defines
        # amplitude_model and, when requested, creates fitted-sample projection
        # plots.  It must run after the Dalitz stage.
        build_complex_amplitude_model_and_projection(self.context)

    @property
    def density_model(self) -> Any:
        return self.context.require("model")

    @property
    def amplitude_model(self) -> Any:
        return self.context.require("amplitude_model")

    @property
    def phase_space(self) -> Any:
        return self.context.require("phsp")

    @property
    def components(self) -> Any:
        return self.context.require("COMPONENTS")
