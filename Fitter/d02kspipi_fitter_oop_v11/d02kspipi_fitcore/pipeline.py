from __future__ import annotations

from dataclasses import dataclass

from d02kspipi_fitcore.config import FitterConfig
from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor
from d02kspipi_fitcore.runtime.environment import RuntimeEnvironment
from d02kspipi_fitcore.output.output_manager import OutputManager
from d02kspipi_fitcore.physics.physics_model import PhysicsModel
from d02kspipi_fitcore.data.data_loader import DataLoader
from d02kspipi_fitcore.fit.dalitz_fit_runner import DalitzFitRunner
from d02kspipi_fitcore.fit.mixing_fit_runner import MixingFitRunner
from d02kspipi_fitcore.diagnostics.diagnostics_runner import DiagnosticsRunner


@dataclass
class FitterPipeline:
    """High-level orchestration of the v11 OOP fitter.

    This is the single place where the execution order is defined. The order is
    intentionally identical to v10/the monolithic fitter to preserve numerical
    behavior while replacing implicit module-level flow with explicit objects.
    """

    config: FitterConfig

    def __post_init__(self) -> None:
        self.context = FitContext.create(self.config)
        self.executor = StepExecutor(self.context)
        self.environment = RuntimeEnvironment(self.context, self.executor)
        self.outputs = OutputManager(self.context, self.executor)
        self.physics = PhysicsModel(self.context, self.executor)
        self.data = DataLoader(self.context, self.executor)
        self.dalitz = DalitzFitRunner(self.context, self.executor)
        self.mixing = MixingFitRunner(self.context, self.executor)
        self.diagnostics = DiagnosticsRunner(self.context, self.executor)

    def run(self) -> FitContext:
        # Early configuration/batch. May sys.exit(0) for the parent batch process.
        self.environment.configure()
        self.environment.run_batch_if_requested()

        # Output/logging starts only in the child/single-fit process.
        self.outputs.setup()

        # Physics/data/Dalitz sequence.
        self.physics.setup_framework_and_constants()
        self.physics.build_density_model()
        self.data.load()
        self.dalitz.run()

        # Complex amplitude model and optional fitted-sample projections.
        self.physics.build_complex_amplitude_model_and_projection()

        # Mixing sequence.
        self.mixing.prepare_samples()
        # Force legacy chunk symbols before cache/diagnostics/mixing-fit registration.
        self.context.namespace["MIXING_NLL_CHUNK"] = int(self.config.mixing_nll_chunk)
        self.context.namespace["MIXING_CACHE_CHUNK"] = int(self.config.mixing_cache_chunk)
        self.mixing.build_cache_and_nll_tools()
        self.diagnostics.register_functions()
        self.mixing.run_fit()

        # Optional debug blocks and final summaries.
        self.diagnostics.run_optional()
        self.outputs.write_final_outputs()

        return self.context


def run(script_path: str | None = None) -> dict:
    """Compatibility entry point for old v10-style callers.

    New code should instantiate FitterConfig and FitterPipeline directly. This
    helper preserves `from d02kspipi_fitcore.pipeline import run`.
    """
    from pathlib import Path

    config = FitterConfig.from_environment(script_path=Path(script_path or __file__).resolve())
    return FitterPipeline(config).run().namespace
