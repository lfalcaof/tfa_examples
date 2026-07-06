from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Iterable

from d02kspipi_fitcore.runtime.context import FitContext


@dataclass
class StepExecutor:
    """Executes one validated fitter block inside a FitContext."""

    context: FitContext

    def run_module(self, module: ModuleType) -> None:
        if not hasattr(module, "run"):
            raise TypeError(f"Step module {module!r} has no run(ctx) function")
        module.run(self.context.namespace)

    def run_many(self, modules: Iterable[ModuleType]) -> None:
        for module in modules:
            self.run_module(module)
