from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from d02kspipi_fitcore.config import FitterConfig


@dataclass
class FitContext:
    """Encapsulated runtime namespace for the fitter.

    The monolithic fitter used module-level globals. In v11 those names live in
    this explicit object. Preserved legacy blocks are still executed with exec(),
    but their globals are the `namespace` dictionary owned by this FitContext,
    not Python process globals.
    """

    config: FitterConfig
    namespace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, config: FitterConfig) -> "FitContext":
        namespace: dict[str, Any] = {
            "__name__": "__main__",
            "__file__": str(config.script_path),
            "__package__": None,
        }
        namespace.update(config.as_legacy_globals())
        return cls(config=config, namespace=namespace)

    def set(self, name: str, value: Any) -> None:
        self.namespace[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self.namespace.get(name, default)

    def require(self, name: str) -> Any:
        if name not in self.namespace:
            raise KeyError(f"Required fitter symbol {name!r} is not available yet.")
        return self.namespace[name]

    def sync_config_symbols(self) -> None:
        """Re-inject typed config names before executing a major block.

        Some preserved blocks may update names such as RUN_DALITZ_FIT internally
        after checking conditions. This method should be used sparingly: it is
        called before the initial configuration block only. Later blocks should
        see the exact state produced by earlier blocks.
        """
        self.namespace.update(self.config.as_legacy_globals())

    @property
    def output_dir(self):
        return self.get("OUTPUT_DIR", self.config.output_dir)
