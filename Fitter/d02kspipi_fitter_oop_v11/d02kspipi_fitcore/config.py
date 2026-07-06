from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ["0", "false", "no", "off"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ["1", "true", "yes", "on"]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


DEFAULT_MODEL_SWITCHES: dict[str, bool] = {
    "a1": False,     # rho(770)
    "a2": True,      # K*(892)-
    "a3": False,     # LASS-
    "a4": False,     # K2*(1430)-
    "a5": False,     # K*(1410)-
    "a6": False,     # K*(1680)-
    "a7": False,     # K*(892)+
    "a8": False,     # LASS+
    "a9": False,     # K2*(1430)+
    "a10": False,    # K*(1410)+
    "a11": False,    # omega
    "a12": False,    # f2(1270)
    "a13": False,    # rho(1450)
    "a14": False,    # pi pi S-wave / K-matrix
    "const": False,
}


@dataclass
class FitterConfig:
    """User/runtime configuration for the fitter.

    This class is intentionally a typed mirror of the environment-variable
    interface of the monolithic/v10 fitter. The validated physics blocks still
    read their historical names from the execution context, so this class also
    provides `as_legacy_globals()` to keep both representations synchronized.
    """

    script_path: Path
    fit_version: str = "v11"
    base_repo: str = "/user/gr1/lhcb/lfalcao/D0toKs0pipi"
    tfaex_root: str = "/user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples"
    dataset: str = "NO ACC"
    nfit: int = 1_000_000
    ntoys: int = 1_000_000
    nnorm: int = 2_000_000
    nfit_mixing: int = 1_000_000
    nnorm_mixing: int = 7_000_000
    use_mixing_amplitude_cache: bool = True
    mixing_cache_chunk: int = 250_000
    mixing_nll_chunk: int = 250_000
    amp_pars_mode: str = "toys"
    reference_component: str = "auto"
    use_gpu: bool = True
    gpu_id: str = ""
    tf_memory_growth: bool = True
    run_dalitz_fit: bool = True
    run_mixing_fit: bool = True
    create_fitted_sample: bool = True
    run_debug_amplitude_comparison: bool = False
    run_nll_debug: bool = False
    run_zero_mix_validation: bool = False
    use_mixing_tmax_cut: bool = True
    mixing_tmax: float = 8.0
    mixing_float_params: list[str] = field(default_factory=lambda: ["x_mix", "y_mix"])
    apply_dalitz_mask: bool = False
    mask_mode: str = "either"
    mask_m2_low: float = 0.66
    mask_m2_high: float = 0.87
    mixing_parameter_config: dict[str, dict[str, float]] = field(default_factory=dict)
    model_switches: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_MODEL_SWITCHES))
    output_dir: Path = Path("./output_files_v11")
    output_base: str = "./output_files_v11"
    random_seed: int = 12345
    batch: bool = False
    seed_start: int = 1
    seed_end: int = 1
    seed_list_env: str = ""
    data_template: str = ""
    integration_template: str = ""
    data_path_override: str = ""
    integration_path_override: str = ""

    @classmethod
    def from_environment(cls, script_path: Path) -> "FitterConfig":
        base_repo = os.environ.get("BASE_REPO", "/user/gr1/lhcb/lfalcao/D0toKs0pipi")
        tfaex_root = os.environ.get("TFAEX_ROOT", os.path.join(base_repo, "tfa_examples"))

        model_switches = dict(DEFAULT_MODEL_SWITCHES)
        model_switches_json = os.environ.get("MODEL_SWITCHES_JSON", "").strip()
        if model_switches_json:
            try:
                model_switches.update(json.loads(model_switches_json))
            except Exception as exc:
                raise ValueError(f"Could not parse MODEL_SWITCHES_JSON={model_switches_json!r}: {exc}") from exc

        nfit = _env_int("NFIT", 1_000_000)
        mixing_parameter_config = {
            "x_mix": {
                "initial": _env_float("X_MIX_INIT", 0.0),
                "lower": _env_float("X_MIX_LOW", -0.2),
                "upper": _env_float("X_MIX_HIGH", 0.2),
            },
            "y_mix": {
                "initial": _env_float("Y_MIX_INIT", 0.0),
                "lower": _env_float("Y_MIX_LOW", -0.05),
                "upper": _env_float("Y_MIX_HIGH", 0.05),
            },
            "qp_abs": {
                "initial": _env_float("QP_ABS_INIT", 1.0),
                "lower": _env_float("QP_ABS_LOW", 0.20),
                "upper": _env_float("QP_ABS_HIGH", 2.00),
            },
            "qp_phi": {
                "initial": _env_float("QP_PHI_INIT", 0.0),
                "lower": _env_float("QP_PHI_LOW", -0.50),
                "upper": _env_float("QP_PHI_HIGH", 0.50),
            },
        }

        output_dir = Path(os.environ.get("OUTPUT_DIR", "./output_files_v11"))
        output_base = os.environ.get("OUTPUT_BASE", str(output_dir))

        return cls(
            script_path=script_path,
            fit_version=os.environ.get("FIT_VERSION", "v11"),
            base_repo=base_repo,
            tfaex_root=tfaex_root,
            dataset=os.environ.get("DATASET", "NO ACC"),
            nfit=nfit,
            ntoys=_env_int("NTOYS", 1_000_000),
            nnorm=_env_int("NNORM", 2_000_000),
            nfit_mixing=_env_int("NFIT_MIXING", nfit),
            nnorm_mixing=_env_int("NNORM_MIXING", 7_000_000),
            use_mixing_amplitude_cache=_env_bool("USE_MIXING_AMPLITUDE_CACHE", True),
            mixing_cache_chunk=_env_int("MIXING_CACHE_CHUNK", 250_000),
            mixing_nll_chunk=_env_int("MIXING_NLL_CHUNK", 250_000),
            amp_pars_mode=os.environ.get("AMP_PARS_MODE", "toys").strip().lower(),
            reference_component=os.environ.get("REFERENCE_COMPONENT", "auto").strip(),
            use_gpu=_env_bool("USE_GPU", True),
            gpu_id=os.environ.get("GPU_ID", "").strip(),
            tf_memory_growth=_env_bool("TF_MEMORY_GROWTH", True),
            run_dalitz_fit=_env_bool("RUN_DALITZ_FIT", True),
            run_mixing_fit=_env_bool("RUN_MIXING_FIT", True),
            create_fitted_sample=_env_bool("CREATE_FITTED_SAMPLE", True),
            run_debug_amplitude_comparison=_env_flag("RUN_DEBUG_AMPLITUDE_COMPARISON", False),
            run_nll_debug=_env_flag("RUN_NLL_DEBUG", False),
            run_zero_mix_validation=_env_flag("RUN_ZERO_MIX_VALIDATION", False),
            use_mixing_tmax_cut=_env_bool("USE_MIXING_TMAX_CUT", True),
            mixing_tmax=_env_float("MIXING_TMAX", 8.0),
            mixing_float_params=_split_csv(os.environ.get("MIXING_FLOAT_PARAMS", "x_mix,y_mix")),
            apply_dalitz_mask=bool(int(os.environ.get("APPLY_DALITZ_MASK", "0"))),
            mask_mode=os.environ.get("MASK_MODE", "either"),
            mask_m2_low=_env_float("MASK_M2_LOW", 0.66),
            mask_m2_high=_env_float("MASK_M2_HIGH", 0.87),
            mixing_parameter_config=mixing_parameter_config,
            model_switches=model_switches,
            output_dir=output_dir,
            output_base=output_base,
            random_seed=_env_int("RANDOM_SEED", 12345),
            batch=_env_flag("BATCH", False),
            seed_start=_env_int("SEED_START", 1),
            seed_end=_env_int("SEED_END", 1),
            seed_list_env=os.environ.get("SEED_LIST", "").strip(),
            data_template=os.environ.get("DATA_TEMPLATE", "").strip(),
            integration_template=os.environ.get("INTEGRATION_TEMPLATE", "").strip(),
            data_path_override=os.environ.get("DATA_PATH", "").strip(),
            integration_path_override=os.environ.get("INTEGRATION_PATH", "").strip(),
        )

    def as_legacy_globals(self) -> dict[str, Any]:
        """Return names expected by the preserved validated v10 blocks."""
        return {
            "FIT_VERSION": self.fit_version,
            "BASE_REPO": self.base_repo,
            "TFAEX_ROOT": self.tfaex_root,
            "DATASET": self.dataset,
            "NFIT": self.nfit,
            "NTOYS": self.ntoys,
            "NNORM": self.nnorm,
            "NFIT_MIXING": self.nfit_mixing,
            "NNORM_MIXING": self.nnorm_mixing,
            "USE_MIXING_AMPLITUDE_CACHE": self.use_mixing_amplitude_cache,
            "MIXING_CACHE_CHUNK": self.mixing_cache_chunk,
            "MIXING_NLL_CHUNK": self.mixing_nll_chunk,
            "AMP_PARS_MODE": self.amp_pars_mode,
            "REFERENCE_COMPONENT": self.reference_component,
            "USE_GPU": self.use_gpu,
            "GPU_ID": self.gpu_id,
            "TF_MEMORY_GROWTH": self.tf_memory_growth,
            "RUN_DALITZ_FIT": self.run_dalitz_fit,
            "RUN_MIXING_FIT": self.run_mixing_fit,
            "CREATE_FITTED_SAMPLE": self.create_fitted_sample,
            "RUN_DEBUG_AMPLITUDE_COMPARISON": self.run_debug_amplitude_comparison,
            "RUN_NLL_DEBUG": self.run_nll_debug,
            "RUN_ZERO_MIX_VALIDATION": self.run_zero_mix_validation,
            "USE_MIXING_TMAX_CUT": self.use_mixing_tmax_cut,
            "MIXING_TMAX": self.mixing_tmax,
            "MIXING_FLOAT_PARAMS": list(self.mixing_float_params),
            "APPLY_DALITZ_MASK": self.apply_dalitz_mask,
            "MASK_MODE": self.mask_mode,
            "MASK_M2_LOW": self.mask_m2_low,
            "MASK_M2_HIGH": self.mask_m2_high,
            "MIXING_PARAMETER_CONFIG": self.mixing_parameter_config,
            "MODEL_SWITCHES": self.model_switches,
            "OUTPUT_DIR": self.output_dir,
            "OUTPUT_BASE": self.output_base,
            "RANDOM_SEED": self.random_seed,
            "BATCH": self.batch,
            "SEED_START": self.seed_start,
            "SEED_END": self.seed_end,
            "SEED_LIST_ENV": self.seed_list_env,
            "DATA_TEMPLATE": self.data_template,
            "INTEGRATION_TEMPLATE": self.integration_template,
            "DATA_PATH_OVERRIDE": self.data_path_override,
            "INTEGRATION_PATH_OVERRIDE": self.integration_path_override,
        }
