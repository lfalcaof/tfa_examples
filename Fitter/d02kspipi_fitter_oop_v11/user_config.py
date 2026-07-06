
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# =============================================================================
# User-facing fitter configuration
# =============================================================================
#
# This is the file new users should edit most of the time.
# The structural code is organized into explicit runner/model classes under d02kspipi_fitcore/.
#
# Most values map directly to the environment variables already supported by the
# original fitter, so command-line overrides still work:
#
#   NFIT=100000 NNORM=2000000 python run_fitter.py
#
# Values set in the shell take precedence over the defaults below.

USER_CONFIG: dict[str, Any] = {
    # General paths/version
    "FIT_VERSION": "v11",
    "BASE_REPO": "/user/gr1/lhcb/lfalcao/D0toKs0pipi",
    # "TFAEX_ROOT": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples",

    # Dataset selection. You can alternatively set DATA_PATH and INTEGRATION_PATH.
    "DATASET": "NO ACC",
    # "DATA_PATH": "../../output/TEST_DEBUG_XY_6/Toy_EqualSeed1_v8_run1.npy",
    # "INTEGRATION_PATH": "/path/to/integration.npy",

    # Event counts
    "NFIT": 1_000_000,
    "NTOYS": 1_000_000,
    "NNORM": 2_000_000,
    "NFIT_MIXING": 1_000_000,
    "NNORM_MIXING": 7_000_000,

    # Output
    "OUTPUT_DIR": "./output_files_v11_oop",
    "RANDOM_SEED": 12345,

    # GPU/CPU controls
    "USE_GPU": True,
    # "GPU_ID": "0",
    "TF_MEMORY_GROWTH": True,

    # Which fits/diagnostics to run
    "RUN_DALITZ_FIT": True,
    "RUN_MIXING_FIT": True,
    "CREATE_FITTED_SAMPLE": True,
    "RUN_DEBUG_AMPLITUDE_COMPARISON": False,
    "RUN_NLL_DEBUG": False,
    "RUN_ZERO_MIX_VALIDATION": False,

    # Mixing controls
    "USE_MIXING_AMPLITUDE_CACHE": True,
    "MIXING_CACHE_CHUNK": 250_000,
    "MIXING_NLL_CHUNK": 250_000,
    "AMP_PARS_MODE": "toys",             # "toys" or "real_data"
    "REFERENCE_COMPONENT": "auto",
    "MIXING_FLOAT_PARAMS": "x_mix,y_mix",

    # Optional K*(892) diagnostic mask
    "APPLY_DALITZ_MASK": False,
    "MASK_MODE": "either",
    "MASK_M2_LOW": 0.66,
    "MASK_M2_HIGH": 0.87,

    # Optional t cut used by previous v9/v10 tests
    "USE_MIXING_TMAX_CUT": True,
    "MIXING_TMAX": 8.0,

    # Initial values and bounds for mixing parameters
    "X_MIX_INIT": 0.0,
    "X_MIX_LOW": -0.2,
    "X_MIX_HIGH": 0.2,
    "Y_MIX_INIT": 0.0,
    "Y_MIX_LOW": -0.05,
    "Y_MIX_HIGH": 0.05,
    "QP_ABS_INIT": 1.0,
    "QP_ABS_LOW": 0.20,
    "QP_ABS_HIGH": 2.00,
    "QP_PHI_INIT": 0.0,
    "QP_PHI_LOW": -0.50,
    "QP_PHI_HIGH": 0.50,

    # Batch mode
    "BATCH": False,
    "SEED_START": 1,
    "SEED_END": 1,
    # "SEED_LIST": "1,2,5-10",
    # "OUTPUT_BASE": "./output_files_v10_modular",
    # "DATA_TEMPLATE": "../../output/.../Toy_seed{seed}_v8.npy",
    # "INTEGRATION_TEMPLATE": "/path/.../Toy_seed{seed}_v7.npy",
}

# Physics model switches. This is the main model selector.
# True = include component; False = exclude component.
MODEL_SWITCHES: dict[str, bool] = {
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


def _to_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def apply_user_config_to_environment() -> None:
    """Apply USER_CONFIG without overwriting explicit shell environment values."""
    for key, value in USER_CONFIG.items():
        if value is None:
            continue
        os.environ.setdefault(key, _to_env_value(value))

    # MODEL_SWITCHES is serialized to the environment so the validated legacy
    # physics blocks and the v11 FitterConfig see the same model definition.
    os.environ.setdefault("MODEL_SWITCHES_JSON", json.dumps(MODEL_SWITCHES))
