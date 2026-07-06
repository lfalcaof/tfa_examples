#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executable version 10 of the notebook:
  d02kspipi-fitToys_with_mixing_cells_ACCEPTANCE_COHERENT_v10.json

This script preserves the v9 notebook workflow and adds disk outputs:
  figures/    PNG/PDF figures and rendered tables
  tables/     CSV, HTML, Markdown and LaTeX tables
  json/       JSON summaries, including Dalitz and mixing fit results
  arrays/     NumPy arrays used by the fits
  root/       fit_results.root with one TTree per resonance and a mixing TTree
  logs/       run.log

======================================================================
1. Environment setup on brownie / LCG
======================================================================
Before running this script, load the same LCG view used by the notebook:

  source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_105b x86_64-el9-gcc13-opt

Example:

  cd /user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples/notebooks
  source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_105b x86_64-el9-gcc13-opt
  python d02kspipi_fitToys_with_mixing_ACCEPTANCE_COHERENT_v10.py

======================================================================
2. Main runtime parameters
======================================================================
All parameters in the "Fit configuration" block below can be edited directly
or overridden with environment variables.

Frequent examples:

  DATASET="NO ACC" NFIT=1000000 NNORM=2000000 NNORM_MIXING=7000000 \
  AMP_PARS_MODE="toys" MIXING_FLOAT_PARAMS="x_mix,y_mix" USE_GPU=1 \
  python d02kspipi_fitToys_with_mixing_ACCEPTANCE_COHERENT_v10.py

  DATA_PATH="../../output/.../Toy_seed10_v8.npy" \
  OUTPUT_DIR="output_files_seed10_v10" \
  python d02kspipi_fitToys_with_mixing_ACCEPTANCE_COHERENT_v10.py

======================================================================
3. Batch mode over many toys
======================================================================
One independent Python process is launched per seed. Each seed receives its
own output directory, which is safer for TensorFlow/GPU memory and PyROOT.

Example:

  BATCH=1 SEED_START=1 SEED_END=100 \
  OUTPUT_BASE="output_files_v10" \
  DATA_TEMPLATE="../../output/.../Toy_seed{seed}_v8.npy" \
  python d02kspipi_fitToys_with_mixing_ACCEPTANCE_COHERENT_v10.py

If DATA_TEMPLATE is not supplied, the script tries to infer it from the
configured DATASET path by replacing Toy_seed<number> or Toy_EqualSeed<number>
with Toy_seed{seed} / Toy_EqualSeed{seed}.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import math
import os
import re
import subprocess
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ============================================================
# Fit configuration
# ============================================================

FIT_VERSION = "v10"

# Base local checkout. Change this if the repository is moved.
BASE_REPO = os.environ.get("BASE_REPO", "/user/gr1/lhcb/lfalcao/D0toKs0pipi")
TFAEX_ROOT = os.environ.get("TFAEX_ROOT", os.path.join(BASE_REPO, "tfa_examples"))

# Select input sample unless DATA_PATH / INTEGRATION_PATH are given.
DATASET = os.environ.get("DATASET", "NO ACC")
# Options:
#   "NO ACC"
#   "selD2Kshh_TwoTracks"
#   "selTwoTracks"
#   "selD2Kshh"

# Main event controls.
NFIT = int(os.environ.get("NFIT", "1000000"))
NTOYS = int(os.environ.get("NTOYS", "1000000"))
NNORM = int(os.environ.get("NNORM", "2000000"))

# Mixing-specific controls. By default, use NFIT events for mixing and up to
# 7M normalization events, matching the previous safer-notebook setting.
NFIT_MIXING = int(os.environ.get("NFIT_MIXING", str(NFIT)))
NNORM_MIXING = int(os.environ.get("NNORM_MIXING", "7000000"))

# Mixing-amplitude cache and chunk controls.
# The cache stores A(D0) and A(D0bar) once, on the CPU, and the NLL
# evaluates data and normalization sums in chunks to avoid filling GPU memory.
USE_MIXING_AMPLITUDE_CACHE = os.environ.get("USE_MIXING_AMPLITUDE_CACHE", "1").strip().lower() not in ["0", "false", "no", "off"]
MIXING_CACHE_CHUNK = int(os.environ.get("MIXING_CACHE_CHUNK", "250000"))
MIXING_NLL_CHUNK = int(os.environ.get("MIXING_NLL_CHUNK", "250000"))

# Choose amplitude parameters used inside the mixing likelihood:
#   "real_data" -> use fitted_pars from the Dalitz fit
#   "toys"      -> use source parameters decoded from belle_model.txt
AMP_PARS_MODE = os.environ.get("AMP_PARS_MODE", "toys").strip().lower()
REFERENCE_COMPONENT = os.environ.get("REFERENCE_COMPONENT", "auto").strip()

# GPU/CPU selection. Must be configured before importing TensorFlow.
USE_GPU = os.environ.get("USE_GPU", "1").strip().lower() not in ["0", "false", "no", "off"]
GPU_ID = os.environ.get("GPU_ID", "").strip()
TF_MEMORY_GROWTH = os.environ.get("TF_MEMORY_GROWTH", "1").strip().lower() not in ["0", "false", "no", "off"]

# Fits and optional diagnostics.
RUN_DALITZ_FIT = os.environ.get("RUN_DALITZ_FIT", "1").strip().lower() not in ["0", "false", "no", "off"]
RUN_MIXING_FIT = os.environ.get("RUN_MIXING_FIT", "1").strip().lower() not in ["0", "false", "no", "off"]
CREATE_FITTED_SAMPLE = os.environ.get("CREATE_FITTED_SAMPLE", "1").strip().lower() not in ["0", "false", "no", "off"]
RUN_DEBUG_AMPLITUDE_COMPARISON = os.environ.get("RUN_DEBUG_AMPLITUDE_COMPARISON", "0").strip().lower() in ["1", "true", "yes", "on"]
RUN_NLL_DEBUG = os.environ.get("RUN_NLL_DEBUG", "0").strip().lower() in ["1", "true", "yes", "on"]
RUN_ZERO_MIX_VALIDATION = os.environ.get("RUN_ZERO_MIX_VALIDATION", "0").strip().lower() in ["1", "true", "yes", "on"]

# Mixing sample restriction used in the v9 notebook's reduced test.
USE_MIXING_TMAX_CUT = os.environ.get("USE_MIXING_TMAX_CUT", "1").strip().lower() not in ["0", "false", "no", "off"]
MIXING_TMAX = float(os.environ.get("MIXING_TMAX", "8.0"))

# By default only x and y float, while |q/p| and phi are fixed.
# To float all four:
#   MIXING_FLOAT_PARAMS="x_mix,y_mix,qp_abs,qp_phi" python script.py
MIXING_FLOAT_PARAMS = [
    x.strip()
    for x in os.environ.get("MIXING_FLOAT_PARAMS", "x_mix,y_mix").split(",")
    if x.strip()
]


APPLY_DALITZ_MASK = bool(int(os.environ.get("APPLY_DALITZ_MASK", "0")))

MASK_MODE = os.environ.get("MASK_MODE", "either")
MASK_M2_LOW = float(os.environ.get("MASK_M2_LOW", "0.66"))
MASK_M2_HIGH = float(os.environ.get("MASK_M2_HIGH", "0.87"))


MIXING_PARAMETER_CONFIG = {
    "x_mix":  {
        "initial": float(os.environ.get("X_MIX_INIT",  "0.0000")),
        "lower":   float(os.environ.get("X_MIX_LOW",   "-0.2")),
        "upper":   float(os.environ.get("X_MIX_HIGH",  "0.2")),
    },
    "y_mix":  {
        "initial": float(os.environ.get("Y_MIX_INIT",  "0.0000")),
        "lower":   float(os.environ.get("Y_MIX_LOW",   "-0.05")),
        "upper":   float(os.environ.get("Y_MIX_HIGH",  "0.05")),
    },
    "qp_abs": {
        "initial": float(os.environ.get("QP_ABS_INIT", "1.0000")),
        "lower":   float(os.environ.get("QP_ABS_LOW",  "0.20")),
        "upper":   float(os.environ.get("QP_ABS_HIGH", "2.00")),
    },
    "qp_phi": {
        "initial": float(os.environ.get("QP_PHI_INIT", "0.0000")),
        "lower":   float(os.environ.get("QP_PHI_LOW",  "-0.50")),
        "upper":   float(os.environ.get("QP_PHI_HIGH", "0.50")),
    },
}

# Model switches.
MODEL_SWITCHES = {
    "a1": False,     # rho(770)
    "a2": True,     # K*(892)-
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
    "a14": False,    # K-matrix
    "const": False,
}

# Outputs.
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output_files_v10"))
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", str(OUTPUT_DIR))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "12345"))

# Batch controls.
BATCH = os.environ.get("BATCH", "0").strip().lower() in ["1", "true", "yes", "on"]
SEED_START = int(os.environ.get("SEED_START", "1"))
SEED_END = int(os.environ.get("SEED_END", "1"))
SEED_LIST_ENV = os.environ.get("SEED_LIST", "").strip()
DATA_TEMPLATE = os.environ.get("DATA_TEMPLATE", "").strip()
INTEGRATION_TEMPLATE = os.environ.get("INTEGRATION_TEMPLATE", "").strip()

# Optional path overrides.
DATA_PATH_OVERRIDE = os.environ.get("DATA_PATH", "").strip()
INTEGRATION_PATH_OVERRIDE = os.environ.get("INTEGRATION_PATH", "").strip()

# Matplotlib must not try to connect to a display in batch jobs.
os.environ.setdefault("MPLBACKEND", "Agg")

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif GPU_ID:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


# ============================================================
# Batch launcher
# ============================================================

def _configured_default_paths() -> Dict[str, Dict[str, str]]:
    return {
        "NO ACC": {
            "data": "../../output/TEST_DEBUG_XY_6/Toy_EqualSeed1_v8_run1.npy",
            "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v7_Amp_flat_NoOtherVars_NoACC_NoTA_NoTFit_WithFlatTime_NoComp_20kk/Toy_seed1_v7.npy",
        },
        "selD2Kshh_TwoTracks": {
            "data": "../../output/Equal_1toys_v6_Amp_babar_NoOtherVars_WithACC_selD2Kshh_TwoTracks_NoTA_WithTFit_selD2Kshh_TwoTracks_WithComp_1kk/Toy_seed1_v6.npy",
            "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v6_Amp_flat_NoOtherVars_WithACC_selD2Kshh_TwoTracks_NoTA_WithTFit_selD2Kshh_TwoTracks_WithComp_20kk/Toy_seed1_v6.npy",
        },
        "selTwoTracks": {
            "data": "../../output/Equal_1toys_v6_Amp_babar_NoOtherVars_WithACC_selTwoTracks_NoTA_WithTFit_selTwoTracks_WithComp_1kk/Toy_seed1_v6.npy",
            "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v6_Amp_flat_NoOtherVars_WithACC_selTwoTracks_NoTA_WithTFit_selTwoTracks_WithComp_20kk/Toy_seed1_v6.npy",
        },
        "selD2Kshh": {
            "data": "../../output/Equal_1toys_v6_Amp_babar_NoOtherVars_WithACC_selD2Kshh_NoTA_WithTFit_selD2Kshh_WithComp_1kk/Toy_seed1_v6.npy",
            "integration": "/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/Equal_1toys_v6_Amp_flat_NoOtherVars_WithACC_selD2Kshh_NoTA_WithTFit_selD2Kshh_WithComp_20kk/Toy_seed1_v6.npy",
        },
    }


def _infer_seed_template(path: str) -> str:
    replacements = [
        (r"Toy_seed\d+(_v\d+\.npy)", r"Toy_seed{seed}\1"),
        (r"Toy_EqualSeed\d+(_v\d+_run\d+\.npy)", r"Toy_EqualSeed{seed}\1"),
        (r"seed\d+", r"seed{seed}"),
        (r"Seed\d+", r"Seed{seed}"),
    ]
    out = path
    for pattern, repl in replacements:
        new = re.sub(pattern, repl, out)
        if new != out:
            return new
    return out


def _parse_seed_list() -> List[int]:
    if SEED_LIST_ENV:
        seeds: List[int] = []
        for item in SEED_LIST_ENV.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                a, b = item.split("-", 1)
                seeds.extend(range(int(a), int(b) + 1))
            else:
                seeds.append(int(item))
        return seeds
    return list(range(SEED_START, SEED_END + 1))


def run_batch() -> None:
    defaults = _configured_default_paths()
    if DATASET not in defaults and not DATA_PATH_OVERRIDE:
        raise ValueError(f"Dataset '{DATASET}' not recognized and DATA_PATH was not supplied.")

    base_data_path = DATA_PATH_OVERRIDE or defaults[DATASET]["data"]
    base_integration_path = INTEGRATION_PATH_OVERRIDE or defaults[DATASET]["integration"]

    data_template = DATA_TEMPLATE or _infer_seed_template(base_data_path)
    integration_template = INTEGRATION_TEMPLATE or base_integration_path

    seeds = _parse_seed_list()

    print("[BATCH] Seeds:", seeds)
    print("[BATCH] DATA_TEMPLATE =", data_template)
    print("[BATCH] INTEGRATION_TEMPLATE =", integration_template)

    for seed in seeds:
        env = os.environ.copy()
        env["_BATCH_CHILD"] = "1"
        env["BATCH"] = "0"
        env["DATA_PATH"] = data_template.format(seed=seed)
        env["INTEGRATION_PATH"] = integration_template.format(seed=seed)
        env["OUTPUT_DIR"] = f"{OUTPUT_BASE}_seed{seed}"

        print()
        print(f"[BATCH] Starting seed {seed}")
        print(f"[BATCH] DATA_PATH  = {env['DATA_PATH']}")
        print(f"[BATCH] OUTPUT_DIR = {env['OUTPUT_DIR']}")

        completed = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env)
        if completed.returncode != 0:
            print(f"[BATCH] Seed {seed}: FAILED with return code {completed.returncode}")
        else:
            print(f"[BATCH] Seed {seed}: OK")


if BATCH and os.environ.get("_BATCH_CHILD", "0") != "1":
    run_batch()
    sys.exit(0)


# ============================================================
# Output and logging utilities
# ============================================================

FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
JSON_DIR = OUTPUT_DIR / "json"
ARRAY_DIR = OUTPUT_DIR / "arrays"
ROOT_DIR = OUTPUT_DIR / "root"
LOG_DIR = OUTPUT_DIR / "logs"

for _d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, JSON_DIR, ARRAY_DIR, ROOT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


_log_file = open(LOG_DIR / "run.log", "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if hasattr(x, "numpy"):
            x = x.numpy()
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        return float(x)
    except Exception:
        return default


def jsonable(obj: Any) -> Any:
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        out = {}
        for key in ["fval", "edm", "is_valid", "has_valid_parameters", "has_accurate_covar", "nfcn", "ngrad"]:
            if hasattr(obj, key):
                out[key] = jsonable(getattr(obj, key))
        if out:
            return out
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(obj), handle, indent=2, sort_keys=True)


def save_dataframe_all_formats(df, stem: str) -> None:
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    df.to_html(TABLE_DIR / f"{stem}.html", index=False)
    with open(TABLE_DIR / f"{stem}.md", "w", encoding="utf-8") as handle:
        try:
            handle.write(df.to_markdown(index=False))
        except Exception:
            handle.write(df.to_string(index=False))
    with open(TABLE_DIR / f"{stem}.tex", "w", encoding="utf-8") as handle:
        try:
            handle.write(df.to_latex(index=False, escape=False))
        except Exception:
            handle.write(df.to_string(index=False))


def save_figure(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")


def render_table_figure(df, stem: str, max_rows: int = 40) -> None:
    import matplotlib.pyplot as plt

    display_df = df.copy()
    if len(display_df) > max_rows:
        display_df = display_df.head(max_rows)

    # ------------------------------------------------------------
    # Format floats before rendering the table figure
    # ------------------------------------------------------------
    for col in display_df.columns:
        if np.issubdtype(display_df[col].dtype, np.number):
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.7f}" if np.isfinite(x) else str(x)
            )

    fig_height = max(2.0, 0.35 * (len(display_df) + 1))
    fig_width = max(8.0, 1.7 * len(display_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=list(display_df.columns),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    save_figure(fig, stem)
    plt.close(fig)


def extract_result_scalar(result: Optional[Dict[str, Any]], keys: Iterable[str], default: float = float("nan")) -> float:
    """Extract a scalar diagnostic from a tfo/iminuit-style result object.

    The previous implementation used recursive alias lookup. That failed for
    aliases such as edm -> edm, producing infinite recursion when the key was
    not present in the result dictionary. This version performs only direct
    lookups and one non-recursive alias expansion.
    """
    if not result:
        return default

    requested_keys = list(keys)

    def _lookup_direct(search_keys: Iterable[str], missing: Any = None) -> Any:
        for key in search_keys:
            if isinstance(result, dict) and key in result:
                return as_float(result[key], default)

        if isinstance(result, dict):
            holders = ["fmin", "minuit", "migrad", "minimizer", "result"]
            for holder_key in holders:
                holder = result.get(holder_key)
                if holder is None:
                    continue
                for key in search_keys:
                    if hasattr(holder, key):
                        return as_float(getattr(holder, key), default)
                    if isinstance(holder, dict) and key in holder:
                        return as_float(holder[key], default)

        for key in search_keys:
            if hasattr(result, key):
                return as_float(getattr(result, key), default)

        return missing

    value = _lookup_direct(requested_keys, missing=None)
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        return value

    # Common aliases in tfo.run_minuit / iminuit-like outputs.
    # In the current printed result, the final FCN/NLL is stored as "loglh".
    aliases = {
        "nll": ["loglh", "fval", "fcn", "fun", "value"],
        "loglh": ["nll", "fval", "fcn", "fun", "value"],
        "edm": [],
        "nfcn": ["func_calls", "iterations"],
        "func_calls": ["nfcn", "iterations"],
        "time": ["fit_time", "fit_time_sec"],
    }

    alias_keys = []
    for requested in requested_keys:
        for alias in aliases.get(requested, []):
            if alias not in requested_keys and alias not in alias_keys:
                alias_keys.append(alias)

    value = _lookup_direct(alias_keys, missing=None)
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        return value

    return default


def get_param_value_error(result: Dict[str, Any], name: str) -> tuple[float, float]:
    params = result.get("params", {})
    if name not in params:
        return float("nan"), float("nan")
    vals = params[name]
    value = vals[0] if len(vals) > 0 else float("nan")
    error = vals[1] if len(vals) > 1 else float("nan")
    return as_float(value), as_float(error)


def sanitize_tree_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "component"
    if cleaned[0].isdigit():
        cleaned = "c_" + cleaned
    return cleaned[:80]


def write_root_output(root_path: Path, df_components, df_mixing_summary, dalitz_result, mixing_result) -> None:
    root_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_columns = [
        ("ID", "id", "I"),
        ("Re", "re", "D"),
        ("Re err", "re_err", "D"),
        ("Im", "im", "D"),
        ("Im err", "im_err", "D"),
        ("Amplitude", "amplitude", "D"),
        ("Amplitude err", "amplitude_err", "D"),
        ("Phase (deg)", "phase_deg", "D"),
        ("Phase err (deg)", "phase_err_deg", "D"),
        ("Fit fraction", "fit_fraction", "D"),
        ("Fit fraction (%)", "fit_fraction_percent", "D"),
        ("Fixed", "fixed", "I"),
    ]

    # ------------------------------------------------------------
    # First try the standard PyROOT writer.
    # If PyROOT is unavailable in the GPU environment, fall back to uproot.
    # ------------------------------------------------------------
    try:
        import ROOT

        root_file = ROOT.TFile(str(root_path), "RECREATE")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not create ROOT file: {root_path}")

        for _, row in df_components.iterrows():
            comp_id = int(row["ID"])
            comp_name = str(row["Component"])
            tree_name = f"res_{comp_id:02d}_{sanitize_tree_name(comp_name)}"
            tree = ROOT.TTree(tree_name, comp_name)

            buffers = {}
            for df_col, branch_name, branch_type in numeric_columns:
                if branch_type == "I":
                    buffers[branch_name] = array(
                        "i",
                        [int(bool(row[df_col])) if df_col == "Fixed" else int(row[df_col])]
                    )
                else:
                    buffers[branch_name] = array("d", [as_float(row[df_col])])

                tree.Branch(branch_name, buffers[branch_name], f"{branch_name}/{branch_type}")

            tree.Fill()
            tree.Write()

        mix_tree = ROOT.TTree("mixing", "Mixing fit results and diagnostics")
        mix_buffers = {}

        def add_double_branch(name: str, value: float) -> None:
            mix_buffers[name] = array("d", [as_float(value)])
            mix_tree.Branch(name, mix_buffers[name], f"{name}/D")

        def add_int_branch(name: str, value: int) -> None:
            mix_buffers[name] = array("i", [int(value)])
            mix_tree.Branch(name, mix_buffers[name], f"{name}/I")

        for par in ["x_mix", "y_mix", "qp_abs", "qp_phi"]:
            val, err = get_param_value_error(mixing_result or {}, par)
            add_double_branch(par, val)
            add_double_branch(f"{par}_err", err)
            add_int_branch(f"{par}_fixed", 0 if par in MIXING_FLOAT_PARAMS else 1)

        add_double_branch("nll", extract_result_scalar(mixing_result, ["nll", "fval", "fcn", "fun", "value"]))
        add_double_branch("edm", extract_result_scalar(mixing_result, ["edm"]))
        add_double_branch("fit_time_sec", extract_result_scalar(mixing_result, ["time"]))
        add_int_branch("func_calls", int(extract_result_scalar(mixing_result, ["func_calls", "nfcn"], default=-1)))
        add_int_branch("nfit_mixing", int(NFIT_MIXING))
        add_int_branch("nnorm_mixing", int(NNORM_MIXING))

        add_double_branch("dalitz_nll", extract_result_scalar(dalitz_result, ["nll", "fval", "fcn", "fun", "value"]))
        add_double_branch("dalitz_fit_time_sec", extract_result_scalar(dalitz_result, ["time"]))
        add_int_branch("dalitz_func_calls", int(extract_result_scalar(dalitz_result, ["func_calls", "nfcn"], default=-1)))
        add_int_branch("nfit_dalitz", int(NFIT))
        add_int_branch("nnorm_dalitz", int(NNORM))

        mix_tree.Fill()
        mix_tree.Write()

        root_file.Close()
        print(f"[OUTPUT] ROOT file saved with PyROOT: {root_path}")
        return

    except Exception as exc:
        print(f"[WARNING] PyROOT could not be used: {exc}")
        print("[INFO] Trying uproot fallback for ROOT output.")

    # ------------------------------------------------------------
    # uproot fallback: this works in the clean TensorFlow GPU venv.
    # It writes one TTree per resonance and one TTree named 'mixing'.
    # ------------------------------------------------------------
    try:
        import numpy as _np
        import uproot

        def one_double(value):
            return _np.array([as_float(value)], dtype=_np.float64)

        def one_int(value):
            return _np.array([int(value)], dtype=_np.int32)

        with uproot.recreate(str(root_path)) as root_file:
            for _, row in df_components.iterrows():
                comp_id = int(row["ID"])
                comp_name = str(row["Component"])
                tree_name = f"res_{comp_id:02d}_{sanitize_tree_name(comp_name)}"

                payload = {}

                for df_col, branch_name, branch_type in numeric_columns:
                    if branch_type == "I":
                        if df_col == "Fixed":
                            payload[branch_name] = one_int(int(bool(row[df_col])))
                        else:
                            payload[branch_name] = one_int(row[df_col])
                    else:
                        payload[branch_name] = one_double(row[df_col])

                root_file[tree_name] = payload

            mix_payload = {}

            def add_double(name: str, value: float) -> None:
                mix_payload[name] = one_double(value)

            def add_int(name: str, value: int) -> None:
                mix_payload[name] = one_int(value)

            for par in ["x_mix", "y_mix", "qp_abs", "qp_phi"]:
                val, err = get_param_value_error(mixing_result or {}, par)
                add_double(par, val)
                add_double(f"{par}_err", err)
                add_int(f"{par}_fixed", 0 if par in MIXING_FLOAT_PARAMS else 1)

            add_double("nll", extract_result_scalar(mixing_result, ["nll", "fval", "fcn", "fun", "value"]))
            add_double("edm", extract_result_scalar(mixing_result, ["edm"]))
            add_double("fit_time_sec", extract_result_scalar(mixing_result, ["time"]))
            add_int("func_calls", int(extract_result_scalar(mixing_result, ["func_calls", "nfcn"], default=-1)))
            add_int("nfit_mixing", int(NFIT_MIXING))
            add_int("nnorm_mixing", int(NNORM_MIXING))

            add_double("dalitz_nll", extract_result_scalar(dalitz_result, ["nll", "fval", "fcn", "fun", "value"]))
            add_double("dalitz_fit_time_sec", extract_result_scalar(dalitz_result, ["time"]))
            add_int("dalitz_func_calls", int(extract_result_scalar(dalitz_result, ["func_calls", "nfcn"], default=-1)))
            add_int("nfit_dalitz", int(NFIT))
            add_int("nnorm_dalitz", int(NNORM))

            root_file["mixing"] = mix_payload

        print(f"[OUTPUT] ROOT file saved with uproot: {root_path}")

    except Exception as exc:
        print(f"[WARNING] uproot fallback also failed. ROOT output skipped: {exc}")


print_section("Run configuration")
print("Started at:", _datetime.datetime.now().isoformat(timespec="seconds"))
print("FIT_VERSION:", FIT_VERSION)
print("BASE_REPO:", BASE_REPO)
print("TFAEX_ROOT:", TFAEX_ROOT)
print("DATASET:", DATASET)
print("NFIT:", NFIT)
print("NNORM:", NNORM)
print("NFIT_MIXING:", NFIT_MIXING)
print("NNORM_MIXING:", NNORM_MIXING)
print("USE_MIXING_AMPLITUDE_CACHE:", USE_MIXING_AMPLITUDE_CACHE)
print("MIXING_CACHE_CHUNK:", MIXING_CACHE_CHUNK)
print("MIXING_NLL_CHUNK:", MIXING_NLL_CHUNK)
print("AMP_PARS_MODE:", AMP_PARS_MODE)
print("MIXING_FLOAT_PARAMS:", MIXING_FLOAT_PARAMS)
print("USE_GPU:", USE_GPU)
print("GPU_ID:", GPU_ID if GPU_ID else "<default visible GPUs>")
print("OUTPUT_DIR:", OUTPUT_DIR)




# ============================================================
# Imports and framework setup
# ============================================================

print_section("Import modules")

base = BASE_REPO

# Remove problematic local virtualenv paths.
sys.path = [
    p for p in sys.path
    if "D0toKs0pipi/tfa/lib/python3.9/site-packages" not in p
    and "D0toKs0pipi/tfa_examples/tfa/lib/python3.9/site-packages" not in p
]

# Add source checkouts.
sys.path.insert(0, os.path.join(base, "AmpliTF"))
sys.path.insert(0, os.path.join(base, "TFA2"))

if TFAEX_ROOT not in sys.path:
    sys.path.insert(0, TFAEX_ROOT)

MACROS_DIR = os.path.join(TFAEX_ROOT, "macros")
if MACROS_DIR not in sys.path:
    sys.path.insert(0, MACROS_DIR)

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("NumPy version:", np.__version__)
print("NumPy path:", np.__file__)
print("TensorFlow version:", tf.__version__)
print("TensorFlow path:", tf.__file__)

if USE_GPU:
    gpus = tf.config.list_physical_devices("GPU")
    print("Visible TensorFlow GPUs:", gpus)
    if TF_MEMORY_GROWTH:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"[WARNING] Could not enable memory growth for {gpu}: {exc}")
else:
    print("Running with CUDA_VISIBLE_DEVICES=-1, CPU mode requested.")

# AmpliTF
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

# TFA
import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo
import tfa.rootio as tfr

print("AmpliTF OK")
print("TFA OK")

os.environ["TFAEX_ROOT"] = TFAEX_ROOT
print("belle_model.txt exists =", os.path.exists(os.path.join(TFAEX_ROOT, "params", "belle_model.txt")))
print("models directory exists =", os.path.exists(os.path.join(TFAEX_ROOT, "models")))



# Masses of final state particles
from particle.particle import literals as lp
import numpy as np
import tensorflow as tf

# Dalitz particles
mkz = atfi.const(lp.K_S_0.mass/1000)
mpi = atfi.const(lp.pi_plus.mass/1000)
md = atfi.const(lp.D_0.mass/1000)

# Resonances
mkst = atfi.const(0.89268)  # atfi.const(lp.Kst_892_0.mass/1000)
wkst = atfi.const(0.04749)  # atfi.const(lp.Kst_892_0.width/1000)
mrho = atfi.const(0.77155)
wrho = atfi.const(0.13469)
mk0st1430 = atfi.const(1.440549945739415)
wk0st1430 = atfi.const(0.192611512914605)

mk2st1430 = atfi.const(lp.K_2st_1430_plus.mass/1000)
wk2st1430 = atfi.const(lp.K_2st_1430_plus.width/1000)
mkst1410 = atfi.const(lp.Kst_1410_plus.mass/1000)
wkst1410 = atfi.const(lp.Kst_1410_plus.width/1000)
mkst1680 = atfi.const(lp.Kst_1680_plus.mass/1000)
wkst1680 = atfi.const(lp.Kst_1680_plus.width/1000)
momega = atfi.const(lp.omega_782.mass/1000)
womega = atfi.const(lp.omega_782.width/1000)
mf2_1270 = atfi.const(lp.f_2_1270.mass/1000)
wf2_1270 = atfi.const(lp.f_2_1270.width/1000)
mrho1450 = atfi.const(lp.rho_1450_0.mass/1000)
wrho1450 = atfi.const(lp.rho_1450_0.width / 1000)

# LASS parameters
lass_M = atfi.const(1.440549945739415)
lass_G = atfi.const(0.192611512914605)
lass_a = atfi.const(0.112673863011817)
lass_r = atfi.const(-33.799002116066454)
lass_R = atfi.const(1.0)
lass_phiR = atfi.const(-1.91450383666684)
lass_F = atfi.const(0.955319683174069)
lass_phiF = atfi.const(0.001737032480754)

# Blatt-Weisskopf radii for Breit-Wigner lineshape
rd = atfi.const(5.0)
rr = atfi.const(1.5)

# K-matrix model parameters
meta = atfi.const(lp.eta.mass/1000.)
metap = atfi.const(lp.etap_958.mass/1000.)

g_poles = atfi.const([
    [0.22889, -0.55377, 0, -0.39899, -0.34639],
    [0.94128,  0.55095, 0,  0.39065,  0.31503],
    [0.36856,  0.23888, 0.55639, 0.18340, 0.18681],
    [0.33650,  0.40907, 0.85679, 0.19906, -0.00984],
    [0.18171, -0.17558, -0.79658, -0.00355, 0.22358]
])

m_poles = atfi.const([0.651, 1.2036, 1.55817, 1.21, 1.82206])

b_poles = atfi.complex(
    atfi.const([3.122415682166643, 11.139907856904129, 29.146102368470210, 6.631556203215280, 0.0]),
    atfi.const([7.928823290976309, 4.948420661321371, -0.053588781806890, -8.455370251307063, 0.0])
)

fprod1 = atfi.complex(
    atfi.const([-4.724094278696236, -23.289333360304212, -1.860311896516422, -13.638752211193912, 0.0]),
    atfi.const([-6.511009103363590, -12.215597571354197, -32.982507366353126, -22.339804683783186, 0.0])
)

fij = atfi.const([
    [ 0.23399,  0.15044, -0.20545,  0.32825,  0.35412],
    [ 0.15044,  0,        0,        0,        0      ],
    [-0.20545,  0,        0,        0,        0      ],
    [ 0.32825,  0,        0,        0,        0      ],
    [ 0.35412,  0,        0,        0,        0      ]
])

s0 = atfi.const(-3.92637)
K_matrix_sprod = atfi.const(-0.070000000000000)


phsp = DalitzPhaseSpace(mpi, mkz, mpi, md)


def helicity_babar2008(mAB, mBC, mAC, mA, mB, mC, mD, spin):
    hel = atfi.where(
        spin == 1,
        mAC * mAC - mBC * mBC
        + ((mD * mD - mC * mC) * (mB * mB - mA * mA) / (mAB * mAB)),
        atfi.where(
            spin == 2,
            atfi.pow(
                mBC * mBC - mAC * mAC
                + (mD * mD - mC * mC) * (mA * mA - mB * mB) / (mAB * mAB),
                2
            )
            - 1.0 / 3.0
            * (
                mAB * mAB
                - 2.0 * (mD * mD + mC * mC)
                + atfi.pow(mD * mD - mC * mC, 2) / (mAB * mAB)
            )
            * (
                mAB * mAB
                - 2.0 * (mA * mA + mB * mB)
                + atfi.pow(mA * mA - mB * mB, 2) / (mAB * mAB)
            ),
            1.0
        )
    )
    return hel


def resonant_lass_lineshape_babar2008(m2ab, m0, gamma0, ma, mb):
    m = atfi.sqrt(m2ab)
    q = atfk.two_body_momentum(m, ma, mb)
    q0 = atfk.two_body_momentum(m0, ma, mb)
    width = gamma0 * q / m * m0 / q0
    ampl = (
        atfd.relativistic_breit_wigner(m2ab, m0, width)
        * atfi.cast_complex(m0 * m0 * gamma0 / q0)
    )
    return ampl


def nonresonant_lass_lineshape_babar2008(m2ab, a, r, ma, mb):
    m = atfi.sqrt(m2ab)
    q = atfk.two_body_momentum(m, ma, mb)
    cot_deltab = 1.0 / a / q + 1.0 / 2.0 * r * q
    ampl = atfi.cast_complex(m) / atfi.complex(q * cot_deltab, -q)
    return ampl


def LASS_babar2008(m2, a, r, m0, gamma0, ma, mb, amp_res, phase_res, amp_nr, phase_nr):
    m = atfi.sqrt(m2)
    q = atfk.two_body_momentum(m, ma, mb)
    cot_delta_beta = atfi.const(1.0) / a / q + atfi.const(0.5) * r * q

    nr = nonresonant_lass_lineshape_babar2008(m2, a, r, ma, mb)
    res = resonant_lass_lineshape_babar2008(m2, m0, gamma0, ma, mb)

    lass = (
        atfi.cast_complex(amp_res)
        * atfi.complex(
            atfi.cos(phase_res + 2.0 * phase_nr),
            atfi.sin(phase_res + 2.0 * phase_nr)
        )
        * atfi.complex(q * cot_delta_beta, q)
        / atfi.complex(q * cot_delta_beta, -q)
        * res
    )

    lass += (
        atfi.cast_complex(amp_nr)
        * atfi.complex(atfi.cos(phase_nr), atfi.sin(phase_nr))
        * atfi.cast_complex(atfi.cos(phase_nr) + atfi.sin(phase_nr) * cot_delta_beta)
        * nr
    )

    return lass

# ============================================================
# Control which resonances are included in the model
# True = included | False = excluded
# ============================================================

COMPONENTS = [
    {"id": 1,  "key": "a1",  "label": r"$\rho(770)$"},
    {"id": 2,  "key": "a2",  "label": r"$K^*(892)^-$"},
    {"id": 3,  "key": "a3",  "label": r"$K_0^*(1430)^-$ / LASS$^-$"},
    {"id": 4,  "key": "a4",  "label": r"$K_2^*(1430)^-$"},
    {"id": 5,  "key": "a5",  "label": r"$K^*(1410)^-$"},
    {"id": 6,  "key": "a6",  "label": r"$K^*(1680)^-$"},
    {"id": 7,  "key": "a7",  "label": r"$K^*(892)^+$"},
    {"id": 8,  "key": "a8",  "label": r"$K_0^*(1430)^+$ / LASS$^+$"},
    {"id": 9,  "key": "a9",  "label": r"$K_2^*(1430)^+$"},
    {"id": 10, "key": "a10", "label": r"$K^*(1410)^+$"},
    {"id": 11, "key": "a11", "label": r"$\omega(782)$"},
    {"id": 12, "key": "a12", "label": r"$f_2(1270)$"},
    {"id": 13, "key": "a13", "label": r"$\rho(1450)$"},
    {"id": 14, "key": "a14", "label": r"$\pi\pi$ S-wave / K-matrix"},
]

ACTIVE_COMPONENT_IDS = [
    comp["id"]
    for comp in COMPONENTS
    if MODEL_SWITCHES[comp["key"]]
]

if len(ACTIVE_COMPONENT_IDS) == 0 and not MODEL_SWITCHES["const"]:
    raise RuntimeError("No active component selected in MODEL_SWITCHES.")



SWITCH = dict(MODEL_SWITCHES)

fit_switches = [
    int(SWITCH["a1"]),
    int(SWITCH["a2"]),
    int(SWITCH["a3"]),
    int(SWITCH["a4"]),
    int(SWITCH["a5"]),
    int(SWITCH["a6"]),
    int(SWITCH["a7"]),
    int(SWITCH["a8"]),
    int(SWITCH["a9"]),
    int(SWITCH["a10"]),
    int(SWITCH["a11"]),
    int(SWITCH["a12"]),
    int(SWITCH["a13"]),
    int(SWITCH["a14"]),
    int(SWITCH["const"]),
]


print("Model switches:")
for key, value in SWITCH.items():
    status = "ON" if value else "OFF"
    print(f"{key:6s} : {status}")


def model(x):

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    mab = atfi.sqrt(m2ab)
    mbc = atfi.sqrt(m2bc)
    mac = atfi.sqrt(m2ac)

    # BABAR 2008 helicity convention
    hel_ab_1 = -atfi.cast_complex(
        helicity_babar2008(mab, mbc, mac, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1)
    )
    hel_bc_1 = -atfi.cast_complex(
        helicity_babar2008(mbc, mab, mac, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1)
    )
    hel_ac_1 = -atfi.cast_complex(
        helicity_babar2008(mac, mbc, mab, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1)
    )

    hel_ab_2 = atfi.cast_complex(
        helicity_babar2008(mab, mbc, mac, phsp.ma, phsp.mb, phsp.mc, phsp.md, 2)
    )
    hel_bc_2 = atfi.cast_complex(
        helicity_babar2008(mbc, mab, mac, phsp.mc, phsp.mb, phsp.ma, phsp.md, 2)
    )
    hel_ac_2 = atfi.cast_complex(
        helicity_babar2008(mac, mbc, mab, phsp.ma, phsp.mc, phsp.mb, phsp.md, 2)
    )

    bw1 = atfd.breit_wigner_lineshape(
        m2ac, mrho, wrho,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw2 = atfd.breit_wigner_lineshape(
        m2ab, mkst, wkst,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw4 = atfd.breit_wigner_lineshape(
        m2ab, mk2st1430, wk2st1430,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 2, 2,
        barrier_factor=False
    )

    bw5 = atfd.breit_wigner_lineshape(
        m2ab, mkst1410, wkst1410,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw6 = atfd.breit_wigner_lineshape(
        m2ab, mkst1680, wkst1680,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw7 = atfd.breit_wigner_lineshape(
        m2bc, mkst, wkst,
        phsp.mc, phsp.mb, phsp.ma, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw9 = atfd.breit_wigner_lineshape(
        m2bc, mk2st1430, wk2st1430,
        phsp.mc, phsp.mb, phsp.ma, phsp.md,
        1.5, 5.0, 2, 2,
        barrier_factor=False
    )

    bw10 = atfd.breit_wigner_lineshape(
        m2bc, mkst1410, wkst1410,
        phsp.mc, phsp.mb, phsp.ma, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw11 = atfd.breit_wigner_lineshape(
        m2ac, momega, womega,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw12 = atfd.breit_wigner_lineshape(
        m2ac, mf2_1270, wf2_1270,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 2, 2,
        barrier_factor=False
    )

    bw13 = atfd.breit_wigner_lineshape(
        m2ac, mrho1450, wrho1450,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False,
        md0=mrho1450 + phsp.mb
    )

    # LASS
    lass_n = LASS_babar2008(
        m2ab, lass_a, lass_r, lass_M, lass_G,
        phsp.mb, phsp.ma,
        lass_R, lass_phiR, lass_F, lass_phiF
    )

    lass_p = LASS_babar2008(
        m2bc, lass_a, lass_r, lass_M, lass_G,
        phsp.mb, phsp.mc,
        lass_R, lass_phiR, lass_F, lass_phiF
    )

    # K matrix
    km = atfd.kmatrix_lineshape(
        m2ac,
        m_poles,
        g_poles,
        s0,
        fij,
        b_poles,
        K_matrix_sprod,
        fprod1,
        [[mpi, mpi], [mkz, mkz], [mpi], [meta, meta], [meta, metap]]
    )

    def _model(a1r,
               a1i,
               a2r,
               a2i,
               a3r,
               a3i,
               a4r,
               a4i,
               a5r,
               a5i,
               a6r,
               a6i,
               a7r,
               a7i,
               a8r,
               a8i,
               a9r,
               a9i,
               a10r,
               a10i,
               a11r,
               a11i,
               a12r,
               a12i,
               a13r,
               a13i,
               a14r,
               a14i,
               switches=fit_switches):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)
        a4 = atfi.complex(a4r, a4i)
        a5 = atfi.complex(a5r, a5i)
        a6 = atfi.complex(a6r, a6i)
        a7 = atfi.complex(a7r, a7i)
        a8 = atfi.complex(a8r, a8i)
        a9 = atfi.complex(a9r, a9i)
        a10 = atfi.complex(a10r, a10i)
        a11 = atfi.complex(a11r, a11i)
        a12 = atfi.complex(a12r, a12i)
        a13 = atfi.complex(a13r, a13i)
        a14 = atfi.complex(a14r, a14i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl += a1 * bw1 * hel_ac_1
        if switches[1]:
            ampl += a2 * bw2 * hel_ab_1
        if switches[2]:
            ampl += a3 * lass_n
        if switches[3]:
            ampl += a4 * bw4 * hel_ab_2
        if switches[4]:
            ampl += a5 * bw5 * hel_ab_1
        if switches[5]:
            ampl += a6 * bw6 * hel_ab_1
        if switches[6]:
            ampl += a7 * bw7 * hel_bc_1
        if switches[7]:
            ampl += a8 * lass_p
        if switches[8]:
            ampl += a9 * bw9 * hel_bc_2
        if switches[9]:
            ampl += a10 * bw10 * hel_bc_1
        if switches[10]:
            ampl += a11 * bw11 * hel_ac_1
        if switches[11]:
            ampl += a12 * bw12 * hel_ac_2
        if switches[12]:
            ampl += a13 * bw13 * hel_ac_1
        if switches[13]:
            ampl += a14 * km
        if switches[14]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        return atfd.density(ampl)

    return _model


# ============================================================
# Data loading
# ============================================================

print_section("Load data and integration samples")

config = _configured_default_paths()

if DATASET not in config:
    raise ValueError(f"Dataset '{DATASET}' not recognized. Options: {sorted(config.keys())}")

data_path = DATA_PATH_OVERRIDE or config[DATASET]["data"]
integration_path = INTEGRATION_PATH_OVERRIDE or config[DATASET]["integration"]

print(f"Using dataset: {DATASET}")
print("Data path:", data_path)
print("Integration path:", integration_path)

data_np = np.load(data_path)
print("Original data shape:", data_np.shape)

if data_np.shape[1] < 3:
    raise ValueError(
        "The selected data sample does not contain the decay-time column. "
        "Expected at least 3 columns: [m2KsPiMinus, m2KsPiPlus, t]."
    )

data_np_dalitz = np.ascontiguousarray(data_np[:, :2])
data_np_mixing = np.ascontiguousarray(data_np[:, :3])

print("Dalitz-only data shape:", data_np_dalitz.shape)
print("Dalitz+time data shape:", data_np_mixing.shape)

integration_np = np.load(integration_path)
print("Original integration shape:", integration_np.shape)

if integration_np.shape[1] < 3:
    raise ValueError(
        "The selected integration sample does not contain the decay-time column. "
        "Expected at least 3 columns: [m2KsPiMinus, m2KsPiPlus, t]."
    )

integration_np_dalitz = np.ascontiguousarray(integration_np[:, :2])
integration_np_mixing = np.ascontiguousarray(integration_np[:, :3])

print("Dalitz-only integration shape:", integration_np_dalitz.shape)
print("Dalitz+time integration shape:", integration_np_mixing.shape)


# ============================================================
# Optional Dalitz mask for K*(892) diagnostic
# ============================================================


def dalitz_keep_mask(arr, mode="either", lo=0.66, hi=0.87):
    """
    Build event-level Dalitz mask.

    Column convention:
      arr[:, 0] = m2(KS pi-)
      arr[:, 1] = m2(KS pi+)
      arr[:, 2] = decay time
    """

    m2m = arr[:, 0]
    m2p = arr[:, 1]

    in_m2p = (m2p > lo) & (m2p < hi)
    in_m2m = (m2m > lo) & (m2m < hi)

    if mode == "m2p":
        bad = in_m2p
    elif mode == "m2m":
        bad = in_m2m
    elif mode == "either":
        bad = in_m2p | in_m2m
    elif mode == "both":
        bad = in_m2p & in_m2m
    else:
        raise ValueError("Unknown MASK_MODE: " + str(mode))

    return ~bad


if APPLY_DALITZ_MASK:

    # ------------------------------------------------------------
    # Diagnostic: check which column contains the K*(892)-like band
    # ------------------------------------------------------------
    data_col0_in_band = (
        (data_np_mixing[:, 0] > MASK_M2_LOW)
        & (data_np_mixing[:, 0] < MASK_M2_HIGH)
    )

    data_col1_in_band = (
        (data_np_mixing[:, 1] > MASK_M2_LOW)
        & (data_np_mixing[:, 1] < MASK_M2_HIGH)
    )

    norm_col0_in_band = (
        (integration_np_mixing[:, 0] > MASK_M2_LOW)
        & (integration_np_mixing[:, 0] < MASK_M2_HIGH)
    )

    norm_col1_in_band = (
        (integration_np_mixing[:, 1] > MASK_M2_LOW)
        & (integration_np_mixing[:, 1] < MASK_M2_HIGH)
    )

    print("")
    print("============================================================")
    print("[DALITZ MASK CHECK] Column-band fractions before masking")
    print("============================================================")
    print("[DALITZ MASK CHECK] Expected convention:")
    print("[DALITZ MASK CHECK]   col0 = m2(KS pi-)")
    print("[DALITZ MASK CHECK]   col1 = m2(KS pi+)")
    print("[DALITZ MASK CHECK] band low :", MASK_M2_LOW)
    print("[DALITZ MASK CHECK] band high:", MASK_M2_HIGH)
    print("[DALITZ MASK CHECK] data col0 in band:", np.mean(data_col0_in_band))
    print("[DALITZ MASK CHECK] data col1 in band:", np.mean(data_col1_in_band))
    print("[DALITZ MASK CHECK] norm col0 in band:", np.mean(norm_col0_in_band))
    print("[DALITZ MASK CHECK] norm col1 in band:", np.mean(norm_col1_in_band))
    print("============================================================")
    print("")

    keep_data = dalitz_keep_mask(
        data_np_mixing,
        mode=MASK_MODE,
        lo=MASK_M2_LOW,
        hi=MASK_M2_HIGH,
    )

    keep_integration = dalitz_keep_mask(
        integration_np_mixing,
        mode=MASK_MODE,
        lo=MASK_M2_LOW,
        hi=MASK_M2_HIGH,
    )

    print("")
    print("============================================================")
    print("[DALITZ MASK] Applying K*(892) diagnostic mask")
    print("============================================================")
    print("[DALITZ MASK] mode       :", MASK_MODE)
    print("[DALITZ MASK] m2 low     :", MASK_M2_LOW)
    print("[DALITZ MASK] m2 high    :", MASK_M2_HIGH)
    print("[DALITZ MASK] data before:", data_np_mixing.shape)
    print("[DALITZ MASK] norm before:", integration_np_mixing.shape)

    data_np_dalitz = np.ascontiguousarray(data_np_dalitz[keep_data])
    data_np_mixing = np.ascontiguousarray(data_np_mixing[keep_data])

    integration_np_dalitz = np.ascontiguousarray(integration_np_dalitz[keep_integration])
    integration_np_mixing = np.ascontiguousarray(integration_np_mixing[keep_integration])

    print("[DALITZ MASK] data after :", data_np_mixing.shape)
    print("[DALITZ MASK] norm after :", integration_np_mixing.shape)
    print("[DALITZ MASK] data removed fraction:", 1.0 - np.mean(keep_data))
    print("[DALITZ MASK] norm removed fraction :", 1.0 - np.mean(keep_integration))
    print("============================================================")
    print("")
else:
    print("[DALITZ MASK] Not applied")


# ============================================================
# Build TensorFlow objects after applying the mask
# ============================================================

data_tf = atfi.const(data_np_dalitz)

nnorm_dalitz_effective = min(NNORM, integration_np_dalitz.shape[0])
integration_sample = atfi.const(integration_np_dalitz[:nnorm_dalitz_effective])

print("Final Dalitz-only data sample shape:", data_np_dalitz.shape)
print("Final Dalitz+time data sample shape:", data_np_mixing.shape)
print("Final Dalitz-only integration sample shape:", integration_np_dalitz[:nnorm_dalitz_effective].shape)
print("Final Dalitz+time integration sample shape:", integration_np_mixing.shape)

np.save(ARRAY_DIR / "data_np_dalitz.npy", data_np_dalitz[:NFIT])
np.save(ARRAY_DIR / "data_np_mixing_preview.npy", data_np_mixing[:min(NFIT_MIXING, len(data_np_mixing))])
np.save(ARRAY_DIR / "integration_np_dalitz.npy", integration_np_dalitz[:nnorm_dalitz_effective])

# ============================================================
# Dalitz likelihood and data subset
# ============================================================

print_section("Prepare Dalitz fit")

def nll(data, norm):
    data_model = model(data)
    norm_model = model(norm)

    @atfi.function
    def _nll(pars):
        return atfl.unbinned_nll(data_model(**pars), atfl.integral(norm_model(**pars)))

    return _nll

data_tf_small = atfi.const(data_np_dalitz[:min(NFIT, data_np_dalitz.shape[0])])
print("data_tf_small shape:", data_tf_small.shape)

# Save initial data Dalitz figure.
try:
    fig_data, _ = plot_data(data_tf_small)
    save_figure(fig_data, "data_dalitz_input")
    import matplotlib.pyplot as plt
    plt.close(fig_data)
    print("[OUTPUT] Saved input Dalitz figure.")
except Exception as exc:
    print(f"[WARNING] Could not save input Dalitz figure: {exc}")




# ============================================================
# Dalitz fit and source-parameter preparation
# ============================================================

print_section("Build Dalitz fit parameters")

# ============================================================
# Define free/fixed parameters consistently with MODEL_SWITCHES
# ============================================================

active_components = [
    i for i in range(1, 15)
    if SWITCH[f"a{i}"]
]

if len(active_components) == 0 and not SWITCH["const"]:
    raise RuntimeError("No active amplitude component selected in MODEL_SWITCHES.")

# Choose the first active component as reference.

def choose_reference_component():
    """
    Choose the reference component for the Dalitz fit.

    Rules:
      - If REFERENCE_COMPONENT is given as a number, use that component.
      - If REFERENCE_COMPONENT is a key like 'a2', use that component.
      - If 'auto', use a1 if active; otherwise use the first active component.
    """

    if len(ACTIVE_COMPONENT_IDS) == 0:
        return None

    if REFERENCE_COMPONENT.lower() == "auto":
        if 1 in ACTIVE_COMPONENT_IDS:
            return 1
        return ACTIVE_COMPONENT_IDS[0]

    if REFERENCE_COMPONENT.startswith("a"):
        ref_id = int(REFERENCE_COMPONENT[1:])
    else:
        ref_id = int(REFERENCE_COMPONENT)

    if ref_id not in ACTIVE_COMPONENT_IDS:
        raise RuntimeError(
            f"REFERENCE_COMPONENT={REFERENCE_COMPONENT} was requested, "
            f"but this component is not active. Active components are: {ACTIVE_COMPONENT_IDS}"
        )

    return ref_id


REFERENCE_COMPONENT_ID = choose_reference_component()

print("Active components:", ACTIVE_COMPONENT_IDS)
print("Reference component:", REFERENCE_COMPONENT_ID)

active_components = ACTIVE_COMPONENT_IDS

FREE = {}

for comp in COMPONENTS:
    i = comp["id"]
    key = comp["key"]

    is_active = MODEL_SWITCHES[key]
    is_reference = (i == REFERENCE_COMPONENT_ID)

    # Inactive components are fixed.
    # Reference component is fixed.
    # Other active components are free.
    FREE[f"a{i}r"] = bool(is_active and not is_reference)
    FREE[f"a{i}i"] = bool(is_active and not is_reference)

print("Active components:", active_components)
print("Reference component:", REFERENCE_COMPONENT_ID)

pars = [
    tfo.FitParameter("a1r",  1.0, -100.0, 100.0),
    tfo.FitParameter("a1i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a2r",  1.5, -100.0, 100.0),
    tfo.FitParameter("a2i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a3r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a3i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a4r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a4i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a5r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a5i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a6r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a6i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a7r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a7i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a8r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a8i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a9r",  2.0, -100.0, 100.0),
    tfo.FitParameter("a9i",  0.0, -100.0, 100.0),
    tfo.FitParameter("a10r", 2.0, -100.0, 100.0),
    tfo.FitParameter("a10i", 0.0, -100.0, 100.0),
    tfo.FitParameter("a11r", 2.0, -100.0, 100.0),
    tfo.FitParameter("a11i", 0.0, -100.0, 100.0),
    tfo.FitParameter("a12r", 2.0, -100.0, 100.0),
    tfo.FitParameter("a12i", 0.0, -100.0, 100.0),
    tfo.FitParameter("a13r", 0.0, -100.0, 100.0),
    tfo.FitParameter("a13i", 0.0, -100.0, 100.0),
    tfo.FitParameter("a14r", 1.0e-2, -100.0, 100.0),
    tfo.FitParameter("a14i", 0.0,    -100.0, 100.0),
]

for par in pars:
    if not FREE[par.name]:
        par.fix()

for par in pars:
    status = "FREE" if FREE[par.name] else "FIXED"
    print(f"{par.name:5s} : {status}")


def load_source_amplitude_parameters():
    from models.helpers import decode_model

    belle_model_path = os.path.join(TFAEX_ROOT, "params", "belle_model.txt")
    belle_model = decode_model(belle_model_path)

    print("Loaded belle_model.txt from:")
    print(belle_model_path)

    return {
        "a1r": 1.0,
        "a1i": 0.0,
        "a2r": belle_model["Kstar892minus_realpart"][0],
        "a2i": belle_model["Kstar892minus_imaginarypart"][0],
        "a3r": belle_model["Kstarzero1430minus_realpart"][0],
        "a3i": belle_model["Kstarzero1430minus_imaginarypart"][0],
        "a4r": belle_model["Kstartwo1430minus_realpart"][0],
        "a4i": belle_model["Kstartwo1430minus_imaginarypart"][0],
        "a5r": belle_model["Kstar1410minus_realpart"][0],
        "a5i": belle_model["Kstar1410minus_imaginarypart"][0],
        "a6r": belle_model["Kstar1680minus_realpart"][0],
        "a6i": belle_model["Kstar1680minus_imaginarypart"][0],
        "a7r": belle_model["Kstar892plus_realpart"][0],
        "a7i": belle_model["Kstar892plus_imaginarypart"][0],
        "a8r": belle_model["Kstarzero1430plus_realpart"][0],
        "a8i": belle_model["Kstarzero1430plus_imaginarypart"][0],
        "a9r": belle_model["Kstartwo1430plus_realpart"][0],
        "a9i": belle_model["Kstartwo1430plus_imaginarypart"][0],
        "a10r": belle_model["Kstar1410plus_realpart"][0],
        "a10i": belle_model["Kstar1410plus_imaginarypart"][0],
        "a11r": belle_model["omega_realpart"][0],
        "a11i": belle_model["omega_imaginarypart"][0],
        "a12r": belle_model["ftwo1270_realpart"][0],
        "a12i": belle_model["ftwo1270_imaginarypart"][0],
        "a13r": belle_model["rho1450_realpart"][0],
        "a13i": belle_model["rho1450_imaginarypart"][0],
        # K-matrix coefficient used as a14 in this fitter.
        "a14r": 1.0,
        "a14i": 0.0,
    }


source_values = load_source_amplitude_parameters()
amp_pars_source = {name: atfi.const(float(value)) for name, value in source_values.items()}

def zero_inactive_amplitude_parameters(pars_dict):
    """
    Force all inactive amplitude coefficients to zero.
    """

    out = dict(pars_dict)

    for comp in COMPONENTS:
        i = comp["id"]
        key = comp["key"]

        if not MODEL_SWITCHES[key]:
            out[f"a{i}r"] = atfi.const(0.0)
            out[f"a{i}i"] = atfi.const(0.0)

    return out

amp_pars_source = zero_inactive_amplitude_parameters(amp_pars_source)


result = None
fitted_pars = None


n_free_dalitz_pars = sum(
    1 for par in pars
    if FREE[par.name]
)

print("Number of free Dalitz parameters:", n_free_dalitz_pars)

if RUN_DALITZ_FIT and n_free_dalitz_pars == 0:
    print("[WARNING] RUN_DALITZ_FIT=1 but there are no free Dalitz parameters.")
    print("[WARNING] Skipping Dalitz MINUIT fit and using source parameters instead.")
    RUN_DALITZ_FIT = False

if RUN_DALITZ_FIT:
    print_section("Run Dalitz MINUIT fit")

    result = tfo.run_minuit(nll(data_tf_small, integration_sample), pars)
    print(result)

    fit_time = extract_result_scalar(result, ["time"])
    print(f"Total Dalitz fit time: {fit_time:.2f} s")

    fcalls = extract_result_scalar(result, ["func_calls", "nfcn"], default=float("nan"))
    if fcalls and np.isfinite(fcalls) and fcalls > 0:
        print(f"{fit_time / fcalls:.6f} sec per function call")

    # ------------------------------------------------------------
    # Raw fitted parameters directly from MINUIT
    # ------------------------------------------------------------
    fitted_pars = {
        p: atfi.const(v[0])
        for p, v in result["params"].items()
    }

    # Optional: save the raw MINUIT output before zeroing anything
    save_json(result, JSON_DIR / "dalitz_fit_result_raw.json")

    # ------------------------------------------------------------
    # Force inactive amplitude parameters to zero before using them later
    # ------------------------------------------------------------
    fitted_pars = zero_inactive_amplitude_parameters(fitted_pars)

    # ------------------------------------------------------------
    # Update the result dictionary so the saved JSON is also consistent
    # with the cleaned fitted_pars
    # ------------------------------------------------------------
    result_clean = dict(result)
    result_clean["params"] = dict(result["params"])

    for name, value in fitted_pars.items():
        value_float = float(value.numpy()) if hasattr(value, "numpy") else float(value)

        old_error = float("nan")
        if name in result["params"]:
            old_error = result["params"][name][1]

        result_clean["params"][name] = [value_float, old_error]

    save_json(result_clean, JSON_DIR / "dalitz_fit_result.json")

    # From this point onward, use the cleaned result
    result = result_clean

else:
    print("[INFO] RUN_DALITZ_FIT=0. Using source parameters as Dalitz output parameters.")

    fitted_pars = dict(amp_pars_source)

    # Also zero inactive components when no Dalitz fit is run
    fitted_pars = zero_inactive_amplitude_parameters(fitted_pars)

    result = {
        "params": {
            name: [
                float(value.numpy()) if hasattr(value, "numpy") else float(value),
                float("nan")
            ]
            for name, value in fitted_pars.items()
        }
    }

    save_json(result, JSON_DIR / "dalitz_fit_result.json")
print("Number of fitted_pars      :", len(fitted_pars))
print("Number of amp_pars_source :", len(amp_pars_source))
print("fit_switches =", fit_switches)
print("Kmatrix source coefficient:")
print("  a14r =", source_values["a14r"])
print("  a14i =", source_values["a14i"])

if AMP_PARS_MODE in ["real_data", "real", "fitted", "data"]:
    if not RUN_DALITZ_FIT:
        raise RuntimeError("AMP_PARS_MODE='real_data' requires RUN_DALITZ_FIT=1.")
    amp_pars_for_mixing = dict(fitted_pars)
elif AMP_PARS_MODE in ["toys", "toy", "source", "closure"]:
    amp_pars_for_mixing = dict(amp_pars_source)
else:
    raise ValueError("AMP_PARS_MODE must be 'real_data' or 'toys'.")


print()
print("=" * 80)
print("Amplitude parameters used for mixing")
print("=" * 80)
print("[MIXING AMP] AMP_PARS_MODE =", AMP_PARS_MODE)

if AMP_PARS_MODE in ["real_data", "real", "fitted", "data"]:
    print("[MIXING AMP] Using fitted Dalitz parameters for mixing.")
elif AMP_PARS_MODE in ["toys", "toy", "source", "closure"]:
    print("[MIXING AMP] Using source/generator amplitude parameters for mixing.")

print("[MIXING AMP] Number of parameters =", len(amp_pars_for_mixing))
print("=" * 80)

# For Dalitz output plots/tables, use the Dalitz-fit parameters when available.
amp_pars_for_dalitz_outputs = dict(fitted_pars)

print("AMP_PARS_MODE selected for mixing:", AMP_PARS_MODE)
save_json(
    {
        "amp_pars_mode": AMP_PARS_MODE,
        "source_values": source_values,
        "mixing_float_params": MIXING_FLOAT_PARAMS,
        "mixing_parameter_config": MIXING_PARAMETER_CONFIG,
        "use_mixing_amplitude_cache": USE_MIXING_AMPLITUDE_CACHE,
        "mixing_cache_chunk": MIXING_CACHE_CHUNK,
        "mixing_nll_chunk": MIXING_NLL_CHUNK,
    },
    JSON_DIR / "configuration_summary.json",
)




# ============================================================
# Fit fractions and Dalitz summary tables
# ============================================================

print_section("Build Dalitz output tables")

def fitted_model(x, switches=fit_switches):
    return model(x)(**amp_pars_for_dalitz_outputs, switches=switches)

def mixing_fixed_amplitude_density_model(x, switches=fit_switches):
    return model(x)(**amp_pars_for_mixing, switches=switches)

ff_raw = [
    as_float(v)
    for v in tfo.calculate_fit_fractions(fitted_model, integration_sample)
]

ff = list(ff_raw)

for comp in COMPONENTS:
    i = comp["id"]
    key = comp["key"]

    if not SWITCH[key]:
        ff[i - 1] = 0.0

active_components = [
    comp["id"]
    for comp in COMPONENTS
    if SWITCH[comp["key"]]
]

print("Raw fit fractions:")
print(ff_raw)

print("Fit fractions after removing inactive components:")
print(ff)

if len(active_components) == 1:
    only_id = active_components[0]
    only_ff = as_float(ff_raw[only_id - 1])

    print("")
    print("[FF CHECK] Single active component:")
    print(f"[FF CHECK] component ID = {only_id}")
    print(f"[FF CHECK] raw FF       = {only_ff:.8f}")
    print(f"[FF CHECK] raw FF (%)   = {100.0 * only_ff:.4f}")
    print("")

# Mapping between ai and physics component.
component_map = {
    comp["id"]: comp["label"]
    for comp in COMPONENTS
}

def is_fixed_component(i):
    """
    Component is considered fixed in the summary table if it is the
    chosen Dalitz reference component.
    """

    return i == REFERENCE_COMPONENT_ID


rows = []
params_dict = result["params"]

for comp in COMPONENTS:
    i = comp["id"]
    key = comp["key"]

    if not MODEL_SWITCHES[key]:
        continue

    r_name = f"a{i}r"
    i_name = f"a{i}i"

    re_val = params_dict[r_name][0]
    im_val = params_dict[i_name][0]

    re_err = params_dict[r_name][1] if len(params_dict[r_name]) > 1 else np.nan
    im_err = params_dict[i_name][1] if len(params_dict[i_name]) > 1 else np.nan

    amp = np.sqrt(re_val**2 + im_val**2)
    phase_deg = np.degrees(np.arctan2(im_val, re_val))

    if np.isfinite(re_err) and np.isfinite(im_err) and amp > 0:
        amp_err = np.sqrt((re_val / amp * re_err)**2 + (im_val / amp * im_err)**2)
    else:
        amp_err = np.nan

    denom = re_val**2 + im_val**2
    if np.isfinite(re_err) and np.isfinite(im_err) and denom > 0:
        phase_err_rad = np.sqrt(
            ((-im_val / denom) * re_err)**2 +
            (( re_val / denom) * im_err)**2
        )
        phase_err_deg = np.degrees(phase_err_rad)
    else:
        phase_err_deg = np.nan

    ff_val = ff[i - 1] if (i - 1) < len(ff) else np.nan
    ff_pct = 100.0 * ff_val if pd.notnull(ff_val) else np.nan

    rows.append({
        "ID": i,
        "Component": component_map.get(i, f"a{i}"),
        "Re": re_val,
        "Re err": re_err,
        "Im": im_val,
        "Im err": im_err,
        "Amplitude": amp,
        "Amplitude err": amp_err,
        "Phase (deg)": phase_deg,
        "Phase err (deg)": phase_err_deg,
        "Fit fraction": ff_val,
        "Fit fraction (%)": ff_pct,
        "Fixed": is_fixed_component(i)
    })

df_components = pd.DataFrame(rows)
print(df_components)

summary_rows = []
for _, row in df_components.iterrows():
    if bool(row["Fixed"]):
        amp_txt = f'{row["Amplitude"]:.3f} (fixed)'
        phase_txt = f'{row["Phase (deg)"]:.1f} (fixed)'
    else:
        amp_txt = f'{row["Amplitude"]:.3f} ± {row["Amplitude err"]:.3f}'
        phase_txt = f'{row["Phase (deg)"]:.1f} ± {row["Phase err (deg)"]:.1f}'

    ff_value = row["Fit fraction (%)"]
    if ff_value < 0.01:
        ff_txt = "<0.01"
    else:
        ff_txt = f'{ff_value:.2f}'

    summary_rows.append({
        "Resonance": row["Component"],
        "Amplitude": amp_txt,
        "Phase (deg)": phase_txt,
        "Fit fraction (%)": ff_txt
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary)

save_dataframe_all_formats(df_components, "dalitz_components")
save_dataframe_all_formats(df_summary, "dalitz_summary")
render_table_figure(df_summary, "table_dalitz_summary")
print("[OUTPUT] Dalitz tables saved.")



# ============================================================
# Complex amplitude version of the Dalitz model
# Same ingredients as model(x), but returns A(m2ab,m2bc)
# instead of |A|^2.
# ============================================================

def amplitude_model(x):

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    mab = atfi.sqrt(m2ab)
    mbc = atfi.sqrt(m2bc)
    mac = atfi.sqrt(m2ac)

    # BABAR 2008 helicity convention
    hel_ab_1 = -atfi.cast_complex(
        helicity_babar2008(mab, mbc, mac, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1)
    )
    hel_bc_1 = -atfi.cast_complex(
        helicity_babar2008(mbc, mab, mac, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1)
    )
    hel_ac_1 = -atfi.cast_complex(
        helicity_babar2008(mac, mbc, mab, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1)
    )

    hel_ab_2 = atfi.cast_complex(
        helicity_babar2008(mab, mbc, mac, phsp.ma, phsp.mb, phsp.mc, phsp.md, 2)
    )
    hel_bc_2 = atfi.cast_complex(
        helicity_babar2008(mbc, mab, mac, phsp.mc, phsp.mb, phsp.ma, phsp.md, 2)
    )
    hel_ac_2 = atfi.cast_complex(
        helicity_babar2008(mac, mbc, mab, phsp.ma, phsp.mc, phsp.mb, phsp.md, 2)
    )

    bw1 = atfd.breit_wigner_lineshape(
        m2ac, mrho, wrho,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw2 = atfd.breit_wigner_lineshape(
        m2ab, mkst, wkst,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw4 = atfd.breit_wigner_lineshape(
        m2ab, mk2st1430, wk2st1430,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 2, 2,
        barrier_factor=False
    )

    bw5 = atfd.breit_wigner_lineshape(
        m2ab, mkst1410, wkst1410,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw6 = atfd.breit_wigner_lineshape(
        m2ab, mkst1680, wkst1680,
        phsp.ma, phsp.mb, phsp.mc, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw7 = atfd.breit_wigner_lineshape(
        m2bc, mkst, wkst,
        phsp.mc, phsp.mb, phsp.ma, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw9 = atfd.breit_wigner_lineshape(
        m2bc, mk2st1430, wk2st1430,
        phsp.mc, phsp.mb, phsp.ma, phsp.md,
        1.5, 5.0, 2, 2,
        barrier_factor=False
    )

    bw10 = atfd.breit_wigner_lineshape(
        m2bc, mkst1410, wkst1410,
        phsp.mc, phsp.mb, phsp.ma, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw11 = atfd.breit_wigner_lineshape(
        m2ac, momega, womega,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False
    )

    bw12 = atfd.breit_wigner_lineshape(
        m2ac, mf2_1270, wf2_1270,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 2, 2,
        barrier_factor=False
    )

    bw13 = atfd.breit_wigner_lineshape(
        m2ac, mrho1450, wrho1450,
        phsp.ma, phsp.mc, phsp.mb, phsp.md,
        1.5, 5.0, 1, 1,
        barrier_factor=False,
        md0=mrho1450 + phsp.mb
    )

    # LASS
    lass_n = LASS_babar2008(
        m2ab, lass_a, lass_r, lass_M, lass_G,
        phsp.mb, phsp.ma,
        lass_R, lass_phiR, lass_F, lass_phiF
    )

    lass_p = LASS_babar2008(
        m2bc, lass_a, lass_r, lass_M, lass_G,
        phsp.mb, phsp.mc,
        lass_R, lass_phiR, lass_F, lass_phiF
    )

    # K matrix
    km = atfd.kmatrix_lineshape(
        m2ac,
        m_poles,
        g_poles,
        s0,
        fij,
        b_poles,
        K_matrix_sprod,
        fprod1,
        [[mpi, mpi], [mkz, mkz], [mpi], [meta, meta], [meta, metap]]
    )

    def _model(a1r,
               a1i,
               a2r,
               a2i,
               a3r,
               a3i,
               a4r,
               a4i,
               a5r,
               a5i,
               a6r,
               a6i,
               a7r,
               a7i,
               a8r,
               a8i,
               a9r,
               a9i,
               a10r,
               a10i,
               a11r,
               a11i,
               a12r,
               a12i,
               a13r,
               a13i,
               a14r,
               a14i,
               switches=fit_switches):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)
        a4 = atfi.complex(a4r, a4i)
        a5 = atfi.complex(a5r, a5i)
        a6 = atfi.complex(a6r, a6i)
        a7 = atfi.complex(a7r, a7i)
        a8 = atfi.complex(a8r, a8i)
        a9 = atfi.complex(a9r, a9i)
        a10 = atfi.complex(a10r, a10i)
        a11 = atfi.complex(a11r, a11i)
        a12 = atfi.complex(a12r, a12i)
        a13 = atfi.complex(a13r, a13i)
        a14 = atfi.complex(a14r, a14i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl += a1 * bw1 * hel_ac_1
        if switches[1]:
            ampl += a2 * bw2 * hel_ab_1
        if switches[2]:
            ampl += a3 * lass_n
        if switches[3]:
            ampl += a4 * bw4 * hel_ab_2
        if switches[4]:
            ampl += a5 * bw5 * hel_ab_1
        if switches[5]:
            ampl += a6 * bw6 * hel_ab_1
        if switches[6]:
            ampl += a7 * bw7 * hel_bc_1
        if switches[7]:
            ampl += a8 * lass_p
        if switches[8]:
            ampl += a9 * bw9 * hel_bc_2
        if switches[9]:
            ampl += a10 * bw10 * hel_bc_1
        if switches[10]:
            ampl += a11 * bw11 * hel_ac_1
        if switches[11]:
            ampl += a12 * bw12 * hel_ac_2
        if switches[12]:
            ampl += a13 * bw13 * hel_ac_1
        if switches[13]:
            ampl += a14 * km
        if switches[14]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        return ampl

    return _model


# ============================================================
# Build fitted-sample projection and save comparison plot
# ============================================================

if CREATE_FITTED_SAMPLE:
    print_section("Build fitted sample for projection plots")

    amps_to_plot = [
        comp["id"] - 1
        for comp in COMPONENTS
        if SWITCH[comp["key"]]
    ]

    if SWITCH["const"]:
        amps_to_plot.append(14)

    n_component_columns = 15





    n_component_columns = max(amps_to_plot) + 1
    n_toy = max(1, int(nnorm_dalitz_effective / 100))

    integration_np_for_plot = integration_sample.numpy() if hasattr(integration_sample, "numpy") else integration_sample
    integration_np_for_plot = np.asarray(integration_np_for_plot)[:, :2]

    eval_chunk = 500000
    weights_list = []

    for i in range(0, len(integration_np_for_plot), eval_chunk):
        x_chunk = atfi.const(integration_np_for_plot[i:i + eval_chunk])
        w_chunk = fitted_model(x_chunk).numpy()
        w_chunk = np.real(w_chunk)
        w_chunk[w_chunk < 0.0] = 0.0
        weights_list.append(w_chunk)

    weights = np.concatenate(weights_list)
    weights_sum = np.sum(weights)
    if weights_sum <= 0.0:
        raise RuntimeError("Fitted-model weights have non-positive sum.")
    weights = weights / weights_sum

    idx = np.random.choice(
        np.arange(len(integration_np_for_plot)),
        size=n_toy,
        replace=True,
        p=weights
    )

    toy_xy = integration_np_for_plot[idx]

    total = fitted_model(atfi.const(toy_xy)).numpy()
    total = np.real(total)
    total[total <= 0.0] = 1e-12

    component_columns = np.zeros((len(toy_xy), n_component_columns))

    for amp_index in amps_to_plot:
        component_switches = [False] * 15
        component_switches[amp_index] = True

        comp = fitted_model(
            atfi.const(toy_xy),
            switches=component_switches
        ).numpy()

        comp = np.real(comp)
        comp[comp < 0.0] = 0.0
        component_columns[:, amp_index] = comp / total

    fitted_sample = np.column_stack([
        toy_xy,
        component_columns
    ])

    print("fitted_sample shape:", fitted_sample.shape)
    np.save(ARRAY_DIR / "fitted_sample_projection.npy", fitted_sample)

    def make_fit_comparison_plot():
        import matplotlib.pyplot as plt

        tfp.set_lhcb_style(size=12, usetex=False)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

        amps_to_plot_local = list(amps_to_plot)

        tfp.plot_distr2d(
            data_tf_small[:, 1],
            data_tf_small[:, 0],
            bins=(50, 50),
            ranges=((0.3, 3.1), (0.3, 3.1)),
            fig=fig,
            ax=ax[0, 0],
            labels=(r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^-)$"),
            units=("MeV$^2$", "MeV$^2$"),
            log=True,
        )

        tfp.plot_distr1d_comparison(
            data_tf_small[:, 1],
            fitted_sample[:, 1],
            cweights=[fitted_sample[:, 2 + i] for i in amps_to_plot_local],
            bins=50,
            range=(0.3, 3.1),
            ax=ax[0, 1],
            label=r"$m^2(K_S^0\pi^+)$",
            units="MeV$^2$",
        )

        tfp.plot_distr1d_comparison(
            data_tf_small[:, 0],
            fitted_sample[:, 0],
            cweights=[fitted_sample[:, 2 + i] for i in amps_to_plot_local],
            bins=50,
            range=(0.3, 3.1),
            ax=ax[1, 0],
            label=r"$m^2(K_S^0\pi^-)$",
            units="MeV$^2$",
        )

        tfp.plot_distr1d_comparison(
            phsp.m2ac(data_tf_small),
            phsp.m2ac(fitted_sample),
            cweights=[fitted_sample[:, 2 + i] for i in amps_to_plot_local],
            bins=50,
            range=(0.05, 1.9),
            ax=ax[1, 1],
            label=r"$m^2(\pi^+\pi^-)$",
            units="MeV$^2$",
            log=True,
        )

        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        return fig

    try:
        fig_fit = make_fit_comparison_plot()
        save_figure(fig_fit, "fit_comparison_dalitz")
        import matplotlib.pyplot as plt
        plt.close(fig_fit)
        print("[OUTPUT] Saved fit comparison figure.")
    except Exception as exc:
        print(f"[WARNING] Could not save fit comparison figure: {exc}")




# ============================================================
# Inspect arrays before mixing fit
# ============================================================

print_section("Prepare mixing fit sample")

print("Full data_np shape:", data_np.shape)
print("Full integration_np shape:", integration_np.shape)

print("\nFirst 5 rows of data_np:")
print(data_np[:5])

print("\nColumn-wise min / max for data_np:")
for i in range(data_np.shape[1]):
    col = data_np[:, i]
    finite = np.isfinite(col)
    if np.any(finite):
        print(f"col {i:2d}: min = {np.nanmin(col): .6g}, max = {np.nanmax(col): .6g}, mean = {np.nanmean(col): .6g}")
    else:
        print(f"col {i:2d}: no finite entries")

# Mixing-fit sample configuration.
MIX_COL_M2_KSPIM = 0
MIX_COL_M2_KSPIP = 1
MIX_COL_TIME = 2

def build_mixing_array(arr):
    m2_kspim = arr[:, MIX_COL_M2_KSPIM]
    m2_kspip = arr[:, MIX_COL_M2_KSPIP]
    time_tau = arr[:, MIX_COL_TIME]

    out = np.column_stack([m2_kspim, m2_kspip, time_tau])
    mask = np.all(np.isfinite(out), axis=1)
    out = out[mask]
    return np.ascontiguousarray(out)

data_mixing_all_np = build_mixing_array(data_np_mixing)

if USE_MIXING_TMAX_CUT:
    before = len(data_mixing_all_np)
    data_mixing_all_np = data_mixing_all_np[data_mixing_all_np[:, 2] < MIXING_TMAX]
    print(f"Applied DATA t < {MIXING_TMAX} cut: {before} -> {len(data_mixing_all_np)}")

nfit_mixing_effective = min(NFIT_MIXING, len(data_mixing_all_np))
nnorm_mixing_effective = min(NNORM_MIXING, integration_np_mixing.shape[0])

data_mixing_np = np.ascontiguousarray(data_mixing_all_np[:nfit_mixing_effective, :3])
integration_mixing_test_np = np.ascontiguousarray(integration_np_mixing[:nnorm_mixing_effective, :3])

# In v10 the default mixing fit uses a CPU-side amplitude cache and chunked
# NLL evaluation. Therefore, avoid creating one huge normalization tensor on
# the GPU unless the old non-cached path is explicitly requested.
if USE_MIXING_AMPLITUDE_CACHE:
    data_mixing = None
    integration_mixing_test = None
else:
    data_mixing = atfi.const(data_mixing_np)
    integration_mixing_test = atfi.const(integration_mixing_test_np)

print("data_mixing:", data_mixing_np.shape)
print("integration_mixing_test_np:", integration_mixing_test_np.shape)

print("\nFirst 5 rows of data_mixing = [m2KsPiMinus, m2KsPiPlus, t/tau]:")
print(data_mixing_np[:5])

print("\nDecay-time range in data_mixing:")
print("  min =", np.min(data_mixing_np[:, 2]))
print("  max =", np.max(data_mixing_np[:, 2]))
print("  mean =", np.mean(data_mixing_np[:, 2]))

print("\nDecay-time range in integration_mixing_test:")
print("  min =", np.min(integration_mixing_test_np[:, 2]))
print("  max =", np.max(integration_mixing_test_np[:, 2]))
print("  mean =", np.mean(integration_mixing_test_np[:, 2]))

np.save(ARRAY_DIR / "data_mixing.npy", data_mixing_np)
np.save(ARRAY_DIR / "integration_mixing.npy", integration_mixing_test_np)


def describe_time(label, arr):
    t = arr[:, 2]
    print(label)
    print("  shape:", arr.shape)
    print("  t min :", np.min(t))
    print("  t max :", np.max(t))
    print("  t mean:", np.mean(t))
    print("  t q10 :", np.quantile(t, 0.10))
    print("  t q50 :", np.quantile(t, 0.50))
    print("  t q90 :", np.quantile(t, 0.90))
    print()

describe_time("DATA MIXING", data_mixing_np)
describe_time("INTEGRATION MIXING", integration_mixing_test_np)




# ============================================================
# Mixing-amplitude cache utilities
# ============================================================

def build_mixing_amplitude_cache(
    x_np,
    label: str,
    chunk_size: int = MIXING_CACHE_CHUNK,
    convention: str = "nominal",
):
    """Precompute A(D0), A(D0bar) and t for the mixing fit.

    This is the core v10 change. The Dalitz amplitudes depend only on the
    phase-space point and on fixed amplitude parameters. They do not change
    while Minuit scans x, y, |q/p| and phi.

    The returned arrays are NumPy arrays kept on CPU memory. During the NLL
    evaluation, only one chunk is converted to TensorFlow at a time.
    """

    x_np = np.ascontiguousarray(x_np[:, :3])
    n_events = len(x_np)

    print(f"[CACHE] Building mixing amplitude cache for {label}")
    print(f"[CACHE] {label}: events = {n_events}")
    print(f"[CACHE] {label}: chunk size = {chunk_size}")

    amp_chunks = []
    ampbar_chunks = []
    time_chunks = []

    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        print(f"[CACHE] {label}: {start} -> {stop}")

        chunk = x_np[start:stop]

        dalitz = atfi.const(chunk[:, 0:2])
        dalitz_swapped = atfi.const(chunk[:, [1, 0]])

        amp_nominal = amplitude_model(dalitz)(
            **amp_pars_for_mixing,
            switches=fit_switches
        ).numpy()

        amp_swapped = amplitude_model(dalitz_swapped)(
            **amp_pars_for_mixing,
            switches=fit_switches
        ).numpy()

        if convention == "nominal":
            amp_chunk = amp_nominal
            ampbar_chunk = amp_swapped

        elif convention == "swap_A_Abar":
            amp_chunk = amp_swapped
            ampbar_chunk = amp_nominal

        elif convention == "conj_Abar":
            amp_chunk = amp_nominal
            ampbar_chunk = np.conjugate(amp_swapped)

        elif convention == "conj_both":
            amp_chunk = np.conjugate(amp_nominal)
            ampbar_chunk = np.conjugate(amp_swapped)

        else:
            raise ValueError(f"Unknown mixing amplitude convention: {convention}")

        amp_chunks.append(np.asarray(amp_chunk, dtype=np.complex128))
        ampbar_chunks.append(np.asarray(ampbar_chunk, dtype=np.complex128))
        time_chunks.append(np.asarray(chunk[:, 2], dtype=np.float64))

    print(f"[CACHE] {label}: convention = {convention}")

    cache = {
        "A": np.ascontiguousarray(np.concatenate(amp_chunks)),
        "Abar": np.ascontiguousarray(np.concatenate(ampbar_chunks)),
        "t": np.ascontiguousarray(np.concatenate(time_chunks)),
        "label": label,
        "n_events": int(n_events),
    }

    print(f"[CACHE] Finished {label}: A shape = {cache['A'].shape}, Abar shape = {cache['Abar'].shape}")
    return cache


@atfi.function
def mixing_density_cached_amplitudes(
    amp_dz,
    amp_dzb,
    time_tau,
    x_mix,
    y_mix,
    qp_abs,
    qp_phi,
):
    """Mixing density using precomputed complex amplitudes."""

    tau = atfi.const(1.0)

    time_tau = tf.cast(time_tau, atfi.fptype())
    x_mix = tf.cast(x_mix, atfi.fptype())
    y_mix = tf.cast(y_mix, atfi.fptype())
    qp_abs = tf.cast(qp_abs, atfi.fptype())
    qp_phi = tf.cast(qp_phi, atfi.fptype())

    tep = atfm.psip(time_tau, y_mix, tau)
    tem = atfm.psim(time_tau, y_mix, tau)
    tei = atfm.psii(time_tau, x_mix, tau)

    qp_re = qp_abs * tf.cos(qp_phi)
    qp_im = qp_abs * tf.sin(qp_phi)
    q_over_p = tf.complex(qp_re, qp_im)

    amp_dz = tf.cast(amp_dz, q_over_p.dtype)
    amp_dzb = tf.cast(amp_dzb, q_over_p.dtype)

    dens = atfm.mixing_density(
        amp_dz,
        amp_dzb,
        q_over_p,
        tep,
        tem,
        tei,
        time_acceptance=None,
    )

    dens = tf.math.real(dens)
    dens = tf.cast(dens, atfi.fptype())
    dens = tf.reshape(dens, (-1,))

    return tf.clip_by_value(
        dens,
        atfi.const(1.0e-300),
        atfi.const(1.0e300),
    )


def nll_mixing_cached(data_cache, norm_cache, chunk_size: int = MIXING_NLL_CHUNK):
    """Chunked NLL for the mixing fit using cached amplitudes.

    The normalization integral is recomputed for every Minuit call because it
    depends on x, y, |q/p| and phi. The expensive Dalitz amplitude evaluation is
    cached through A and Abar.
    """

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    def _sum_log_pdf(cache, pars):
        total = atfi.const(0.0)
        n_total = int(cache["n_events"])

        for start in range(0, n_total, chunk_size):
            stop = min(start + chunk_size, n_total)

            pdf = mixing_density_cached_amplitudes(
                tf.constant(cache["A"][start:stop]),
                tf.constant(cache["Abar"][start:stop]),
                atfi.const(cache["t"][start:stop]),
                pars["x_mix"],
                pars["y_mix"],
                pars["qp_abs"],
                pars["qp_phi"],
            )

            pdf = tf.clip_by_value(
                pdf,
                atfi.const(1.0e-300),
                atfi.const(1.0e300),
            )

            total += tf.reduce_sum(tf.math.log(pdf))

        return total

    def _sum_pdf(cache, pars):
        total = atfi.const(0.0)
        n_total = int(cache["n_events"])

        for start in range(0, n_total, chunk_size):
            stop = min(start + chunk_size, n_total)

            pdf = mixing_density_cached_amplitudes(
                tf.constant(cache["A"][start:stop]),
                tf.constant(cache["Abar"][start:stop]),
                atfi.const(cache["t"][start:stop]),
                pars["x_mix"],
                pars["y_mix"],
                pars["qp_abs"],
                pars["qp_phi"],
            )

            pdf = tf.clip_by_value(
                pdf,
                atfi.const(1.0e-300),
                atfi.const(1.0e300),
            )

            total += tf.reduce_sum(pdf)

        return total

    def _nll_mixing_cached(pars):
        data_log_sum = _sum_log_pdf(data_cache, pars)
        norm_sum = _sum_pdf(norm_cache, pars)

        norm_int = norm_sum / tf.cast(n_norm_total, atfi.fptype())
        norm_int = tf.clip_by_value(
            norm_int,
            atfi.const(1.0e-300),
            atfi.const(1.0e300),
        )

        nll_value = -data_log_sum
        nll_value += tf.cast(n_data_total, atfi.fptype()) * tf.math.log(norm_int)

        return nll_value

    return _nll_mixing_cached

# ============================================================
# Fixed-point NLL diagnostic utilities
# ============================================================

def make_fixed_mixing_pars(x, y, qp_abs=1.0, qp_phi=0.0):
    """Build fixed mixing parameters for direct NLL evaluation."""

    return {
        "x_mix": atfi.const(float(x)),
        "y_mix": atfi.const(float(y)),
        "qp_abs": atfi.const(float(qp_abs)),
        "qp_phi": atfi.const(float(qp_phi)),
    }


def evaluate_cached_nll_terms(
    data_cache,
    norm_cache,
    x,
    y,
    qp_abs=1.0,
    qp_phi=0.0,
    chunk_size: int = MIXING_NLL_CHUNK,
):
    """Evaluate cached NLL terms at fixed x,y without running Minuit.

    This uses exactly the same cached density as the fit:
        mixing_density_cached_amplitudes(...)

    It returns separately:
        data_term = -sum log(pdf_data)
        norm_term = N_data * log(<pdf_norm>)
        total_nll = data_term + norm_term
    """

    pars = make_fixed_mixing_pars(x, y, qp_abs=qp_abs, qp_phi=qp_phi)

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    data_log_sum = 0.0
    norm_sum = 0.0

    data_min_pdf = +1.0e300
    data_max_pdf = -1.0e300
    norm_min_pdf = +1.0e300
    norm_max_pdf = -1.0e300

    data_bad_pdf = 0
    norm_bad_pdf = 0

    # ------------------------------------------------------------
    # Data term: -sum log(pdf_data)
    # ------------------------------------------------------------
    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        pdf = mixing_density_cached_amplitudes(
            tf.constant(data_cache["A"][start:stop]),
            tf.constant(data_cache["Abar"][start:stop]),
            atfi.const(data_cache["t"][start:stop]),
            pars["x_mix"],
            pars["y_mix"],
            pars["qp_abs"],
            pars["qp_phi"],
        )

        pdf_np = pdf.numpy()

        data_min_pdf = min(data_min_pdf, float(np.min(pdf_np)))
        data_max_pdf = max(data_max_pdf, float(np.max(pdf_np)))
        data_bad_pdf += int(np.sum(pdf_np <= 0.0))

        data_log_sum += float(tf.reduce_sum(tf.math.log(pdf)).numpy())

    # ------------------------------------------------------------
    # Normalization term: N_data * log(mean(pdf_norm))
    # ------------------------------------------------------------
    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        pdf = mixing_density_cached_amplitudes(
            tf.constant(norm_cache["A"][start:stop]),
            tf.constant(norm_cache["Abar"][start:stop]),
            atfi.const(norm_cache["t"][start:stop]),
            pars["x_mix"],
            pars["y_mix"],
            pars["qp_abs"],
            pars["qp_phi"],
        )

        pdf_np = pdf.numpy()

        norm_min_pdf = min(norm_min_pdf, float(np.min(pdf_np)))
        norm_max_pdf = max(norm_max_pdf, float(np.max(pdf_np)))
        norm_bad_pdf += int(np.sum(pdf_np <= 0.0))

        norm_sum += float(tf.reduce_sum(pdf).numpy())

    norm_int = norm_sum / float(n_norm_total)
    norm_int = np.clip(norm_int, 1.0e-300, 1.0e300)

    data_term = -data_log_sum
    norm_term = float(n_data_total) * np.log(norm_int)
    total_nll = data_term + norm_term

    return {
        "x": float(x),
        "y": float(y),
        "data_term": data_term,
        "norm_term": norm_term,
        "total_nll": total_nll,
        "norm_int": norm_int,
        "data_min_pdf": data_min_pdf,
        "data_max_pdf": data_max_pdf,
        "norm_min_pdf": norm_min_pdf,
        "norm_max_pdf": norm_max_pdf,
        "data_bad_pdf": data_bad_pdf,
        "norm_bad_pdf": norm_bad_pdf,
    }




def finite_difference_x_diagnostics(
    data_cache,
    norm_cache,
    y_fixed=0.0,
    x0=0.0,
    eps=1.0e-4,
    chunk_size=MIXING_NLL_CHUNK,
):
    """
    Compute finite-difference gradient and curvature in x at fixed y.

    g_x  = dNLL/dx
    H_xx = d2NLL/dx2

    The local predicted minimum, at fixed y, is approximately:

        x_pred = x0 - g_x / H_xx
    """

    nll_m = evaluate_cached_nll_terms(
        data_cache=data_cache,
        norm_cache=norm_cache,
        x=x0 - eps,
        y=y_fixed,
        chunk_size=chunk_size,
    )["total_nll"]

    nll_0 = evaluate_cached_nll_terms(
        data_cache=data_cache,
        norm_cache=norm_cache,
        x=x0,
        y=y_fixed,
        chunk_size=chunk_size,
    )["total_nll"]

    nll_p = evaluate_cached_nll_terms(
        data_cache=data_cache,
        norm_cache=norm_cache,
        x=x0 + eps,
        y=y_fixed,
        chunk_size=chunk_size,
    )["total_nll"]

    grad_x = (nll_p - nll_m) / (2.0 * eps)
    hess_xx = (nll_p - 2.0 * nll_0 + nll_m) / (eps * eps)

    x_pred = np.nan
    if hess_xx != 0.0:
        x_pred = x0 - grad_x / hess_xx

    print()
    print("=" * 100)
    print("Finite-difference x diagnostic")
    print("=" * 100)
    print(f"x0       = {x0:.8f}")
    print(f"y_fixed  = {y_fixed:.8f}")
    print(f"eps      = {eps:.8e}")
    print(f"NLL(x-e) = {nll_m:.10f}")
    print(f"NLL(x0)  = {nll_0:.10f}")
    print(f"NLL(x+e) = {nll_p:.10f}")
    print(f"grad_x   = {grad_x:.10e}")
    print(f"hess_xx  = {hess_xx:.10e}")
    print(f"x_pred   = {x_pred:.10e}")
    print("=" * 100)

    return {
        "x0": x0,
        "y_fixed": y_fixed,
        "eps": eps,
        "nll_minus": nll_m,
        "nll_zero": nll_0,
        "nll_plus": nll_p,
        "grad_x": grad_x,
        "hess_xx": hess_xx,
        "x_pred": x_pred,
    }




def run_fixed_point_nll_check(data_cache, norm_cache):
    """Run the first fixed-point NLL checks for the zero-mixing toy."""

    test_points = [
        ("truth",       0.000000,  0.000000),
        ("fit_result", -0.005482, -0.000917),
        ("x_offset",   -0.005600,  0.000000),
        ("opposite_x", +0.005600,  0.000000),
        ("y_offset",    0.000000, -0.000917),
    ]

    rows = []

    print()
    print("=" * 80)
    print("Fixed-point NLL check")
    print("=" * 80)

    for label, x, y in test_points:
        print(f"[NLL CHECK] Evaluating {label}: x = {x:.6f}, y = {y:.6f}")

        out = evaluate_cached_nll_terms(
            data_cache=data_cache,
            norm_cache=norm_cache,
            x=x,
            y=y,
        )

        out["label"] = label
        rows.append(out)

    min_total = min(row["total_nll"] for row in rows)
    min_data = min(row["data_term"] for row in rows)
    min_norm = min(row["norm_term"] for row in rows)

    print()
    print("=" * 120)
    print("NLL term decomposition")
    print("=" * 120)
    print(
        f"{'label':<15} {'x':>11} {'y':>11} "
        f"{'DeltaNLL':>15} {'DeltaData':>15} {'DeltaNorm':>15} "
        f"{'norm_int':>15} {'data_min_pdf':>15} {'norm_min_pdf':>15} "
        f"{'data_bad':>10} {'norm_bad':>10}"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['label']:<15} "
            f"{row['x']:>11.6f} "
            f"{row['y']:>11.6f} "
            f"{row['total_nll'] - min_total:>15.6f} "
            f"{row['data_term'] - min_data:>15.6f} "
            f"{row['norm_term'] - min_norm:>15.6f} "
            f"{row['norm_int']:>15.8e} "
            f"{row['data_min_pdf']:>15.8e} "
            f"{row['norm_min_pdf']:>15.8e} "
            f"{row['data_bad_pdf']:>10d} "
            f"{row['norm_bad_pdf']:>10d}"
        )

    print("=" * 120)

    return rows
    

def run_x_scan_nll_check(
    data_cache,
    norm_cache,
    y_fixed=0.0,
    x_min=-0.010,
    x_max=0.002,
    n_points=49,
    chunk_size: int = MIXING_NLL_CHUNK,
):
    """Scan the cached NLL as a function of x with y fixed."""

    print()
    print("=" * 100)
    print("DEBUG: 1D NLL scan in x")
    print("=" * 100)
    print(f"[X SCAN] y fixed = {y_fixed}")
    print(f"[X SCAN] range   = [{x_min}, {x_max}]")
    print(f"[X SCAN] points  = {n_points}")

    x_values = np.linspace(x_min, x_max, n_points)

    rows = []

    for x_value in x_values:
        print(f"[X SCAN] Evaluating x = {x_value:.8f}, y = {y_fixed:.8f}")

        out = evaluate_cached_nll_terms(
            data_cache=data_cache,
            norm_cache=norm_cache,
            x=float(x_value),
            y=float(y_fixed),
            qp_abs=1.0,
            qp_phi=0.0,
            chunk_size=chunk_size,
        )

        rows.append(out)

    min_total = min(row["total_nll"] for row in rows)
    min_row = min(rows, key=lambda row: row["total_nll"])

    print()
    print("=" * 100)
    print("1D x scan result")
    print("=" * 100)
    print(f"[X SCAN] minimum x = {min_row['x']:.8f}")
    print(f"[X SCAN] minimum y = {min_row['y']:.8f}")
    print(f"[X SCAN] minimum NLL = {min_row['total_nll']:.8f}")

    print()
    print(f"{'x':>12} {'y':>12} {'NLL':>20} {'DeltaNLL':>15}")
    print("-" * 65)

    for row in rows:
        print(
            f"{row['x']:>12.8f} "
            f"{row['y']:>12.8f} "
            f"{row['total_nll']:>20.6f} "
            f"{row['total_nll'] - min_total:>15.6f}"
        )

    print("=" * 100)

    return rows



def run_score_profile_in_dalitz_at_zero(
    data_np,
    data_cache,
    norm_np,
    norm_cache,
    eps=1.0e-4,
    n_bins_x=50,
    n_bins_y=50,
    chunk_size: int = MIXING_NLL_CHUNK,
    output_dir=None,
):
    """Map the local x-score contribution in the Dalitz plot at x=0, y=0.

    The local contribution to the NLL gradient is

        grad_bin = - sum_data_bin score_x
                   + N_data * sum_norm_bin[pdf0 * score_x] / sum_norm_all[pdf0]

    Positive grad_bin means that bin pushes the NLL to decrease for x < 0.
    Negative grad_bin means that bin pushes the NLL to decrease for x > 0.
    """

    print()
    print("=" * 100)
    print("DEBUG: score_x profile in Dalitz plot at x=0, y=0")
    print("=" * 100)

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    data_xy = np.asarray(data_np[:n_data_total, 0:2], dtype=np.float64)
    norm_xy = np.asarray(norm_np[:n_norm_total, 0:2], dtype=np.float64)

    x_min = min(float(np.min(data_xy[:, 0])), float(np.min(norm_xy[:, 0])))
    x_max = max(float(np.max(data_xy[:, 0])), float(np.max(norm_xy[:, 0])))
    y_min = min(float(np.min(data_xy[:, 1])), float(np.min(norm_xy[:, 1])))
    y_max = max(float(np.max(data_xy[:, 1])), float(np.max(norm_xy[:, 1])))

    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)

    print(f"[DALITZ SCORE] x range = [{x_min:.8f}, {x_max:.8f}]")
    print(f"[DALITZ SCORE] y range = [{y_min:.8f}, {y_max:.8f}]")
    print(f"[DALITZ SCORE] bins    = {n_bins_x} x {n_bins_y}")

    data_count = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    data_score_sum = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    norm_weight_sum = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    norm_weighted_score_sum = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    # ------------------------------------------------------------
    # Data part
    # ------------------------------------------------------------
    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        A = tf.constant(data_cache["A"][start:stop])
        Abar = tf.constant(data_cache["Abar"][start:stop])
        t = atfi.const(data_cache["t"][start:stop])

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        score_x_np = score_x.numpy()

        xy_chunk = data_xy[start:stop]
        ix = np.searchsorted(x_edges, xy_chunk[:, 0], side="right") - 1
        iy = np.searchsorted(y_edges, xy_chunk[:, 1], side="right") - 1

        valid = (ix >= 0) & (ix < n_bins_x) & (iy >= 0) & (iy < n_bins_y)

        np.add.at(data_count, (ix[valid], iy[valid]), 1.0)
        np.add.at(data_score_sum, (ix[valid], iy[valid]), score_x_np[valid])

    # ------------------------------------------------------------
    # Normalization expectation
    # ------------------------------------------------------------
    total_norm_weight = 0.0

    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        A = tf.constant(norm_cache["A"][start:stop])
        Abar = tf.constant(norm_cache["Abar"][start:stop])
        t = atfi.const(norm_cache["t"][start:stop])

        pdf_0 = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        pdf_0_np = pdf_0.numpy()
        score_x_np = score_x.numpy()

        total_norm_weight += float(np.sum(pdf_0_np))

        xy_chunk = norm_xy[start:stop]
        ix = np.searchsorted(x_edges, xy_chunk[:, 0], side="right") - 1
        iy = np.searchsorted(y_edges, xy_chunk[:, 1], side="right") - 1

        valid = (ix >= 0) & (ix < n_bins_x) & (iy >= 0) & (iy < n_bins_y)

        np.add.at(norm_weight_sum, (ix[valid], iy[valid]), pdf_0_np[valid])
        np.add.at(
            norm_weighted_score_sum,
            (ix[valid], iy[valid]),
            pdf_0_np[valid] * score_x_np[valid],
        )

    # ------------------------------------------------------------
    # Local normalized NLL gradient contribution
    # ------------------------------------------------------------
    local_grad_x = (
        -data_score_sum
        + float(n_data_total) * norm_weighted_score_sum / total_norm_weight
    )

    data_frac = data_count / float(np.sum(data_count))
    norm_frac = norm_weight_sum / float(np.sum(norm_weight_sum))

    data_mean_score_x = data_score_sum / np.maximum(data_count, 1.0)
    norm_mean_score_x = norm_weighted_score_sum / np.maximum(norm_weight_sum, 1.0)

    score_diff = data_mean_score_x - norm_mean_score_x
    frac_diff = data_frac - norm_frac

    total_grad_x = float(np.sum(local_grad_x))

    print()
    print("=" * 100)
    print("Dalitz score_x profile summary")
    print("=" * 100)
    print(f"[DALITZ SCORE] total local grad_x = {total_grad_x:.10e}")
    print("[DALITZ SCORE] positive local grad_x pushes the fit toward x < 0")
    print("[DALITZ SCORE] negative local grad_x pushes the fit toward x > 0")

    flat_grad = local_grad_x.reshape(-1)
    top_pos = np.argsort(flat_grad)[-15:][::-1]
    top_neg = np.argsort(flat_grad)[:15]

    print()
    print("Top positive bins: strongest pull toward x < 0")
    print("-" * 110)
    print(
        f"{'rank':>4} {'ix':>4} {'iy':>4} "
        f"{'m2p_low':>11} {'m2p_high':>11} "
        f"{'m2m_low':>11} {'m2m_high':>11} "
        f"{'grad_x':>15} {'data_frac':>12} {'norm_frac':>12} {'score_diff':>15}"
    )

    for rank, idx in enumerate(top_pos, start=1):
        ix = idx // n_bins_y
        iy = idx % n_bins_y

        print(
            f"{rank:>4d} {ix:>4d} {iy:>4d} "
            f"{x_edges[ix]:>11.6f} {x_edges[ix+1]:>11.6f} "
            f"{y_edges[iy]:>11.6f} {y_edges[iy+1]:>11.6f} "
            f"{local_grad_x[ix, iy]:>15.6e} "
            f"{data_frac[ix, iy]:>12.6e} "
            f"{norm_frac[ix, iy]:>12.6e} "
            f"{score_diff[ix, iy]:>15.6e}"
        )

    print()
    print("Top negative bins: strongest pull toward x > 0")
    print("-" * 110)
    print(
        f"{'rank':>4} {'ix':>4} {'iy':>4} "
        f"{'m2p_low':>11} {'m2p_high':>11} "
        f"{'m2m_low':>11} {'m2m_high':>11} "
        f"{'grad_x':>15} {'data_frac':>12} {'norm_frac':>12} {'score_diff':>15}"
    )

    for rank, idx in enumerate(top_neg, start=1):
        ix = idx // n_bins_y
        iy = idx % n_bins_y

        print(
            f"{rank:>4d} {ix:>4d} {iy:>4d} "
            f"{x_edges[ix]:>11.6f} {x_edges[ix+1]:>11.6f} "
            f"{y_edges[iy]:>11.6f} {y_edges[iy+1]:>11.6f} "
            f"{local_grad_x[ix, iy]:>15.6e} "
            f"{data_frac[ix, iy]:>12.6e} "
            f"{norm_frac[ix, iy]:>12.6e} "
            f"{score_diff[ix, iy]:>15.6e}"
        )

    print("=" * 100)

    # ------------------------------------------------------------
    # Save arrays and optional plots
    # ------------------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_npz = output_dir / "dalitz_score_x_profile_zero.npz"

        np.savez(
            out_npz,
            x_edges=x_edges,
            y_edges=y_edges,
            local_grad_x=local_grad_x,
            data_frac=data_frac,
            norm_frac=norm_frac,
            frac_diff=frac_diff,
            data_mean_score_x=data_mean_score_x,
            norm_mean_score_x=norm_mean_score_x,
            score_diff=score_diff,
        )

        print(f"[DALITZ SCORE] Saved arrays: {out_npz}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

            plt.figure(figsize=(8, 7))
            plt.imshow(
                local_grad_x.T,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(label="local grad_x")
            plt.xlabel(r"$m^2(K_S\pi^+)$")
            plt.ylabel(r"$m^2(K_S\pi^-)$")
            plt.title(r"Local contribution to $d\mathrm{NLL}/dx$ at $x=y=0$")
            plt.tight_layout()
            plt.savefig(output_dir / "dalitz_local_grad_x.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 7))
            plt.imshow(
                score_diff.T,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(label="data score_x - norm score_x")
            plt.xlabel(r"$m^2(K_S\pi^+)$")
            plt.ylabel(r"$m^2(K_S\pi^-)$")
            plt.title(r"Difference in mean score_x")
            plt.tight_layout()
            plt.savefig(output_dir / "dalitz_score_x_diff.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 7))
            plt.imshow(
                frac_diff.T,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(label="data fraction - norm fraction")
            plt.xlabel(r"$m^2(K_S\pi^+)$")
            plt.ylabel(r"$m^2(K_S\pi^-)$")
            plt.title("Dalitz density difference")
            plt.tight_layout()
            plt.savefig(output_dir / "dalitz_density_frac_diff.png", dpi=160)
            plt.close()

            print(f"[DALITZ SCORE] Saved plots in: {output_dir}")

        except Exception as exc:
            print(f"[DALITZ SCORE] Could not save plots: {exc}")

    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "local_grad_x": local_grad_x,
        "data_frac": data_frac,
        "norm_frac": norm_frac,
        "frac_diff": frac_diff,
        "data_mean_score_x": data_mean_score_x,
        "norm_mean_score_x": norm_mean_score_x,
        "score_diff": score_diff,
        "total_grad_x": total_grad_x,
    }



def run_score_check_at_zero(
    data_cache,
    norm_cache,
    eps=1.0e-4,
    chunk_size: int = MIXING_NLL_CHUNK,
):
    """Finite-difference score check at x=0,y=0.

    For the normalized likelihood,

        dNLL/dx per event = - <d log pdf / dx>_data + d log(norm_int)/dx

    If this derivative is positive at x=0, the NLL decreases when x moves negative.
    """

    print()
    print("=" * 100)
    print("DEBUG: score check at x=0, y=0")
    print("=" * 100)
    print(f"[SCORE] finite-difference eps = {eps}")

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    # ------------------------------------------------------------
    # Data scores
    # ------------------------------------------------------------
    data_score_x_sum = 0.0
    data_score_y_sum = 0.0
    data_score_x2_sum = 0.0
    data_score_y2_sum = 0.0

    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        A = tf.constant(data_cache["A"][start:stop])
        Abar = tf.constant(data_cache["Abar"][start:stop])
        t = atfi.const(data_cache["t"][start:stop])

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(+eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(-eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)) / atfi.const(2.0 * eps)
        score_y = (tf.math.log(pdf_y_plus) - tf.math.log(pdf_y_minus)) / atfi.const(2.0 * eps)

        score_x_np = score_x.numpy()
        score_y_np = score_y.numpy()

        data_score_x_sum += float(np.sum(score_x_np))
        data_score_y_sum += float(np.sum(score_y_np))
        data_score_x2_sum += float(np.sum(score_x_np * score_x_np))
        data_score_y2_sum += float(np.sum(score_y_np * score_y_np))

    data_mean_score_x = data_score_x_sum / float(n_data_total)
    data_mean_score_y = data_score_y_sum / float(n_data_total)

    data_std_score_x = np.sqrt(data_score_x2_sum / float(n_data_total) - data_mean_score_x**2)
    data_std_score_y = np.sqrt(data_score_y2_sum / float(n_data_total) - data_mean_score_y**2)

    # ------------------------------------------------------------
    # Normalization derivative
    # ------------------------------------------------------------
    norm_sum_0 = 0.0
    norm_sum_x_plus = 0.0
    norm_sum_x_minus = 0.0
    norm_sum_y_plus = 0.0
    norm_sum_y_minus = 0.0

    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        A = tf.constant(norm_cache["A"][start:stop])
        Abar = tf.constant(norm_cache["Abar"][start:stop])
        t = atfi.const(norm_cache["t"][start:stop])

        pdf_0 = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(+eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(-eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        norm_sum_0 += float(tf.reduce_sum(pdf_0).numpy())
        norm_sum_x_plus += float(tf.reduce_sum(pdf_x_plus).numpy())
        norm_sum_x_minus += float(tf.reduce_sum(pdf_x_minus).numpy())
        norm_sum_y_plus += float(tf.reduce_sum(pdf_y_plus).numpy())
        norm_sum_y_minus += float(tf.reduce_sum(pdf_y_minus).numpy())

    dlogI_dx = ((norm_sum_x_plus - norm_sum_x_minus) / (2.0 * eps)) / norm_sum_0
    dlogI_dy = ((norm_sum_y_plus - norm_sum_y_minus) / (2.0 * eps)) / norm_sum_0

    grad_x_per_event = -data_mean_score_x + dlogI_dx
    grad_y_per_event = -data_mean_score_y + dlogI_dy

    print()
    print("=" * 100)
    print("Score check summary")
    print("=" * 100)
    print(f"data <d log pdf / dx> = {data_mean_score_x:.10e}")
    print(f"norm d log I / dx     = {dlogI_dx:.10e}")
    print(f"dNLL/dx per event     = {grad_x_per_event:.10e}")
    print(f"dNLL/dx total         = {grad_x_per_event * float(n_data_total):.10e}")
    print()
    print(f"data <d log pdf / dy> = {data_mean_score_y:.10e}")
    print(f"norm d log I / dy     = {dlogI_dy:.10e}")
    print(f"dNLL/dy per event     = {grad_y_per_event:.10e}")
    print(f"dNLL/dy total         = {grad_y_per_event * float(n_data_total):.10e}")
    print()
    print(f"std score x, data     = {data_std_score_x:.10e}")
    print(f"std score y, data     = {data_std_score_y:.10e}")
    print("=" * 100)

    print()
    print("[SCORE] Interpretation:")
    print("[SCORE] If dNLL/dx > 0 at x=0, the NLL decreases by moving x negative.")
    print("[SCORE] If dNLL/dx < 0 at x=0, the NLL decreases by moving x positive.")

    return {
        "data_mean_score_x": data_mean_score_x,
        "data_mean_score_y": data_mean_score_y,
        "dlogI_dx": dlogI_dx,
        "dlogI_dy": dlogI_dy,
        "grad_x_per_event": grad_x_per_event,
        "grad_y_per_event": grad_y_per_event,
        "grad_x_total": grad_x_per_event * float(n_data_total),
        "grad_y_total": grad_y_per_event * float(n_data_total),
        "data_std_score_x": data_std_score_x,
        "data_std_score_y": data_std_score_y,
    }

def run_score_profile_in_time_at_zero(
    data_cache,
    norm_cache,
    eps=1.0e-4,
    n_t_bins=12,
    chunk_size: int = MIXING_NLL_CHUNK,
):
    """Compare the x-score profile in time for data and normalized expectation.

    Data:
        unweighted mean score_x in each t bin.

    Norm:
        weighted mean score_x in each t bin using pdf(x=0,y=0) as weight.
        This represents the expectation from the fitted PDF at zero mixing.
    """

    print()
    print("=" * 100)
    print("DEBUG: score_x profile in time at x=0, y=0")
    print("=" * 100)

    data_t_min = float(np.min(data_cache["t"]))
    data_t_max = float(np.max(data_cache["t"]))

    t_edges = np.linspace(data_t_min, data_t_max, n_t_bins + 1)

    print(f"[SCORE PROFILE] t range = [{data_t_min:.6f}, {data_t_max:.6f}]")
    print(f"[SCORE PROFILE] n_t_bins = {n_t_bins}")

    data_count = np.zeros(n_t_bins, dtype=np.float64)
    data_score_x_sum = np.zeros(n_t_bins, dtype=np.float64)

    norm_weight_sum = np.zeros(n_t_bins, dtype=np.float64)
    norm_weighted_score_x_sum = np.zeros(n_t_bins, dtype=np.float64)

    # ------------------------------------------------------------
    # Data score profile
    # ------------------------------------------------------------
    n_data_total = int(data_cache["n_events"])

    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        A = tf.constant(data_cache["A"][start:stop])
        Abar = tf.constant(data_cache["Abar"][start:stop])
        t_tf = atfi.const(data_cache["t"][start:stop])
        t_np = data_cache["t"][start:stop]

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        score_x_np = score_x.numpy()

        bin_idx = np.searchsorted(t_edges, t_np, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < n_t_bins)

        np.add.at(data_count, bin_idx[valid], 1.0)
        np.add.at(data_score_x_sum, bin_idx[valid], score_x_np[valid])

    # ------------------------------------------------------------
    # Norm expected score profile
    # Weighted by pdf(x=0,y=0)
    # ------------------------------------------------------------
    n_norm_total = int(norm_cache["n_events"])

    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        A = tf.constant(norm_cache["A"][start:stop])
        Abar = tf.constant(norm_cache["Abar"][start:stop])
        t_tf = atfi.const(norm_cache["t"][start:stop])
        t_np = norm_cache["t"][start:stop]

        pdf_0 = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(0.0),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        pdf_0_np = pdf_0.numpy()
        score_x_np = score_x.numpy()

        bin_idx = np.searchsorted(t_edges, t_np, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < n_t_bins)

        np.add.at(norm_weight_sum, bin_idx[valid], pdf_0_np[valid])
        np.add.at(
            norm_weighted_score_x_sum,
            bin_idx[valid],
            pdf_0_np[valid] * score_x_np[valid],
        )

    data_mean_score_x = data_score_x_sum / np.maximum(data_count, 1.0)
    norm_mean_score_x = norm_weighted_score_x_sum / np.maximum(norm_weight_sum, 1.0)

    data_frac = data_count / np.sum(data_count)
    norm_frac = norm_weight_sum / np.sum(norm_weight_sum)

    print()
    print("=" * 130)
    print("score_x profile in time")
    print("=" * 130)
    print(
        f"{'bin':>4} {'t_low':>10} {'t_high':>10} "
        f"{'data_frac':>12} {'norm_frac':>12} "
        f"{'data_score_x':>16} {'norm_score_x':>16} {'diff':>16}"
    )
    print("-" * 130)

    for i in range(n_t_bins):
        diff = data_mean_score_x[i] - norm_mean_score_x[i]

        print(
            f"{i:>4d} "
            f"{t_edges[i]:>10.5f} "
            f"{t_edges[i+1]:>10.5f} "
            f"{data_frac[i]:>12.6e} "
            f"{norm_frac[i]:>12.6e} "
            f"{data_mean_score_x[i]:>16.8e} "
            f"{norm_mean_score_x[i]:>16.8e} "
            f"{diff:>16.8e}"
        )

    print("=" * 130)

    return {
        "t_edges": t_edges,
        "data_frac": data_frac,
        "norm_frac": norm_frac,
        "data_mean_score_x": data_mean_score_x,
        "norm_mean_score_x": norm_mean_score_x,
    }


# ============================================================
# Time-dependent mixing density
# ============================================================

def mixing_density_d0_only(
    xdata,
    x_mix,
    y_mix,
    qp_abs,
    qp_phi,
    switches=fit_switches,
    **amp_pars
):
    x3 = tf.reshape(xdata, (-1, 3))

    dalitz = x3[:, 0:2]
    time_tau = x3[:, 2]

    dalitz_swapped = tf.stack(
        [x3[:, 1], x3[:, 0]],
        axis=1
    )

    ampl_dz = amplitude_model(dalitz)(
        **amp_pars,
        switches=switches
    )

    ampl_dzb = amplitude_model(dalitz_swapped)(
        **amp_pars,
        switches=switches
    )

    tau = atfi.const(1.0)

    x_mix = tf.cast(x_mix, atfi.fptype())
    y_mix = tf.cast(y_mix, atfi.fptype())
    qp_abs = tf.cast(qp_abs, atfi.fptype())
    qp_phi = tf.cast(qp_phi, atfi.fptype())

    tep = atfm.psip(time_tau, y_mix, tau)
    tem = atfm.psim(time_tau, y_mix, tau)
    tei = atfm.psii(time_tau, x_mix, tau)

    qp_re = qp_abs * tf.cos(qp_phi)
    qp_im = qp_abs * tf.sin(qp_phi)

    q_over_p = tf.complex(qp_re, qp_im)

    dens = atfm.mixing_density(
        ampl_dz,
        ampl_dzb,
        q_over_p,
        tep,
        tem,
        tei,
        time_acceptance=None,
    )

    dens = tf.math.real(dens)
    dens = tf.cast(dens, atfi.fptype())
    dens = tf.reshape(dens, (-1,))

    return tf.clip_by_value(
        dens,
        atfi.const(1.0e-300),
        atfi.const(1.0e300)
    )


def nll_mixing(data, norm):
    @atfi.function
    def _nll_mixing(pars):
        data_pdf = mixing_density_d0_only(data, **pars)
        norm_pdf = mixing_density_d0_only(norm, **pars)

        norm_int = tf.reduce_mean(norm_pdf)

        norm_int = tf.clip_by_value(
            norm_int,
            atfi.const(1.0e-300),
            atfi.const(1.0e300)
        )

        n_events = tf.cast(tf.shape(data_pdf)[0], atfi.fptype())

        nll_value = -tf.reduce_sum(tf.math.log(data_pdf))
        nll_value += n_events * tf.math.log(norm_int)

        return nll_value

    return _nll_mixing


def build_mixing_fit_parameters():
    mixing_pars = []

    for name, value in amp_pars_for_mixing.items():
        v = float(value.numpy()) if hasattr(value, "numpy") else float(value)
        p = tfo.FitParameter(name, v, -100.0, 100.0)
        p.fix()
        mixing_pars.append(p)

    for name in ["x_mix", "y_mix", "qp_abs", "qp_phi"]:
        cfg = MIXING_PARAMETER_CONFIG[name]
        p = tfo.FitParameter(name, cfg["initial"], cfg["lower"], cfg["upper"])
        if name not in MIXING_FLOAT_PARAMS:
            p.fix()
        mixing_pars.append(p)

    print("Parameters used in the mixing fit:")
    for p in mixing_pars:
        try:
            fixed = p.fixed
        except AttributeError:
            fixed = "?"
        print(f"{p.name:12s} fixed = {fixed}")

    return mixing_pars


mixing_result = None

if RUN_MIXING_FIT:
    print_section("Run mixing MINUIT fit")

    mixing_pars = build_mixing_fit_parameters()

    if USE_MIXING_AMPLITUDE_CACHE:
        print("[CACHE] USE_MIXING_AMPLITUDE_CACHE=1: using v10 cached/chunked mixing NLL.")
        data_mixing_cache = build_mixing_amplitude_cache(
            data_mixing_np,
            label="data_mixing",
            chunk_size=MIXING_CACHE_CHUNK,
        )
        integration_mixing_cache = build_mixing_amplitude_cache(
            integration_mixing_test_np,
            label="integration_mixing",
            chunk_size=MIXING_CACHE_CHUNK,
        )
        mixing_nll_function = nll_mixing_cached(
            data_mixing_cache,
            integration_mixing_cache,
            chunk_size=MIXING_NLL_CHUNK,
        )
    else:
        print("[CACHE] USE_MIXING_AMPLITUDE_CACHE=0: using original v9 non-cached mixing NLL.")
        mixing_nll_function = nll_mixing(data_mixing, integration_mixing_test)

    print("[DEBUG MIXING NLL] object  =", mixing_nll_function)
    print("[DEBUG MIXING NLL] type    =", type(mixing_nll_function))
    print("[DEBUG MIXING NLL] callable=", callable(mixing_nll_function))

    if mixing_nll_function is None:
        raise RuntimeError(
            "mixing_nll_function is None before tfo.run_minuit. "
            "nll_mixing_cached(...) did not return a callable NLL function."
        )

    if not callable(mixing_nll_function):
        raise RuntimeError(
            "mixing_nll_function is not callable before tfo.run_minuit. "
            f"type = {type(mixing_nll_function)}"
        )

    mixing_result = tfo.run_minuit(
        mixing_nll_function,
        mixing_pars
    )

    print(mixing_result)
    save_json(mixing_result, JSON_DIR / "mixing_fit_result.json")
else:
    print("[INFO] RUN_MIXING_FIT=0. Mixing fit skipped.")
    mixing_result = {
        "params": {
            "x_mix": [MIXING_PARAMETER_CONFIG["x_mix"]["initial"], np.nan],
            "y_mix": [MIXING_PARAMETER_CONFIG["y_mix"]["initial"], np.nan],
            "qp_abs": [MIXING_PARAMETER_CONFIG["qp_abs"]["initial"], np.nan],
            "qp_phi": [MIXING_PARAMETER_CONFIG["qp_phi"]["initial"], np.nan],
        }
    }




# ============================================================
# Optional diagnostics
# ============================================================

if RUN_DEBUG_AMPLITUDE_COMPARISON:
    print_section("DEBUG: compare fitter amplitudes with generator amplitudes")

    debug_points_np = np.array([
        [1.2684540424717581, 1.7152151941093681, 1.0419191221101014],
        [1.1278512036779169, 2.0845581905735284, 2.7374944938177341],
        [0.64519714817279461, 1.7015042728467948, 0.065377127005068078],
        [0.70079010193561819, 1.411915982125246, 1.5532807861479996],
        [1.063549970925167, 1.0207691065513718, 0.23716528036219839],
        [1.7606842271770025, 0.52756466925569434, 0.27253950153339257],
        [1.1726773024324177, 1.3094838534913757, 1.2773223941391754],
        [0.5705989631823134, 2.4850078706385386, 0.25247505035053469],
        [1.2580494085910787, 1.5280504232052925, 0.22891758159108447],
        [1.3253105723027796, 1.457570149739535, 0.13924997717204798],
    ], dtype=np.float64)

    A_gen_np = np.array([
        -3.0206051258152185 + 3.8864635144381907j,
         7.0189406696334711 + 7.5516174191855434j,
         9.5567351552712818 + 1.9700351961244751j,
        18.950597677541278  - 2.5192646579057314j,
        -3.9906879014602183 + 11.340853329379925j,
        13.256287045375393  + 10.644608750978243j,
        -0.84259504742133373 + 9.7167872626544316j,
        -9.3107387579356011 + 12.509401745796158j,
         6.3422972364112979 - 2.9365907467403365j,
         7.4112699450693311 - 2.2422370026047571j,
    ], dtype=np.complex128)

    Abar_gen_np = np.array([
        -0.65526189387621048 + 4.6090696487088536j,
        -6.4025187243217392  - 4.4299555997122972j,
         8.0097348255493994  + 9.6095278198213059j,
         1.1927667389440755  + 10.813793786944338j,
        -4.9898605772432099  + 13.023881457992921j,
         8.4842872159165434  + 4.142055769759553j,
         0.31879014499771863 + 9.0959550122390755j,
         2.9564001037168799  - 6.3425978655529613j,
         7.8241812769815278  - 2.5404526977428543j,
         8.1437357119195255  - 2.0632239784783351j,
    ], dtype=np.complex128)

    dalitz_debug = debug_points_np[:, :2]
    dalitz_debug_tf = atfi.const(dalitz_debug)
    dalitz_swapped_debug_tf = atfi.const(dalitz_debug[:, ::-1])

    A_fit = amplitude_model(dalitz_debug_tf)(**amp_pars_for_mixing, switches=fit_switches).numpy()
    Abar_fit = amplitude_model(dalitz_swapped_debug_tf)(**amp_pars_for_mixing, switches=fit_switches).numpy()

    debug_rows = []
    for i in range(len(debug_points_np)):
        debug_rows.append({
            "point": i,
            "A_gen": complex(A_gen_np[i]),
            "A_fit": complex(A_fit[i]),
            "diff_A": complex(A_fit[i] - A_gen_np[i]),
            "rel_A2": (abs(A_fit[i])**2 - abs(A_gen_np[i])**2) / abs(A_gen_np[i])**2,
            "Abar_gen": complex(Abar_gen_np[i]),
            "Abar_fit": complex(Abar_fit[i]),
            "diff_Abar": complex(Abar_fit[i] - Abar_gen_np[i]),
            "rel_Abar2": (abs(Abar_fit[i])**2 - abs(Abar_gen_np[i])**2) / abs(Abar_gen_np[i])**2,
        })

    save_json(debug_rows, JSON_DIR / "debug_amplitude_comparison.json")

if RUN_NLL_DEBUG:
    print_section("DEBUG: fixed-point NLL check for zero-mixing toy")

    if not USE_MIXING_AMPLITUDE_CACHE:
        print("[DEBUG NLL] This debug block is currently implemented only for cached NLL.")
        print("[DEBUG NLL] Please set USE_MIXING_AMPLITUDE_CACHE=1.")

    else:
        print("[DEBUG NLL] Using cached mixing NLL.")

        # ------------------------------------------------------------
        # Reuse caches if they already exist from RUN_MIXING_FIT.
        # Otherwise build them here.
        # ------------------------------------------------------------
        try:
            data_cache_debug = data_mixing_cache
            integration_cache_debug = integration_mixing_cache
            print("[DEBUG NLL] Reusing existing mixing caches.")
        except NameError:
            print("[DEBUG NLL] Mixing caches not found. Building debug caches.")

            data_cache_debug = build_mixing_amplitude_cache(
                data_mixing_np,
                label="data_mixing_debug",
                chunk_size=MIXING_CACHE_CHUNK,
                convention="nominal",
            )

            integration_cache_debug = build_mixing_amplitude_cache(
                integration_mixing_test_np,
                label="integration_mixing_debug",
                chunk_size=MIXING_CACHE_CHUNK,
                convention="nominal",
            )

        # ------------------------------------------------------------
        # Finite-difference diagnostic in x at x = y = 0
        # ------------------------------------------------------------
        diag_x_debug = finite_difference_x_diagnostics(
            data_cache=data_cache_debug,
            norm_cache=integration_cache_debug,
            y_fixed=0.0,
            x0=0.0,
            eps=1.0e-4,
            chunk_size=MIXING_NLL_CHUNK,
        )

        try:
            save_json(diag_x_debug, JSON_DIR / "finite_difference_x_diagnostic.json")
        except Exception as exc:
            print("[DEBUG NLL] Could not save finite-difference diagnostic:", exc)


        # ------------------------------------------------------------
        # Fixed points to test.
        # IMPORTANT:
        # Since RUN_MIXING_FIT=0, do not use mixing_result here.
        # The previous seed-16 fit result is hard-coded explicitly.
        # ------------------------------------------------------------
        debug_points = [
            ("truth_zero",       0.000000,  0.000000),
            ("old_fit_result",  -0.005482, -0.000917),
            ("x_offset",        -0.005600,  0.000000),
            ("opposite_x",      +0.005600,  0.000000),
            ("y_offset",         0.000000, -0.000917),
            ("nominal_mixing",  +0.004000, +0.006000),
        ]

        rows = []

        print()
        print("=" * 100)
        print("Fixed-point cached NLL check")
        print("=" * 100)

        for label, x_value, y_value in debug_points:
            print(f"[DEBUG NLL] Evaluating {label}: x = {x_value:.6f}, y = {y_value:.6f}")

            out = evaluate_cached_nll_terms(
                data_cache=data_cache_debug,
                norm_cache=integration_cache_debug,
                x=x_value,
                y=y_value,
                qp_abs=1.0,
                qp_phi=0.0,
                chunk_size=MIXING_NLL_CHUNK,
            )

            out["label"] = label
            rows.append(out)

        min_total = min(row["total_nll"] for row in rows)
        min_data = min(row["data_term"] for row in rows)
        min_norm = min(row["norm_term"] for row in rows)

        print()
        print("=" * 140)
        print("NLL term decomposition")
        print("=" * 140)
        print(
            f"{'label':<18} {'x':>11} {'y':>11} "
            f"{'NLL':>20} {'DeltaNLL':>15} "
            f"{'DeltaData':>15} {'DeltaNorm':>15} "
            f"{'norm_int':>15} {'data_min_pdf':>15} {'norm_min_pdf':>15} "
            f"{'data_bad':>10} {'norm_bad':>10}"
        )
        print("-" * 140)

        for row in rows:
            print(
                f"{row['label']:<18} "
                f"{row['x']:>11.6f} "
                f"{row['y']:>11.6f} "
                f"{row['total_nll']:>20.6f} "
                f"{row['total_nll'] - min_total:>15.6f} "
                f"{row['data_term'] - min_data:>15.6f} "
                f"{row['norm_term'] - min_norm:>15.6f} "
                f"{row['norm_int']:>15.8e} "
                f"{row['data_min_pdf']:>15.8e} "
                f"{row['norm_min_pdf']:>15.8e} "
                f"{row['data_bad_pdf']:>10d} "
                f"{row['norm_bad_pdf']:>10d}"
            )

        print("=" * 140)

        # ------------------------------------------------------------
        # 1D scan in x for the zero-mixing toy
        # ------------------------------------------------------------
        run_x_scan_nll_check(
            data_cache_debug,
            integration_cache_debug,
            y_fixed=0.0,
            x_min=-0.010,
            x_max=0.002,
            n_points=49,
        )

        # ------------------------------------------------------------
        # Score check at x=0, y=0
        # ------------------------------------------------------------
        run_score_check_at_zero(
            data_cache_debug,
            integration_cache_debug,
            eps=1.0e-4,
        )

        # ------------------------------------------------------------
        # Score profile in decay time
        # ------------------------------------------------------------
        run_score_profile_in_time_at_zero(
            data_cache_debug,
            integration_cache_debug,
            eps=1.0e-4,
            n_t_bins=12,
        )

        # ------------------------------------------------------------
        # Score profile in the Dalitz plot
        # ------------------------------------------------------------
        run_score_profile_in_dalitz_at_zero(
            data_np=data_mixing_np,
            data_cache=data_cache_debug,
            norm_np=integration_mixing_test_np,
            norm_cache=integration_cache_debug,
            eps=1.0e-4,
            n_bins_x=50,
            n_bins_y=50,
            output_dir=OUTPUT_DIR,
        )


# ============================================================
# Mixing-fit summary and final outputs
# ============================================================

print_section("Build mixing summary table")

mix_rows = []

for par in ["x_mix", "y_mix", "qp_abs", "qp_phi"]:
    val, err = get_param_value_error(mixing_result, par)

    if par == "qp_phi":
        mix_rows.append({
            "parameter": par,
            "value": val,
            "error": err,
            "value (%) or deg": np.degrees(val),
            "error (%) or deg": np.degrees(err) if np.isfinite(err) else np.nan,
            "floating": par in MIXING_FLOAT_PARAMS,
        })
    elif par in ["x_mix", "y_mix"]:
        mix_rows.append({
            "parameter": par,
            "value": val,
            "error": err,
            "value (%) or deg": 100.0 * val,
            "error (%) or deg": 100.0 * err if np.isfinite(err) else np.nan,
            "floating": par in MIXING_FLOAT_PARAMS,
        })
    else:
        mix_rows.append({
            "parameter": par,
            "value": val,
            "error": err,
            "value (%) or deg": val,
            "error (%) or deg": err,
            "floating": par in MIXING_FLOAT_PARAMS,
        })

df_mixing_summary = pd.DataFrame(mix_rows)
print(df_mixing_summary)

save_dataframe_all_formats(df_mixing_summary, "mixing_summary")
render_table_figure(df_mixing_summary, "table_mixing_summary")

qp_abs_val = get_param_value_error(mixing_result, "qp_abs")[0]
qp_phi_val = get_param_value_error(mixing_result, "qp_phi")[0]
qp_complex = qp_abs_val * np.exp(1j * qp_phi_val)

print(f"q/p = {qp_complex.real:.8f} {qp_complex.imag:+.8f} i")
print(f"|q/p| = {qp_abs_val:.8f}")
print(f"phi(q/p) = {qp_phi_val:.8f} rad = {np.degrees(qp_phi_val):.6f} deg")

mixing_extra = {
    "q_over_p_real": float(qp_complex.real),
    "q_over_p_imag": float(qp_complex.imag),
    "qp_abs": float(qp_abs_val),
    "qp_phi_rad": float(qp_phi_val),
    "qp_phi_deg": float(np.degrees(qp_phi_val)),
    "nll": extract_result_scalar(mixing_result, ["nll", "fval", "fcn", "fun", "value"]),
    "edm": extract_result_scalar(mixing_result, ["edm"]),
    "func_calls": extract_result_scalar(mixing_result, ["func_calls", "nfcn"]),
    "fit_time_sec": extract_result_scalar(mixing_result, ["time"]),
}

save_json(mixing_extra, JSON_DIR / "mixing_extra_summary.json")

write_root_output(
    ROOT_DIR / "fit_results.root",
    df_components=df_components,
    df_mixing_summary=df_mixing_summary,
    dalitz_result=result,
    mixing_result=mixing_result,
)

print_section("Finished")
print("Finished at:", _datetime.datetime.now().isoformat(timespec="seconds"))
print("Outputs written to:", OUTPUT_DIR)

