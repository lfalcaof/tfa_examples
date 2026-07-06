from __future__ import annotations

import json
import math
import re
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor


class Tee:
    """Mirror writes to several streams.

    This is the native version of the Tee helper used by the monolithic fitter.
    It keeps the same behavior: every write is flushed immediately both to the
    terminal and to OUTPUT_DIR/logs/run.log.
    """

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class OutputManager:
    """Creates output folders, logging, tables, figures, JSON and ROOT outputs.

    This is the first block converted from preserved step-code to native class
    methods. It still publishes the historical function/global names into the
    FitContext namespace because later validated blocks call names such as
    save_json(...), render_table_figure(...), extract_result_scalar(...), etc.
    """

    context: FitContext
    executor: StepExecutor

    def setup(self) -> None:
        """Create output directories, install logging and register utilities."""
        ns = self.context.namespace

        output_dir = Path(ns["OUTPUT_DIR"])
        fig_dir = output_dir / "figures"
        table_dir = output_dir / "tables"
        json_dir = output_dir / "json"
        array_dir = output_dir / "arrays"
        root_dir = output_dir / "root"
        log_dir = output_dir / "logs"

        for directory in [output_dir, fig_dir, table_dir, json_dir, array_dir, root_dir, log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        ns.update(
            {
                "FIG_DIR": fig_dir,
                "TABLE_DIR": table_dir,
                "JSON_DIR": json_dir,
                "ARRAY_DIR": array_dir,
                "ROOT_DIR": root_dir,
                "LOG_DIR": log_dir,
                "Tee": Tee,
            }
        )

        log_file = open(log_dir / "run.log", "w", buffering=1)
        ns["_log_file"] = log_file
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        self._register_serialization_helpers()
        self._register_root_writer()
        self.print_run_configuration()

    def _register_serialization_helpers(self) -> None:
        ns = self.context.namespace
        table_dir = ns["TABLE_DIR"]
        fig_dir = ns["FIG_DIR"]

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

        def save_dataframe_all_formats(df: Any, stem: str) -> None:
            df.to_csv(table_dir / f"{stem}.csv", index=False)
            df.to_html(table_dir / f"{stem}.html", index=False)
            with open(table_dir / f"{stem}.md", "w", encoding="utf-8") as handle:
                try:
                    handle.write(df.to_markdown(index=False))
                except Exception:
                    handle.write(df.to_string(index=False))
            with open(table_dir / f"{stem}.tex", "w", encoding="utf-8") as handle:
                try:
                    handle.write(df.to_latex(index=False, escape=False))
                except Exception:
                    handle.write(df.to_string(index=False))

        def save_figure(fig: Any, stem: str) -> None:
            fig.savefig(fig_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
            fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")

        def render_table_figure(df: Any, stem: str, max_rows: int = 40) -> None:
            import matplotlib.pyplot as plt
            import numpy as np

            display_df = df.copy()
            if len(display_df) > max_rows:
                display_df = display_df.head(max_rows)

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

        def extract_result_scalar(
            result: Optional[Dict[str, Any]],
            keys: Iterable[str],
            default: float = float("nan"),
        ) -> float:
            """Extract a scalar diagnostic from a tfo/iminuit-style result object."""
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

        ns.update(
            {
                "print_section": print_section,
                "as_float": as_float,
                "jsonable": jsonable,
                "save_json": save_json,
                "save_dataframe_all_formats": save_dataframe_all_formats,
                "save_figure": save_figure,
                "render_table_figure": render_table_figure,
                "extract_result_scalar": extract_result_scalar,
                "get_param_value_error": get_param_value_error,
                "sanitize_tree_name": sanitize_tree_name,
            }
        )

    def _register_root_writer(self) -> None:
        ns = self.context.namespace

        def write_root_output(root_path: Path, df_components: Any, df_mixing_summary: Any, dalitz_result: Any, mixing_result: Any) -> None:
            as_float = ns["as_float"]
            extract_result_scalar = ns["extract_result_scalar"]
            get_param_value_error = ns["get_param_value_error"]
            sanitize_tree_name = ns["sanitize_tree_name"]

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
                                [int(bool(row[df_col])) if df_col == "Fixed" else int(row[df_col])],
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
                    add_int_branch(f"{par}_fixed", 0 if par in ns["MIXING_FLOAT_PARAMS"] else 1)

                add_double_branch("nll", extract_result_scalar(mixing_result, ["nll", "fval", "fcn", "fun", "value"]))
                add_double_branch("edm", extract_result_scalar(mixing_result, ["edm"]))
                add_double_branch("fit_time_sec", extract_result_scalar(mixing_result, ["time"]))
                add_int_branch("func_calls", int(extract_result_scalar(mixing_result, ["func_calls", "nfcn"], default=-1)))
                add_int_branch("nfit_mixing", int(ns["NFIT_MIXING"]))
                add_int_branch("nnorm_mixing", int(ns["NNORM_MIXING"]))

                add_double_branch("dalitz_nll", extract_result_scalar(dalitz_result, ["nll", "fval", "fcn", "fun", "value"]))
                add_double_branch("dalitz_fit_time_sec", extract_result_scalar(dalitz_result, ["time"]))
                add_int_branch("dalitz_func_calls", int(extract_result_scalar(dalitz_result, ["func_calls", "nfcn"], default=-1)))
                add_int_branch("nfit_dalitz", int(ns["NFIT"]))
                add_int_branch("nnorm_dalitz", int(ns["NNORM"]))

                mix_tree.Fill()
                mix_tree.Write()

                root_file.Close()
                print(f"[OUTPUT] ROOT file saved with PyROOT: {root_path}")
                return

            except Exception as exc:
                print(f"[WARNING] PyROOT could not be used: {exc}")
                print("[INFO] Trying uproot fallback for ROOT output.")

            try:
                import numpy as _np
                import uproot

                def one_double(value: Any):
                    return _np.array([as_float(value)], dtype=_np.float64)

                def one_int(value: Any):
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
                        add_int(f"{par}_fixed", 0 if par in ns["MIXING_FLOAT_PARAMS"] else 1)

                    add_double("nll", extract_result_scalar(mixing_result, ["nll", "fval", "fcn", "fun", "value"]))
                    add_double("edm", extract_result_scalar(mixing_result, ["edm"]))
                    add_double("fit_time_sec", extract_result_scalar(mixing_result, ["time"]))
                    add_int("func_calls", int(extract_result_scalar(mixing_result, ["func_calls", "nfcn"], default=-1)))
                    add_int("nfit_mixing", int(ns["NFIT_MIXING"]))
                    add_int("nnorm_mixing", int(ns["NNORM_MIXING"]))

                    add_double("dalitz_nll", extract_result_scalar(dalitz_result, ["nll", "fval", "fcn", "fun", "value"]))
                    add_double("dalitz_fit_time_sec", extract_result_scalar(dalitz_result, ["time"]))
                    add_int("dalitz_func_calls", int(extract_result_scalar(dalitz_result, ["func_calls", "nfcn"], default=-1)))
                    add_int("nfit_dalitz", int(ns["NFIT"]))
                    add_int("nnorm_dalitz", int(ns["NNORM"]))

                    root_file["mixing"] = mix_payload

                print(f"[OUTPUT] ROOT file saved with uproot: {root_path}")

            except Exception as exc:
                print(f"[WARNING] uproot fallback also failed. ROOT output skipped: {exc}")

        ns["write_root_output"] = write_root_output

    def print_run_configuration(self) -> None:
        ns = self.context.namespace
        print_section = ns["print_section"]
        import datetime as _datetime

        print_section("Run configuration")
        print("Started at:", _datetime.datetime.now().isoformat(timespec="seconds"))
        print("FIT_VERSION:", ns["FIT_VERSION"])
        print("BASE_REPO:", ns["BASE_REPO"])
        print("TFAEX_ROOT:", ns["TFAEX_ROOT"])
        print("DATASET:", ns["DATASET"])
        print("NFIT:", ns["NFIT"])
        print("NNORM:", ns["NNORM"])
        print("NFIT_MIXING:", ns["NFIT_MIXING"])
        print("NNORM_MIXING:", ns["NNORM_MIXING"])
        print("USE_MIXING_AMPLITUDE_CACHE:", ns["USE_MIXING_AMPLITUDE_CACHE"])
        print("MIXING_CACHE_CHUNK:", ns["MIXING_CACHE_CHUNK"])
        print("MIXING_NLL_CHUNK:", ns["MIXING_NLL_CHUNK"])
        print("AMP_PARS_MODE:", ns["AMP_PARS_MODE"])
        print("MIXING_FLOAT_PARAMS:", ns["MIXING_FLOAT_PARAMS"])
        print("USE_GPU:", ns["USE_GPU"])
        print("GPU_ID:", ns["GPU_ID"] if ns["GPU_ID"] else "<default visible GPUs>")
        print("OUTPUT_DIR:", ns["OUTPUT_DIR"])

    def write_final_outputs(self) -> None:
        """Build mixing summary tables and write final artifacts natively.

        This replaces the preserved step_14_final_outputs block. It runs only
        after the Dalitz and mixing fit steps have produced df_components,
        result and mixing_result in the shared FitContext namespace. The code
        intentionally preserves the old variable names and output filenames so
        validation against the previous step remains direct.
        """
        ns = self.context.namespace

        np = ns["np"]
        pd = ns["pd"]
        print_section = ns["print_section"]
        get_param_value_error = ns["get_param_value_error"]
        save_dataframe_all_formats = ns["save_dataframe_all_formats"]
        render_table_figure = ns["render_table_figure"]
        save_json = ns["save_json"]
        extract_result_scalar = ns["extract_result_scalar"]
        write_root_output = ns["write_root_output"]

        mixing_result = ns["mixing_result"]
        result = ns["result"]
        df_components = ns["df_components"]
        mixing_float_params = ns["MIXING_FLOAT_PARAMS"]
        json_dir = ns["JSON_DIR"]
        root_dir = ns["ROOT_DIR"]
        output_dir = ns["OUTPUT_DIR"]

        import datetime as _datetime

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
                    "floating": par in mixing_float_params,
                })
            elif par in ["x_mix", "y_mix"]:
                mix_rows.append({
                    "parameter": par,
                    "value": val,
                    "error": err,
                    "value (%) or deg": 100.0 * val,
                    "error (%) or deg": 100.0 * err if np.isfinite(err) else np.nan,
                    "floating": par in mixing_float_params,
                })
            else:
                mix_rows.append({
                    "parameter": par,
                    "value": val,
                    "error": err,
                    "value (%) or deg": val,
                    "error (%) or deg": err,
                    "floating": par in mixing_float_params,
                })

        df_mixing_summary = pd.DataFrame(mix_rows)
        ns["df_mixing_summary"] = df_mixing_summary
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
        ns["mixing_extra"] = mixing_extra

        save_json(mixing_extra, json_dir / "mixing_extra_summary.json")

        write_root_output(
            root_dir / "fit_results.root",
            df_components=df_components,
            df_mixing_summary=df_mixing_summary,
            dalitz_result=result,
            mixing_result=mixing_result,
        )

        print_section("Finished")
        print("Finished at:", _datetime.datetime.now().isoformat(timespec="seconds"))
        print("Outputs written to:", output_dir)

    def save_json(self, obj: Any, path: Path) -> None:
        self.context.require("save_json")(obj, path)

    def save_dataframe_all_formats(self, df: Any, stem: str) -> None:
        self.context.require("save_dataframe_all_formats")(df, stem)

    def save_figure(self, fig: Any, stem: str) -> None:
        self.context.require("save_figure")(fig, stem)
