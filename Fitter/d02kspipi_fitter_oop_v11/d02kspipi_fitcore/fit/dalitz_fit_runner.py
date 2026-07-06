from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor


@dataclass
class DalitzFitRunner:
    """Builds/runs the Dalitz fit and prepares amplitude parameters.

    Native v11 implementation of the old step_07_dalitz_fit block.  The
    validated fitter still expects a number of legacy symbols to exist in the
    shared namespace, so this class writes those symbols back explicitly
    (nll, data_tf_small, pars, fitted_pars, amp_pars_for_mixing,
    df_components, df_summary, etc.).
    """

    context: FitContext
    executor: StepExecutor

    @property
    def ns(self) -> dict[str, Any]:
        return self.context.namespace

    def run(self) -> None:
        self.prepare_likelihood()
        self.build_fit_parameters()
        self.run_or_load_dalitz_result()
        self.prepare_mixing_amplitudes()
        self.build_output_tables()

    def prepare_likelihood(self) -> None:
        ns = self.ns
        atfi = ns["atfi"]
        atfl = ns["atfl"]
        np = ns["np"]

        ns["print_section"]("Prepare Dalitz fit")

        def nll(data, norm):
            data_model = ns["model"](data)
            norm_model = ns["model"](norm)

            @atfi.function
            def _nll(pars):
                return atfl.unbinned_nll(
                    data_model(**pars),
                    atfl.integral(norm_model(**pars)),
                )

            return _nll

        data_np_dalitz = ns["data_np_dalitz"]
        data_tf_small = atfi.const(data_np_dalitz[: min(ns["NFIT"], data_np_dalitz.shape[0])])
        print("data_tf_small shape:", data_tf_small.shape)

        try:
            fig_data, _ = ns["plot_data"](data_tf_small)
            ns["save_figure"](fig_data, "data_dalitz_input")
            import matplotlib.pyplot as plt

            plt.close(fig_data)
            print("[OUTPUT] Saved input Dalitz figure.")
        except Exception as exc:
            print(f"[WARNING] Could not save input Dalitz figure: {exc}")

        ns.update({"nll": nll, "data_tf_small": data_tf_small})

    def choose_reference_component(self):
        ns = self.ns
        active_ids = ns["ACTIVE_COMPONENT_IDS"]
        reference_component = ns["REFERENCE_COMPONENT"]

        if len(active_ids) == 0:
            return None

        if reference_component.lower() == "auto":
            if 1 in active_ids:
                return 1
            return active_ids[0]

        if reference_component.startswith("a"):
            ref_id = int(reference_component[1:])
        else:
            ref_id = int(reference_component)

        if ref_id not in active_ids:
            raise RuntimeError(
                f"REFERENCE_COMPONENT={reference_component} was requested, "
                f"but this component is not active. Active components are: {active_ids}"
            )

        return ref_id

    def build_fit_parameters(self) -> None:
        ns = self.ns
        tfo = ns["tfo"]

        ns["print_section"]("Build Dalitz fit parameters")

        SWITCH = ns["SWITCH"]
        COMPONENTS = ns["COMPONENTS"]
        MODEL_SWITCHES = ns["MODEL_SWITCHES"]

        active_components = [i for i in range(1, 15) if SWITCH[f"a{i}"]]

        if len(active_components) == 0 and not SWITCH["const"]:
            raise RuntimeError("No active amplitude component selected in MODEL_SWITCHES.")

        REFERENCE_COMPONENT_ID = self.choose_reference_component()

        print("Active components:", ns["ACTIVE_COMPONENT_IDS"])
        print("Reference component:", REFERENCE_COMPONENT_ID)

        active_components = ns["ACTIVE_COMPONENT_IDS"]

        FREE: dict[str, bool] = {}
        for comp in COMPONENTS:
            i = comp["id"]
            key = comp["key"]

            is_active = MODEL_SWITCHES[key]
            is_reference = i == REFERENCE_COMPONENT_ID

            FREE[f"a{i}r"] = bool(is_active and not is_reference)
            FREE[f"a{i}i"] = bool(is_active and not is_reference)

        print("Active components:", active_components)
        print("Reference component:", REFERENCE_COMPONENT_ID)

        pars = [
            tfo.FitParameter("a1r", 1.0, -100.0, 100.0),
            tfo.FitParameter("a1i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a2r", 1.5, -100.0, 100.0),
            tfo.FitParameter("a2i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a3r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a3i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a4r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a4i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a5r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a5i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a6r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a6i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a7r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a7i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a8r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a8i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a9r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a9i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a10r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a10i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a11r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a11i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a12r", 2.0, -100.0, 100.0),
            tfo.FitParameter("a12i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a13r", 0.0, -100.0, 100.0),
            tfo.FitParameter("a13i", 0.0, -100.0, 100.0),
            tfo.FitParameter("a14r", 1.0e-2, -100.0, 100.0),
            tfo.FitParameter("a14i", 0.0, -100.0, 100.0),
        ]

        for par in pars:
            if not FREE[par.name]:
                par.fix()

        for par in pars:
            status = "FREE" if FREE[par.name] else "FIXED"
            print(f"{par.name:5s} : {status}")

        ns.update(
            {
                "choose_reference_component": self.choose_reference_component,
                "REFERENCE_COMPONENT_ID": REFERENCE_COMPONENT_ID,
                "active_components": active_components,
                "FREE": FREE,
                "pars": pars,
            }
        )

    def load_source_amplitude_parameters(self):
        ns = self.ns
        from models.helpers import decode_model

        belle_model_path = ns["os"].path.join(ns["TFAEX_ROOT"], "params", "belle_model.txt")
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
            "a14r": 1.0,
            "a14i": 0.0,
        }

    def zero_inactive_amplitude_parameters(self, pars_dict):
        ns = self.ns
        atfi = ns["atfi"]
        out = dict(pars_dict)

        for comp in ns["COMPONENTS"]:
            i = comp["id"]
            key = comp["key"]
            if not ns["MODEL_SWITCHES"][key]:
                out[f"a{i}r"] = atfi.const(0.0)
                out[f"a{i}i"] = atfi.const(0.0)

        return out

    def run_or_load_dalitz_result(self) -> None:
        ns = self.ns
        atfi = ns["atfi"]
        np = ns["np"]
        tfo = ns["tfo"]

        source_values = self.load_source_amplitude_parameters()
        amp_pars_source = {name: atfi.const(float(value)) for name, value in source_values.items()}
        amp_pars_source = self.zero_inactive_amplitude_parameters(amp_pars_source)

        result = None
        fitted_pars = None

        n_free_dalitz_pars = sum(1 for par in ns["pars"] if ns["FREE"][par.name])
        print("Number of free Dalitz parameters:", n_free_dalitz_pars)

        if ns["RUN_DALITZ_FIT"] and n_free_dalitz_pars == 0:
            print("[WARNING] RUN_DALITZ_FIT=1 but there are no free Dalitz parameters.")
            print("[WARNING] Skipping Dalitz MINUIT fit and using source parameters instead.")
            ns["RUN_DALITZ_FIT"] = False

        if ns["RUN_DALITZ_FIT"]:
            ns["print_section"]("Run Dalitz MINUIT fit")

            result = tfo.run_minuit(ns["nll"](ns["data_tf_small"], ns["integration_sample"]), ns["pars"])
            print(result)

            fit_time = ns["extract_result_scalar"](result, ["time"])
            print(f"Total Dalitz fit time: {fit_time:.2f} s")

            fcalls = ns["extract_result_scalar"](
                result,
                ["func_calls", "nfcn"],
                default=float("nan"),
            )
            if fcalls and np.isfinite(fcalls) and fcalls > 0:
                print(f"{fit_time / fcalls:.6f} sec per function call")

            fitted_pars = {p: atfi.const(v[0]) for p, v in result["params"].items()}
            ns["save_json"](result, ns["JSON_DIR"] / "dalitz_fit_result_raw.json")

            fitted_pars = self.zero_inactive_amplitude_parameters(fitted_pars)

            result_clean = dict(result)
            result_clean["params"] = dict(result["params"])

            for name, value in fitted_pars.items():
                value_float = float(value.numpy()) if hasattr(value, "numpy") else float(value)
                old_error = float("nan")
                if name in result["params"]:
                    old_error = result["params"][name][1]
                result_clean["params"][name] = [value_float, old_error]

            ns["save_json"](result_clean, ns["JSON_DIR"] / "dalitz_fit_result.json")
            result = result_clean
        else:
            print("[INFO] RUN_DALITZ_FIT=0. Using source parameters as Dalitz output parameters.")
            fitted_pars = dict(amp_pars_source)
            fitted_pars = self.zero_inactive_amplitude_parameters(fitted_pars)

            result = {
                "params": {
                    name: [
                        float(value.numpy()) if hasattr(value, "numpy") else float(value),
                        float("nan"),
                    ]
                    for name, value in fitted_pars.items()
                }
            }
            ns["save_json"](result, ns["JSON_DIR"] / "dalitz_fit_result.json")

        print("Number of fitted_pars      :", len(fitted_pars))
        print("Number of amp_pars_source :", len(amp_pars_source))
        print("fit_switches =", ns["fit_switches"])
        print("Kmatrix source coefficient:")
        print("  a14r =", source_values["a14r"])
        print("  a14i =", source_values["a14i"])

        ns.update(
            {
                "load_source_amplitude_parameters": self.load_source_amplitude_parameters,
                "zero_inactive_amplitude_parameters": self.zero_inactive_amplitude_parameters,
                "source_values": source_values,
                "amp_pars_source": amp_pars_source,
                "result": result,
                "fitted_pars": fitted_pars,
                "n_free_dalitz_pars": n_free_dalitz_pars,
            }
        )

    def prepare_mixing_amplitudes(self) -> None:
        ns = self.ns

        amp_pars_mode = ns["AMP_PARS_MODE"]
        if amp_pars_mode in ["real_data", "real", "fitted", "data"]:
            if not ns["RUN_DALITZ_FIT"]:
                raise RuntimeError("AMP_PARS_MODE='real_data' requires RUN_DALITZ_FIT=1.")
            amp_pars_for_mixing = dict(ns["fitted_pars"])
        elif amp_pars_mode in ["toys", "toy", "source", "closure"]:
            amp_pars_for_mixing = dict(ns["amp_pars_source"])
        else:
            raise ValueError("AMP_PARS_MODE must be 'real_data' or 'toys'.")

        print()
        print("=" * 80)
        print("Amplitude parameters used for mixing")
        print("=" * 80)
        print("[MIXING AMP] AMP_PARS_MODE =", amp_pars_mode)

        if amp_pars_mode in ["real_data", "real", "fitted", "data"]:
            print("[MIXING AMP] Using fitted Dalitz parameters for mixing.")
        elif amp_pars_mode in ["toys", "toy", "source", "closure"]:
            print("[MIXING AMP] Using source/generator amplitude parameters for mixing.")

        print("[MIXING AMP] Number of parameters =", len(amp_pars_for_mixing))
        print("=" * 80)

        amp_pars_for_dalitz_outputs = dict(ns["fitted_pars"])

        print("AMP_PARS_MODE selected for mixing:", amp_pars_mode)
        ns["save_json"](
            {
                "amp_pars_mode": amp_pars_mode,
                "source_values": ns["source_values"],
                "mixing_float_params": ns["MIXING_FLOAT_PARAMS"],
                "mixing_parameter_config": ns["MIXING_PARAMETER_CONFIG"],
                "use_mixing_amplitude_cache": ns["USE_MIXING_AMPLITUDE_CACHE"],
                "mixing_cache_chunk": ns["MIXING_CACHE_CHUNK"],
                "mixing_nll_chunk": ns["MIXING_NLL_CHUNK"],
            },
            ns["JSON_DIR"] / "configuration_summary.json",
        )

        ns.update(
            {
                "amp_pars_for_mixing": amp_pars_for_mixing,
                "amp_pars_for_dalitz_outputs": amp_pars_for_dalitz_outputs,
            }
        )

    def build_output_tables(self) -> None:
        ns = self.ns
        np = ns["np"]
        pd = ns["pd"]
        tfo = ns["tfo"]

        ns["print_section"]("Build Dalitz output tables")

        def fitted_model(x, switches=ns["fit_switches"]):
            return ns["model"](x)(**ns["amp_pars_for_dalitz_outputs"], switches=switches)

        def mixing_fixed_amplitude_density_model(x, switches=ns["fit_switches"]):
            return ns["model"](x)(**ns["amp_pars_for_mixing"], switches=switches)

        ff_raw = [ns["as_float"](v) for v in tfo.calculate_fit_fractions(fitted_model, ns["integration_sample"])]
        ff = list(ff_raw)

        for comp in ns["COMPONENTS"]:
            i = comp["id"]
            key = comp["key"]
            if not ns["SWITCH"][key]:
                ff[i - 1] = 0.0

        active_components = [comp["id"] for comp in ns["COMPONENTS"] if ns["SWITCH"][comp["key"]]]

        print("Raw fit fractions:")
        print(ff_raw)
        print("Fit fractions after removing inactive components:")
        print(ff)

        if len(active_components) == 1:
            only_id = active_components[0]
            only_ff = ns["as_float"](ff_raw[only_id - 1])
            print("")
            print("[FF CHECK] Single active component:")
            print(f"[FF CHECK] component ID = {only_id}")
            print(f"[FF CHECK] raw FF       = {only_ff:.8f}")
            print(f"[FF CHECK] raw FF (%)   = {100.0 * only_ff:.4f}")
            print("")

        component_map = {comp["id"]: comp["label"] for comp in ns["COMPONENTS"]}

        def is_fixed_component(i):
            return i == ns["REFERENCE_COMPONENT_ID"]

        rows = []
        params_dict = ns["result"]["params"]

        for comp in ns["COMPONENTS"]:
            i = comp["id"]
            key = comp["key"]

            if not ns["MODEL_SWITCHES"][key]:
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
                amp_err = np.sqrt((re_val / amp * re_err) ** 2 + (im_val / amp * im_err) ** 2)
            else:
                amp_err = np.nan

            denom = re_val**2 + im_val**2
            if np.isfinite(re_err) and np.isfinite(im_err) and denom > 0:
                phase_err_rad = np.sqrt(
                    ((-im_val / denom) * re_err) ** 2
                    + ((re_val / denom) * im_err) ** 2
                )
                phase_err_deg = np.degrees(phase_err_rad)
            else:
                phase_err_deg = np.nan

            ff_val = ff[i - 1] if (i - 1) < len(ff) else np.nan
            ff_pct = 100.0 * ff_val if pd.notnull(ff_val) else np.nan

            rows.append(
                {
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
                    "Fixed": is_fixed_component(i),
                }
            )

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
                ff_txt = f"{ff_value:.2f}"

            summary_rows.append(
                {
                    "Resonance": row["Component"],
                    "Amplitude": amp_txt,
                    "Phase (deg)": phase_txt,
                    "Fit fraction (%)": ff_txt,
                }
            )

        df_summary = pd.DataFrame(summary_rows)
        print(df_summary)

        ns["save_dataframe_all_formats"](df_components, "dalitz_components")
        ns["save_dataframe_all_formats"](df_summary, "dalitz_summary")
        ns["render_table_figure"](df_summary, "table_dalitz_summary")
        print("[OUTPUT] Dalitz tables saved.")

        ns.update(
            {
                "fitted_model": fitted_model,
                "mixing_fixed_amplitude_density_model": mixing_fixed_amplitude_density_model,
                "ff_raw": ff_raw,
                "ff": ff,
                "active_components": active_components,
                "component_map": component_map,
                "is_fixed_component": is_fixed_component,
                "rows": rows,
                "df_components": df_components,
                "summary_rows": summary_rows,
                "df_summary": df_summary,
            }
        )

    @property
    def result(self) -> Any:
        return self.context.get("result")

    @property
    def fitted_parameters(self) -> Any:
        return self.context.get("fitted_pars")

    @property
    def amplitude_parameters_for_mixing(self) -> Any:
        return self.context.get("amp_pars_for_mixing")
