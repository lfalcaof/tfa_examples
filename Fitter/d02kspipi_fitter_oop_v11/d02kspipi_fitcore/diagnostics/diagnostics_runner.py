from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor


@dataclass
class DiagnosticsRunner:
    """Registers and runs optional NLL, score and amplitude diagnostics.

    In this stage both diagnostic helper functions (former step_11) and the
    optional diagnostic runner (former step_13) are native Python code. The
    implementation still writes the same legacy symbols/results into the shared
    FitContext namespace so the rest of the validated pipeline remains stable.
    """

    context: FitContext
    executor: StepExecutor

    def register_functions(self) -> None:
        """Register optional diagnostic helper functions only when requested.

        The nominal fit path has diagnostics disabled; in that case this method
        is intentionally a no-op apart from publishing the chunk-size legacy
        symbols in the shared namespace. When diagnostics are requested, the
        helper module is executed in a controlled namespace populated from the
        typed configuration and the current FitContext namespace. This avoids
        fragile module-level globals while preserving the old helper function
        names expected by the optional diagnostic runner.
        """
        ns = self.context.namespace

        # Keep the legacy config names available for any remaining code path,
        # but do not overwrite runtime products such as caches or fit results.
        for key, value in self.context.config.as_legacy_globals().items():
            ns.setdefault(key, value)
        ns["MIXING_NLL_CHUNK"] = int(getattr(self.context.config, "mixing_nll_chunk", 250000))
        ns["MIXING_CACHE_CHUNK"] = int(getattr(self.context.config, "mixing_cache_chunk", 250000))

        diagnostics_requested = bool(
            ns.get("RUN_NLL_DEBUG", False)
            or ns.get("RUN_DEBUG_AMPLITUDE_COMPARISON", False)
            or ns.get("RUN_ZERO_MIX_VALIDATION", False)
        )

        if not diagnostics_requested:
            print("[DIAGNOSTICS] Optional diagnostics disabled; helper functions not loaded. [diagnostics_clean]")
            return

        module_path = Path(__file__).with_name("diagnostic_functions.py")
        module_name = f"_d02kspipi_native_diagnostics_{id(self.context)}"

        module_globals = {
            "__name__": module_name,
            "__file__": str(module_path),
            "__package__": "d02kspipi_fitcore.diagnostics",
            "__builtins__": __builtins__,
        }

        module_globals.update(self.context.config.as_legacy_globals())
        module_globals["MIXING_NLL_CHUNK"] = ns["MIXING_NLL_CHUNK"]
        module_globals["MIXING_CACHE_CHUNK"] = ns["MIXING_CACHE_CHUNK"]

        for key, value in ns.items():
            if key.startswith("__") and key.endswith("__"):
                continue
            module_globals[key] = value

        source = module_path.read_text()
        code = compile(source, str(module_path), "exec")
        exec(code, module_globals)

        for name in module_globals["DIAGNOSTIC_FUNCTION_NAMES"]:
            ns[name] = module_globals[name]

        ns["_native_diagnostic_functions_globals"] = module_globals
        print("[DIAGNOSTICS] Registered native diagnostic helper functions. [diagnostics_clean]")

    def run_optional(self) -> None:
        """Run optional diagnostics previously implemented in step_13.

        This method is intentionally a close native transcription of the
        validated step_13_optional_diagnostics.py block. It is executed after
        the mixing fit and before final output summaries. With the usual
        RUN_NLL_DEBUG=0 and RUN_DEBUG_AMPLITUDE_COMPARISON=0 settings it is a
        no-op, matching the validated behavior.
        """
        ns = self.context.namespace

        if ns.get("RUN_DEBUG_AMPLITUDE_COMPARISON", False):
            self._run_amplitude_comparison()

        if ns.get("RUN_NLL_DEBUG", False):
            self._run_nll_debug()

    def _run_amplitude_comparison(self) -> None:
        ns = self.context.namespace
        np = ns["np"]
        atfi = ns["atfi"]

        ns["print_section"]("DEBUG: compare fitter amplitudes with generator amplitudes")

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

        A_fit = ns["amplitude_model"](dalitz_debug_tf)(
            **ns["amp_pars_for_mixing"],
            switches=ns["fit_switches"],
        ).numpy()
        Abar_fit = ns["amplitude_model"](dalitz_swapped_debug_tf)(
            **ns["amp_pars_for_mixing"],
            switches=ns["fit_switches"],
        ).numpy()

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

        ns["save_json"](
            debug_rows,
            ns["JSON_DIR"] / "debug_amplitude_comparison.json",
        )

    def _run_nll_debug(self) -> None:
        ns = self.context.namespace

        ns["print_section"]("DEBUG: fixed-point NLL check for zero-mixing toy")

        if not ns.get("USE_MIXING_AMPLITUDE_CACHE", False):
            print("[DEBUG NLL] This debug block is currently implemented only for cached NLL.")
            print("[DEBUG NLL] Please set USE_MIXING_AMPLITUDE_CACHE=1.")
            return

        print("[DEBUG NLL] Using cached mixing NLL.")

        if "data_mixing_cache" in ns and "integration_mixing_cache" in ns:
            data_cache_debug = ns["data_mixing_cache"]
            integration_cache_debug = ns["integration_mixing_cache"]
            print("[DEBUG NLL] Reusing existing mixing caches.")
        else:
            print("[DEBUG NLL] Mixing caches not found. Building debug caches.")
            data_cache_debug = ns["build_mixing_amplitude_cache"](
                ns["data_mixing_np"],
                label="data_mixing_debug",
                chunk_size=ns["MIXING_CACHE_CHUNK"],
                convention="nominal",
            )
            integration_cache_debug = ns["build_mixing_amplitude_cache"](
                ns["integration_mixing_test_np"],
                label="integration_mixing_debug",
                chunk_size=ns["MIXING_CACHE_CHUNK"],
                convention="nominal",
            )

        diag_x_debug = ns["finite_difference_x_diagnostics"](
            data_cache=data_cache_debug,
            norm_cache=integration_cache_debug,
            y_fixed=0.0,
            x0=0.0,
            eps=1.0e-4,
            chunk_size=ns["MIXING_NLL_CHUNK"],
        )

        try:
            ns["save_json"](
                diag_x_debug,
                ns["JSON_DIR"] / "finite_difference_x_diagnostic.json",
            )
        except Exception as exc:
            print("[DEBUG NLL] Could not save finite-difference diagnostic:", exc)

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

            out = ns["evaluate_cached_nll_terms"](
                data_cache=data_cache_debug,
                norm_cache=integration_cache_debug,
                x=x_value,
                y=y_value,
                qp_abs=1.0,
                qp_phi=0.0,
                chunk_size=ns["MIXING_NLL_CHUNK"],
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

        ns["run_x_scan_nll_check"](
            data_cache_debug,
            integration_cache_debug,
            y_fixed=0.0,
            x_min=-0.010,
            x_max=0.002,
            n_points=49,
        )

        ns["run_score_check_at_zero"](
            data_cache_debug,
            integration_cache_debug,
            eps=1.0e-4,
        )

        ns["run_score_profile_in_time_at_zero"](
            data_cache_debug,
            integration_cache_debug,
            eps=1.0e-4,
            n_t_bins=12,
        )

        ns["run_score_profile_in_dalitz_at_zero"](
            data_np=ns["data_mixing_np"],
            data_cache=data_cache_debug,
            norm_np=ns["integration_mixing_test_np"],
            norm_cache=integration_cache_debug,
            eps=1.0e-4,
            n_bins_x=50,
            n_bins_y=50,
            output_dir=ns["OUTPUT_DIR"],
        )
