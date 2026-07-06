from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor
from d02kspipi_fitcore.fit.mixing_cache import register_mixing_cache_tools


@dataclass
class MixingFitRunner:
    """Prepares mixing samples, cache utilities and the time-dependent fit.

    Native v11 implementation status:
      - prepare_samples() is native and replaces step_09_mixing_samples.
      - cache/NLL tools are native and replace step_10_mixing_cache.
      - run_fit() is native and replaces step_12_mixing_fit.

    The remaining physics model definitions are still supplied by the validated
    upstream steps/native modules and are accessed through FitContext.namespace.
    """

    context: FitContext
    executor: StepExecutor

    @property
    def ns(self) -> dict[str, Any]:
        return self.context.namespace

    @staticmethod
    def build_mixing_array(arr, np, mix_col_m2_kspim: int = 0, mix_col_m2_kspip: int = 1, mix_col_time: int = 2):
        """Build [m2(KS pi-), m2(KS pi+), t/tau] array with finite rows only."""
        m2_kspim = arr[:, mix_col_m2_kspim]
        m2_kspip = arr[:, mix_col_m2_kspip]
        time_tau = arr[:, mix_col_time]

        out = np.column_stack([m2_kspim, m2_kspip, time_tau])
        mask = np.all(np.isfinite(out), axis=1)
        out = out[mask]
        return np.ascontiguousarray(out)

    @staticmethod
    def describe_time(label: str, arr, np) -> None:
        """Print the same decay-time summary used in the v10 monolithic fitter."""
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

    def prepare_samples(self) -> None:
        """Native replacement for step_09_mixing_samples."""
        ns = self.ns
        np = ns["np"]
        atfi = ns["atfi"]
        print_section = ns["print_section"]

        print_section("Prepare mixing fit sample")

        data_np = ns["data_np"]
        integration_np = ns["integration_np"]
        data_np_mixing = ns["data_np_mixing"]
        integration_np_mixing = ns["integration_np_mixing"]

        print("Full data_np shape:", data_np.shape)
        print("Full integration_np shape:", integration_np.shape)

        print("\nFirst 5 rows of data_np:")
        print(data_np[:5])

        print("\nColumn-wise min / max for data_np:")
        for i in range(data_np.shape[1]):
            col = data_np[:, i]
            finite = np.isfinite(col)
            if np.any(finite):
                print(
                    f"col {i:2d}: min = {np.nanmin(col): .6g}, "
                    f"max = {np.nanmax(col): .6g}, mean = {np.nanmean(col): .6g}"
                )
            else:
                print(f"col {i:2d}: no finite entries")

        mix_col_m2_kspim = 0
        mix_col_m2_kspip = 1
        mix_col_time = 2

        ns.update(
            {
                "MIX_COL_M2_KSPIM": mix_col_m2_kspim,
                "MIX_COL_M2_KSPIP": mix_col_m2_kspip,
                "MIX_COL_TIME": mix_col_time,
                "build_mixing_array": lambda arr: self.build_mixing_array(
                    arr,
                    np=np,
                    mix_col_m2_kspim=mix_col_m2_kspim,
                    mix_col_m2_kspip=mix_col_m2_kspip,
                    mix_col_time=mix_col_time,
                ),
                "describe_time": lambda label, arr: self.describe_time(label, arr, np=np),
            }
        )

        data_mixing_all_np = self.build_mixing_array(
            data_np_mixing,
            np=np,
            mix_col_m2_kspim=mix_col_m2_kspim,
            mix_col_m2_kspip=mix_col_m2_kspip,
            mix_col_time=mix_col_time,
        )

        if ns["USE_MIXING_TMAX_CUT"]:
            before = len(data_mixing_all_np)
            data_mixing_all_np = data_mixing_all_np[
                data_mixing_all_np[:, 2] < ns["MIXING_TMAX"]
            ]
            print(
                f"Applied DATA t < {ns['MIXING_TMAX']} cut: "
                f"{before} -> {len(data_mixing_all_np)}"
            )

        nfit_mixing_effective = min(ns["NFIT_MIXING"], len(data_mixing_all_np))
        nnorm_mixing_effective = min(ns["NNORM_MIXING"], integration_np_mixing.shape[0])

        data_mixing_np = np.ascontiguousarray(data_mixing_all_np[:nfit_mixing_effective, :3])
        integration_mixing_test_np = np.ascontiguousarray(
            integration_np_mixing[:nnorm_mixing_effective, :3]
        )

        if ns["USE_MIXING_AMPLITUDE_CACHE"]:
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

        np.save(ns["ARRAY_DIR"] / "data_mixing.npy", data_mixing_np)
        np.save(ns["ARRAY_DIR"] / "integration_mixing.npy", integration_mixing_test_np)

        self.describe_time("DATA MIXING", data_mixing_np, np=np)
        self.describe_time("INTEGRATION MIXING", integration_mixing_test_np, np=np)

        ns.update(
            {
                "data_mixing_all_np": data_mixing_all_np,
                "nfit_mixing_effective": nfit_mixing_effective,
                "nnorm_mixing_effective": nnorm_mixing_effective,
                "data_mixing_np": data_mixing_np,
                "integration_mixing_test_np": integration_mixing_test_np,
                "data_mixing": data_mixing,
                "integration_mixing_test": integration_mixing_test,
            }
        )

    def build_cache_and_nll_tools(self) -> None:
        """Native replacement for step_10_mixing_cache."""
        # Keep legacy config symbols available for closure defaults and runtime lookups.
        for key, value in self.context.config.as_legacy_globals().items():
            self.ns.setdefault(key, value)
        self.ns["MIXING_NLL_CHUNK"] = int(getattr(self.context.config, "mixing_nll_chunk", 250000))
        self.ns["MIXING_CACHE_CHUNK"] = int(getattr(self.context.config, "mixing_cache_chunk", 250000))
        register_mixing_cache_tools(self.ns)

    def register_non_cached_density_tools(self) -> None:
        """Register the non-cached mixing density/NLL utilities.

        These utilities are used only when USE_MIXING_AMPLITUDE_CACHE=0, but
        they are registered unconditionally to preserve the old global API from
        step_12_mixing_fit.
        """
        ns = self.ns
        tf = ns["tf"]
        atfi = ns["atfi"]
        atfm = ns["atfm"]

        def mixing_density_d0_only(
            xdata,
            x_mix,
            y_mix,
            qp_abs,
            qp_phi,
            switches=ns["fit_switches"],
            **amp_pars,
        ):
            x3 = tf.reshape(xdata, (-1, 3))

            dalitz = x3[:, 0:2]
            time_tau = x3[:, 2]

            dalitz_swapped = tf.stack(
                [x3[:, 1], x3[:, 0]],
                axis=1,
            )

            ampl_dz = ns["amplitude_model"](dalitz)(
                **amp_pars,
                switches=switches,
            )

            ampl_dzb = ns["amplitude_model"](dalitz_swapped)(
                **amp_pars,
                switches=switches,
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
                atfi.const(1.0e300),
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
                    atfi.const(1.0e300),
                )

                n_events = tf.cast(tf.shape(data_pdf)[0], atfi.fptype())

                nll_value = -tf.reduce_sum(tf.math.log(data_pdf))
                nll_value += n_events * tf.math.log(norm_int)

                return nll_value

            return _nll_mixing

        ns.update(
            {
                "mixing_density_d0_only": mixing_density_d0_only,
                "nll_mixing": nll_mixing,
            }
        )

    def build_mixing_fit_parameters(self):
        """Native replacement for build_mixing_fit_parameters() from step_12."""
        ns = self.ns
        tfo = ns["tfo"]

        mixing_pars = []

        for name, value in ns["amp_pars_for_mixing"].items():
            v = float(value.numpy()) if hasattr(value, "numpy") else float(value)
            p = tfo.FitParameter(name, v, -100.0, 100.0)
            p.fix()
            mixing_pars.append(p)

        for name in ["x_mix", "y_mix", "qp_abs", "qp_phi"]:
            cfg = ns["MIXING_PARAMETER_CONFIG"][name]
            p = tfo.FitParameter(name, cfg["initial"], cfg["lower"], cfg["upper"])
            if name not in ns["MIXING_FLOAT_PARAMS"]:
                p.fix()
            mixing_pars.append(p)

        print("Parameters used in the mixing fit:")
        for p in mixing_pars:
            try:
                fixed = p.fixed
            except AttributeError:
                fixed = "?"
            print(f"{p.name:12s} fixed = {fixed}")

        ns["build_mixing_fit_parameters"] = self.build_mixing_fit_parameters
        return mixing_pars

    def run_fit(self) -> None:
        """Native replacement for step_12_mixing_fit.

        This keeps the exact validated control flow: build parameters, choose
        cached vs non-cached NLL, sanity-check the callable, run Minuit, save
        JSON, or create a fixed-parameter placeholder when RUN_MIXING_FIT=0.
        """
        ns = self.ns
        for key, value in self.context.config.as_legacy_globals().items():
            ns.setdefault(key, value)
        ns["MIXING_NLL_CHUNK"] = int(getattr(self.context.config, "mixing_nll_chunk", 250000))
        ns["MIXING_CACHE_CHUNK"] = int(getattr(self.context.config, "mixing_cache_chunk", 250000))
        np = ns["np"]
        print_section = ns["print_section"]

        self.register_non_cached_density_tools()
        ns["build_mixing_fit_parameters"] = self.build_mixing_fit_parameters

        mixing_result = None

        if ns["RUN_MIXING_FIT"]:
            print_section("Run mixing MINUIT fit")

            mixing_pars = self.build_mixing_fit_parameters()
            ns["mixing_pars"] = mixing_pars

            if ns["USE_MIXING_AMPLITUDE_CACHE"]:
                print("[CACHE] USE_MIXING_AMPLITUDE_CACHE=1: using v10 cached/chunked mixing NLL.")
                data_mixing_cache = ns["build_mixing_amplitude_cache"](
                    ns["data_mixing_np"],
                    label="data_mixing",
                    chunk_size=ns["MIXING_CACHE_CHUNK"],
                )
                integration_mixing_cache = ns["build_mixing_amplitude_cache"](
                    ns["integration_mixing_test_np"],
                    label="integration_mixing",
                    chunk_size=ns["MIXING_CACHE_CHUNK"],
                )
                mixing_nll_function = ns["nll_mixing_cached"](
                    data_mixing_cache,
                    integration_mixing_cache,
                    chunk_size=ns["MIXING_NLL_CHUNK"],
                )
                ns["data_mixing_cache"] = data_mixing_cache
                ns["integration_mixing_cache"] = integration_mixing_cache
            else:
                print("[CACHE] USE_MIXING_AMPLITUDE_CACHE=0: using original v9 non-cached mixing NLL.")
                mixing_nll_function = ns["nll_mixing"](
                    ns["data_mixing"],
                    ns["integration_mixing_test"],
                )

            ns["mixing_nll_function"] = mixing_nll_function

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

            mixing_result = ns["tfo"].run_minuit(
                mixing_nll_function,
                mixing_pars,
            )

            print(mixing_result)
            ns["save_json"](mixing_result, ns["JSON_DIR"] / "mixing_fit_result.json")
        else:
            print("[INFO] RUN_MIXING_FIT=0. Mixing fit skipped.")
            mixing_result = {
                "params": {
                    "x_mix": [ns["MIXING_PARAMETER_CONFIG"]["x_mix"]["initial"], np.nan],
                    "y_mix": [ns["MIXING_PARAMETER_CONFIG"]["y_mix"]["initial"], np.nan],
                    "qp_abs": [ns["MIXING_PARAMETER_CONFIG"]["qp_abs"]["initial"], np.nan],
                    "qp_phi": [ns["MIXING_PARAMETER_CONFIG"]["qp_phi"]["initial"], np.nan],
                }
            }

        ns["mixing_result"] = mixing_result

    @property
    def result(self) -> Any:
        return self.context.get("mixing_result")

    @property
    def data_cache(self) -> Any:
        return self.context.get("data_mixing_cache")

    @property
    def integration_cache(self) -> Any:
        return self.context.get("integration_mixing_cache")
