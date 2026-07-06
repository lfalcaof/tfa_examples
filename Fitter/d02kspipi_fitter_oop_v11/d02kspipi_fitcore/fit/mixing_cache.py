from __future__ import annotations

from typing import Any


def register_mixing_cache_tools(ns: dict[str, Any]) -> None:
    """Register cached/chunked mixing likelihood utilities in the fit context.

    Native replacement for steps/step_10_mixing_cache.py.

    The functions are intentionally registered back into the legacy namespace
    because the still-preserved mixing fit block and the native diagnostics
    helpers expect these names to exist globally.
    """

    np = ns["np"]
    tf = ns["tf"]
    atfi = ns["atfi"]
    atfm = ns["atfm"]

    def build_mixing_amplitude_cache(
        x_np,
        label: str,
        chunk_size=None,
        convention: str = "nominal",
    ):
        """Precompute A(D0), A(D0bar) and t for the mixing fit.

        The Dalitz amplitudes depend only on the phase-space point and fixed
        amplitude parameters. They do not change while Minuit scans x, y,
        |q/p| and phi. The returned arrays are NumPy arrays kept on CPU memory.
        During the NLL evaluation, only one chunk is converted to TensorFlow at
        a time.
        """

        if chunk_size is None:
            chunk_size = int(ns.get("MIXING_CACHE_CHUNK", 250000))

        x_np = np.ascontiguousarray(x_np[:, :3])
        n_events = len(x_np)

        print(f"[CACHE] Building mixing amplitude cache for {label}")
        print(f"[CACHE] {label}: events = {n_events}")
        print(f"[CACHE] {label}: chunk size = {chunk_size}")

        amp_chunks = []
        ampbar_chunks = []
        time_chunks = []

        amplitude_model = ns["amplitude_model"]
        amp_pars_for_mixing = ns["amp_pars_for_mixing"]
        fit_switches = ns["fit_switches"]

        for start in range(0, n_events, chunk_size):
            stop = min(start + chunk_size, n_events)
            print(f"[CACHE] {label}: {start} -> {stop}")

            chunk = x_np[start:stop]

            dalitz = atfi.const(chunk[:, 0:2])
            dalitz_swapped = atfi.const(chunk[:, [1, 0]])

            amp_nominal = amplitude_model(dalitz)(
                **amp_pars_for_mixing,
                switches=fit_switches,
            ).numpy()

            amp_swapped = amplitude_model(dalitz_swapped)(
                **amp_pars_for_mixing,
                switches=fit_switches,
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

        print(
            f"[CACHE] Finished {label}: "
            f"A shape = {cache['A'].shape}, Abar shape = {cache['Abar'].shape}"
        )
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

    def nll_mixing_cached(
        data_cache,
        norm_cache,
        chunk_size=None,
    ):
        """Chunked NLL for the mixing fit using cached amplitudes."""

        if chunk_size is None:
            chunk_size = int(ns.get("MIXING_NLL_CHUNK", 250000))

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

    ns.update(
        {
            "build_mixing_amplitude_cache": build_mixing_amplitude_cache,
            "mixing_density_cached_amplitudes": mixing_density_cached_amplitudes,
            "nll_mixing_cached": nll_mixing_cached,
        }
    )
