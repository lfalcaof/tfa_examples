
import os, time
import numpy as np
import tensorflow as tf
import uproot 


import sys
tfa_root = os.path.abspath(
    os.path.join(os.environ['TFAEX_ROOT'], "..", "TFA2")
)
if tfa_root not in sys.path:
    sys.path.insert(0, tfa_root)

# AmpliTF
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

# TFA
import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo

# Masses & plotting
from particle.particle import literals as lp
import matplotlib.pyplot as plt

# Model
from context import models
from models.d02kspipi import babar2008_model_amp
from models.helpers import decode_model, plot_data, plot_data_mix, plot_data_comparison_mix

# Set up command line argument parsing
import argparse
parser = argparse.ArgumentParser(description="Run toy MC for D0 -> Kspipi (time-dependent DP acceptance)")
parser.add_argument("--ntoys", type=int, default=100000, help="Number of toys to generate")
parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--x", type=float, default=0.004, help="Mixing parameter x")
parser.add_argument("--y", type=float, default=0.0064, help="Mixing parameter y")
parser.add_argument("--qop", type=float, default=1.0, help="|q/p| for CPV in mixing")
parser.add_argument("--qop_phase", type=float, default=0.0, help="arg(q/p) [rad]")
parser.add_argument("--output", type=str, default="d02kspipi_toy_v2", help="Output file stem (without extension)")
parser.add_argument("--dryrun", action="store_true", help="Parse & exit without running")

# ACC switches
parser.add_argument(
    "--use_acceptance",
    action="store_true",
    help="Apply Dalitz×time acceptance from acc_time_dep_v3_dalitz.root"
)
parser.add_argument(
    "--acc_root_dp",
    type=str,
    default="acc_time_dep_v3_dalitz.root",
    help="ROOT file with time-binned Dalitz acceptance TH2F"
)
parser.add_argument(
    "--acc_hist_pattern",
    type=str,
    default="accDP_tbin_{:02d}",
    help="Histogram name pattern for time bins (e.g. accDP_tbin_00 .. accDP_tbin_09)"
)
parser.add_argument(
    "--acc_single_tbin",
    type=int,
    default=None,
    help=(
        "If set (0–9), forces the use of this time-bin index for all events in the acceptance, ignoring the actual t/tal value."
    ),
)

args = parser.parse_args()

# -------------------- constants --------------------
mkz  = atfi.const(lp.K_S_0.mass/1000)
mpi  = atfi.const(lp.pi_plus.mass/1000)
md   = atfi.const(lp.D_0.mass/1000)
meta = atfi.const(lp.eta.mass/1000.)
metap= atfi.const(lp.etap_958.mass/1000.)

belle_model = decode_model(os.environ['TFAEX_ROOT']+'/params/belle_model.txt')

# -------------------- phase spaces --------------------
phsp  = DalitzPhaseSpace(mpi, mkz, mpi, md)
tdz   = atfi.const(1.)
tphsp = DecayTimePhaseSpace(tdz)
c_phsp= CombinedPhaseSpace(phsp, tphsp)

# -------------------- amplitude model --------------------
def babar_model_amp(x):
    return babar2008_model_amp(
        x, phsp,
        atfi.const(belle_model['rho770_Mass'][0]),
        atfi.const(belle_model['rho770_Width'][0]),
        atfi.const(belle_model['Kstar892_Mass'][0]),
        atfi.const(belle_model['Kstar892_Width'][0]),
        atfi.const(belle_model['Kstartwo1430_Mass'][0]),
        atfi.const(belle_model['Kstartwo1430_Width'][0]),
        atfi.const(belle_model['Kstar1410_Mass'][0]),
        atfi.const(belle_model['Kstar1410_Width'][0]),
        atfi.const(belle_model['Kstar1680_Mass'][0]),
        atfi.const(belle_model['Kstar1680_Width'][0]),
        atfi.const(belle_model['omega_Mass'][0]),
        atfi.const(belle_model['omega_Width'][0]),
        atfi.const(belle_model['ftwo1270_Mass'][0]),
        atfi.const(belle_model['ftwo1270_Width'][0]),
        atfi.const(belle_model['rho1450_Mass'][0]),
        atfi.const(belle_model['rho1450_Width'][0]),
        # LASS
        atfi.const(belle_model['LASS_a'][0]),
        atfi.const(belle_model['LASS_r'][0]),
        atfi.const(1.4617),
        atfi.const(0.2683),
        atfi.const(belle_model['LASS_R'][0]),
        atfi.const(belle_model['LASS_phi_R'][0]),
        atfi.const(belle_model['LASS_F'][0]),
        atfi.const(belle_model['LASS_phi_F'][0]),
        # K-matrix
        atfi.const([0.651, 1.2036, 1.55817, 1.21, 1.82206]),
        atfi.const([[0.22889, -0.55377, 0, -0.39899, -0.34639],
                    [0.94128,  0.55095, 0,  0.39065,  0.31503],
                    [0.36856,  0.23888, 0.55639, 0.18340, 0.18681],
                    [0.33650,  0.40907, 0.85679, 0.19906,-0.00984],
                    [0.18171, -0.17558,-0.79658,-0.00355, 0.22358]]),
        atfi.const(-3.92637),
        atfi.const([[ 0.23399, 0.15044,-0.20545, 0.32825, 0.35412],
                    [ 0.15044, 0, 0, 0, 0],
                    [-0.20545, 0, 0, 0, 0],
                    [ 0.32825, 0, 0, 0, 0],
                    [ 0.35412, 0, 0, 0, 0]]),
        atfi.stack([belle_model[f'Kmatrix_beta{i}_realpart'][0]
                    + 1.j*belle_model[f'Kmatrix_beta{i}_imaginarypart'][0]
                    for i in range(1,6)]),
        atfi.const(-0.07),
        atfi.stack([belle_model[f'Kmatrix_f_prod_1{i}_realpart'][0]
                    + 1.j*belle_model[f'Kmatrix_f_prod_1{i}_imaginarypart'][0]
                    for i in range(1,6)]),
        [[mpi,mpi],[mkz,mkz],[mpi],[meta,meta],[meta,metap]]
    )

def Af(x, switches=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]):
    x = tf.reshape(x, (-1, 2))
    return babar_model_amp(x)(
        switches=switches,
        a1r=atfi.const(1.0), a1i=atfi.const(0.0),
        a2r=atfi.const(belle_model['Kstar892minus_realpart'][0]),
        a2i=atfi.const(belle_model['Kstar892minus_imaginarypart'][0]),
        a3r=atfi.const(belle_model['Kstarzero1430minus_realpart'][0]),
        a3i=atfi.const(belle_model['Kstarzero1430minus_imaginarypart'][0]),
        a4r=atfi.const(belle_model['Kstartwo1430minus_realpart'][0]),
        a4i=atfi.const(belle_model['Kstartwo1430minus_imaginarypart'][0]),
        a5r=atfi.const(belle_model['Kstar1410minus_realpart'][0]),
        a5i=atfi.const(belle_model['Kstar1410minus_imaginarypart'][0]),
        a6r=atfi.const(belle_model['Kstar1680minus_realpart'][0]),
        a6i=atfi.const(belle_model['Kstar1680minus_imaginarypart'][0]),
        a7r=atfi.const(belle_model['Kstar892plus_realpart'][0]),
        a7i=atfi.const(belle_model['Kstar892plus_imaginarypart'][0]),
        a8r=atfi.const(belle_model['Kstarzero1430plus_realpart'][0]),
        a8i=atfi.const(belle_model['Kstarzero1430plus_imaginarypart'][0]),
        a9r=atfi.const(belle_model['Kstartwo1430plus_realpart'][0]),
        a9i=atfi.const(belle_model['Kstartwo1430plus_imaginarypart'][0]),
        a10r=atfi.const(belle_model['Kstar1410plus_realpart'][0]),
        a10i=atfi.const(belle_model['Kstar1410plus_imaginarypart'][0]),
        a11r=atfi.const(belle_model['Kstar1680plus_realpart'][0]),
        a11i=atfi.const(belle_model['Kstar1680plus_imaginarypart'][0]),
        a12r=atfi.const(belle_model['omega_realpart'][0]),
        a12i=atfi.const(belle_model['omega_imaginarypart'][0]),
        a13r=atfi.const(belle_model['ftwo1270_realpart'][0]),
        a13i=atfi.const(belle_model['ftwo1270_imaginarypart'][0]),
        a14r=atfi.const(belle_model['rho1450_realpart'][0]),
        a14i=atfi.const(belle_model['rho1450_imaginarypart'][0]),
    )

def Afbar(x):
    x = tf.reshape(x, (-1, 2))
    return Af(x[:, ::-1])




# -------------------- mixing PDF --------------------
def mixing_model(x):
    def _model(x_mix_par, y_mix_par, qoverp_re, qoverp_im):
        x2 = tf.reshape(x, (-1, 3))
        s = c_phsp.data1(x2)         
        t = c_phsp.phsp2.t(c_phsp.data2(x2))  
        ampl_dz  = Af(s)
        ampl_dzb = Afbar(s)
        tep = atfm.psip(t, y_mix_par, atfi.const(1.0))
        tem = atfm.psim(t, y_mix_par, atfi.const(1.0))
        tei = atfm.psii(t, x_mix_par, atfi.const(1.0))

        return atfm.mixing_density(
            ampl_dz, ampl_dzb, atfi.complex(qoverp_re, qoverp_im),
            tep, tem, tei
        )

    return _model

def base_density(x):
    return mixing_model(x)(
        x_mix_par = atfi.const(args.x),
        y_mix_par = atfi.const(args.y),
        qoverp_re = atfi.const(args.qop) * atfi.cos(atfi.const(args.qop_phase)),
        qoverp_im = atfi.const(args.qop) * atfi.sin(atfi.const(args.qop_phase)),
    )

# -------------------- time-binned Dalitz acceptance --------------------
class TimeBinnedDalitzAcceptance:
    """
    Acceptance epsilon(m2_12, m2_13, t/τ) defined by 10 TH2F maps,
    one for each time slice.
    
    t-bins (in units of t/tal):
      bin 0: 0.5  – 1.25  -> accDP_tbin_00
      bin 1: 1.25 – 2.0   -> accDP_tbin_01
      ...
      bin 9: 7.25 – 8.0   -> accDP_tbin_09
    
    If single_tbin is not None, it ALWAYS uses that index (0–9),
    ignoring the actual event t value.
    """
    def __init__(self, root_file, hist_pattern, t_edges, single_tbin=None):
        self.dtype = atfi.fptype()
        self.t_edges = tf.constant(t_edges, dtype=self.dtype)
        self.single_tbin = single_tbin  

        self.hists = []
        with uproot.open(root_file) as f:
            for i in range(len(t_edges) - 1):
                hname = hist_pattern.format(i)
                h = f[hname]
                vals = h.values(flow=False)  
                x_edges = h.axes[0].edges()
                y_edges = h.axes[1].edges()

                self.hists.append({
                    "vals":    tf.constant(vals,    dtype=self.dtype),
                    "x_edges": tf.constant(x_edges, dtype=self.dtype),
                    "y_edges": tf.constant(y_edges, dtype=self.dtype),
                })

        self.n_t_bins = len(self.hists)

    def _weight_single_bin(self, bin_index, s):
        """
        s: [N,2] -> (m2_12, m2_13)
        Returns epsilon(s) for a bin time fixed
        """
        hinfo = self.hists[bin_index]
        vals    = hinfo["vals"]
        x_edges = hinfo["x_edges"]
        y_edges = hinfo["y_edges"]

        x = tf.cast(s[:, 0], self.dtype)
        y = tf.cast(s[:, 1], self.dtype)

        ix = tf.searchsorted(x_edges, x, side='right') - 1
        iy = tf.searchsorted(y_edges, y, side='right') - 1

        nx = tf.shape(vals)[0]
        ny = tf.shape(vals)[1]

        ix = tf.clip_by_value(ix, 0, nx - 1)
        iy = tf.clip_by_value(iy, 0, ny - 1)

        indices = tf.stack([ix, iy], axis=1)  
        w = tf.gather_nd(vals, indices)     
        return w

    def weight(self, s, t):
        """
        s: [N,2] (m2_12, m2_13)
        t: [N,1] (t/tau)
        Returns [N,1] com epsilon(m2_12, m2_13, t)
        """
        s = tf.cast(s, self.dtype)
        t = tf.cast(tf.reshape(t, [-1]), self.dtype)

        if self.single_tbin is not None:
            # Always uses the same bin for all events
            tb = tf.fill(tf.shape(t), tf.cast(self.single_tbin, tf.int32))
        else:
            # default: t_edges[i] <= t < t_edges[i+1]
            tb = tf.searchsorted(self.t_edges, t, side='right') - 1
            tb = tf.clip_by_value(tb, 0, self.n_t_bins - 1)

        w = tf.zeros_like(t, dtype=self.dtype)

        # For each time bin, it computes epsilon(s) and fills only the events belonging to that bin
        for ibin in range(self.n_t_bins):
            mask = tf.where(tf.equal(tb, ibin))[:, 0]  
            n_in_bin = tf.shape(mask)[0]
            w_bin = tf.cond(
                tf.equal(n_in_bin, 0),
                lambda: tf.zeros([0], dtype=self.dtype),
                lambda: self._weight_single_bin(ibin, tf.gather(s, mask))
            )
            w = tf.tensor_scatter_nd_update(
                w,
                tf.expand_dims(mask, 1),
                w_bin
            )

        if self.single_tbin is None:
            tmin = self.t_edges[0]
            tmax = self.t_edges[-1]
            mask_in = tf.logical_and(t >= tmin, t <= tmax)
            w = tf.where(mask_in, w, tf.zeros_like(w))

        return tf.reshape(w, (-1, 1))


# global acceptance instance (constructed only if --use_acceptance)
td_acc = None

def build_acceptance_if_requested():
    global td_acc
    if not args.use_acceptance:
        return

    # edges of the 10 time bins in t/tau:
    # 0: [0.5, 1.25)
    # 1: [1.25, 2.0)
    # 2: [2.0, 2.75)
    # 3: [2.75, 3.5)
    # 4: [3.5, 4.25)
    # 5: [4.25, 5.0)
    # 6: [5.0, 5.75)
    # 7: [5.75, 6.5)
    # 8: [6.5, 7.25)
    # 9: [7.25, 8.0]
    t_edges = [0.5, 1.25, 2.0, 2.75, 3.5, 4.25, 5.0, 5.75, 6.5, 7.25, 8.0]

    # if the user requested a single bin, check that it is in the range 0..9
    if args.acc_single_tbin is not None:
        if not (0 <= args.acc_single_tbin <= len(t_edges) - 2):
            raise ValueError(
                f"--acc_single_tbin={args.acc_single_tbin} fora do range [0, {len(t_edges)-2}]"
            )

    td_acc = TimeBinnedDalitzAcceptance(
        root_file=args.acc_root_dp,
        hist_pattern=args.acc_hist_pattern,
        t_edges=t_edges,
        single_tbin=args.acc_single_tbin,
    )

# -------------------- density with optional acceptance --------------------
def density_with_optional_acc(x):
    # # Wrapper density used by the ToyMC: always returns the physical mixing density,
    # and multiplies it by the (Dalitz × time) acceptance only if --use_acceptance is enabled
    dens = base_density(x)  
    if not args.use_acceptance:
        return dens

    x2 = tf.reshape(x, (-1, 3))

    s = c_phsp.data1(x2)                     
    t = c_phsp.phsp2.t(c_phsp.data2(x2))    

    # Acceptance weight as a function of (s, t)
    w = td_acc.weight(s, t)                 
    dens = tf.reshape(dens, (-1,))
    w    = tf.reshape(w,    (-1,))

    return dens * w

# -------------------- main --------------------
def main():
    if args.seed is not None:
        atfi.set_seed(args.seed)

    build_acceptance_if_requested()

    start_time = time.time()
    toy_sample = tft.run_toymc(
        density_with_optional_acc,
        c_phsp,
        args.ntoys,
        maximum=1.0e-20,
        chunk=1000000,
        components=False,
    )
    end_time = time.time()
    print(f"Generated {args.ntoys} toys in {end_time - start_time:.2f} seconds.")

    out = os.environ['TFAEX_ROOT'] + '/../output/' + args.output + '.npy'
    np.save(out, toy_sample.numpy())
    print(f"Saved: {out}")

if __name__ == "__main__":
    if not args.dryrun:
        main()
