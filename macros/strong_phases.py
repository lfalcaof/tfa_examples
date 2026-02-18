"""This macros calculates the strong phases in a given binning scenario for a toy sample
generated according to the Belle model for D0 -> Kspipi.
"""

import os, time, uproot
# import NumPy
import numpy as np
# Import Tensorflow
import tensorflow as tf
# Import AmpliTF modules
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace
#from amplitf.mixing import psip, psim, psii, mixing_density

# Import TFA modules
import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo

# Masses of final state particles
from particle.particle import literals as lp

# Import plotting module
import matplotlib.pyplot as plt

#from context import models
from context import models
from models.d02kspipi import babar2008_model_amp
from models.helpers import decode_model, plot_data, plot_data_mix, plot_data_comparison_mix

# from binflipfitter import add_binflip_binid
from helpers import add_binflip_binid

# Import argparse for command line arguments
import argparse
# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Calculate strong phases for D0 -> Kspipi")
parser.add_argument("--input", type=str, default=None, help="File name for the sample to calculate strong phases for", required=True)
#parser.add_argument("--binning", type=str, default=None, help="Binning scheme to use for the calculation of strong phases")
#parser.add_argument("--output", type=str, default='cb_', help="Output file name for the strong phases results")
parser.add_argument("--dryrun", action="store_true", help="Run without executing the main function")
args = parser.parse_args()

# define some global variables 
mkz = atfi.const(lp.K_S_0.mass/1000)
mpi = atfi.const(lp.pi_plus.mass/1000)
md = atfi.const(lp.D_0.mass/1000)
meta = atfi.const(lp.eta.mass/1000.)
metap = atfi.const(lp.etap_958.mass/1000.)
belle_model = decode_model(os.environ['TFAEX_ROOT']+'/params/belle_model.txt')

#Define the phase space for the decay D0 -> Kspipi.
# Create Dalitz Phase Space
phsp = DalitzPhaseSpace(mpi, mkz, mpi, md)

def babar_model_amp(x):
    return babar2008_model_amp(x, phsp,
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
        # K matrix model parameters
        atfi.const( [0.651, 1.2036, 1.55817, 1.21, 1.82206] ),
        atfi.const( [ [0.22889, -0.55377, 0, -0.39899, -0.34639],
                        [0.94128, 0.55095, 0, 0.39065, 0.31503],
                        [0.36856, 0.23888, 0.55639, 0.18340, 0.18681],
                        [0.33650, 0.40907, 0.85679, 0.19906, -0.00984],
                        [0.18171, -0.17558, -0.79658, -0.00355, 0.22358]] ),
        atfi.const(-3.92637),
        atfi.const([ [  0.23399,  0.15044, -0.20545,  0.32825,  0.35412],
                   [  0.15044, 0, 0, 0, 0],
                   [ -0.20545, 0, 0, 0, 0],
                   [  0.32825, 0, 0, 0, 0],
                   [  0.35412, 0, 0, 0, 0]]),
        atfi.stack([belle_model[f'Kmatrix_beta{i}_realpart'][0]+\
                    1.j*belle_model[f'Kmatrix_beta{i}_imaginarypart'][0] for i in range(1, 6)]),
        atfi.const(-0.070000000000000),
        atfi.stack([belle_model[f'Kmatrix_f_prod_1{i}_realpart'][0]+\
                    1.j*belle_model[f'Kmatrix_f_prod_1{i}_imaginarypart'][0] for i in range(1, 6)]),
        [[mpi,mpi], [mkz, mkz], [mpi], [meta, meta], [meta, metap]])

def Af(x, switches=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]):#15 * [1]):
    return babar_model_amp(x)(
        switches=switches,
        a1r=atfi.const(1.0), 
        a1i=atfi.const(0.0),
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
    # the conjugate of the amplitude
    # is the same as the amplitude with the masses swapped
    return Af(x[:,::-1])

def get_model_cache(model, xbins, ybins):
    # Create a grid of points at the center of each bin
    x_centers = (xbins[:-1] + xbins[1:]) / 2
    y_centers = (ybins[:-1] + ybins[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    grid_points = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
    # Filter points inside the phase space
    inside_mask = phsp.inside(grid_points)
    inside_indices = np.where(inside_mask)[0]
    
    # Initialize output arrays with zeros
    amp_full = np.zeros(len(grid_points))
    amp2_full = np.zeros(len(grid_points))
    phase_full = np.zeros(len(grid_points))
    
    # Calculate the amplitude and phase only for points inside the phase space
    if len(inside_indices) > 0:
        grid_points_inside = grid_points[inside_indices]
        amp_inside, amp2_inside, phase_inside = get_amplitude_and_phase_from_model(grid_points_inside, model)
        amp_full[inside_indices] = amp_inside
        amp2_full[inside_indices] = amp2_inside
        phase_full[inside_indices] = phase_inside
    
    return amp_full.reshape(x_grid.shape), amp2_full.reshape(x_grid.shape), phase_full.reshape(x_grid.shape), xbins, ybins


def get_belle_cache():
    belle_cache = uproot.open(os.environ['TFAEX_ROOT']+'/../d02kspipi_toys/generator/inputs/belle_cache.root')
    amp, xbins, ybins = belle_cache['histo_amplitude'].to_numpy()
    amp2, _, _ = belle_cache['histo_amplitude_squared'].to_numpy()
    phase, _, _ = belle_cache['histo_phase'].to_numpy()
    return amp, amp2, phase, xbins, ybins


def get_amplitude_and_phase_from_model(x, model):
    # Calculate the amplitude on  data
    phsp_data = x[:, :2]
    a = model(phsp_data)
    amp2 = atfi.real(a * atfi.conjugate(a)).numpy()
    amp = atfi.sqrt(amp2)
    phase = atfi.atan2(atfi.imaginary(a), atfi.real(a)).numpy()
    return amp, amp2, phase


def get_strong_phase_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    strong_phases = phase[y_bin_indices, x_bin_indices]
    return strong_phases


def get_delta_strong_phase_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    delta_strong_phases = phase[y_bin_indices, x_bin_indices] - phase[x_bin_indices, y_bin_indices]  # Subtract the phase of the same bin (this is a placeholder for actual logic)
    return delta_strong_phases


def get_amp_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    amp_values = amp[y_bin_indices, x_bin_indices]
    return amp_values


def get_ampsq_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    amp2_values = amp2[y_bin_indices, x_bin_indices]
    return amp2_values


def plot_2d_binned_array(array_2d, xbins, ybins, xlabel="X", ylabel="Y", title="2D Binned Data", cmap='RdBu_r', vcenter=0):
    """Plot a 2D array with bin edges as axes.
    
    Args:
        array_2d: 2D numpy array with shape (len(ybins)-1, len(xbins)-1)
        xbins: bin edges for x-axis
        ybins: bin edges for y-axis
        xlabel: label for x-axis
        ylabel: label for y-axis
        title: plot title
        cmap: colormap name (default 'RdBu_r' for diverging colormap)
        vcenter: value mapped to center of colormap (white) - default 0
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    from matplotlib.colors import TwoSlopeNorm, Normalize
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin, vmax = np.nanmin(array_2d), np.nanmax(array_2d)
    
    # Check if data contains both positive and negative values
    has_positive = vmax > vcenter
    has_negative = vmin < vcenter
    
    if has_positive and has_negative:
        # Use diverging colormap with white at vcenter (mixed signs)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        used_cmap = cmap
    elif has_positive:
        # All positive: white at 0, gradient to blue
        norm = Normalize(vmin=vcenter, vmax=vmax)
        used_cmap = 'Blues'
    else:
        # All negative: gradient from blue to white at 0
        norm = Normalize(vmin=vmin, vmax=vcenter)
        used_cmap = 'Blues_r'
    
    mesh = ax.pcolormesh(xbins, ybins, array_2d, cmap=used_cmap, norm=norm, shading='auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = plt.colorbar(mesh, ax=ax)
    return fig, ax


def plot_strong_phase_by_bin(x, delta_strong):
    sort_idx = np.argsort(x[:, 2])
    x_sorted = x[sort_idx]
    ds_sorted = delta_strong[sort_idx]
    bin_ids = x_sorted[:, 2].astype(int)
    unique_bins, bin_starts = np.unique(bin_ids, return_index=True)
    bin_starts = np.append(bin_starts, len(bin_ids))
    fig, ax = plt.subplots()
    for i in range(8):
        ax.hist(ds_sorted[bin_starts[i]:bin_starts[i+1]], bins=100, range=(-np.pi,np.pi), alpha=0.5)
    ax.set_xlabel("Strong Phase Difference")
    ax.set_ylabel("Events")
    ax.set_title("Strong Phase Differences by Bin")
    #plt.savefig("strong_phase_by_bin.png")
    return fig, ax


def calculate_strong_phases(x):
    r"""Calculate the :math:`c_b` and :math:`s_b` strong phases average difference in the binflip bins by using the formulas

    ..math::

        \begin{align}
        c_i &= \frac{1}{F_i F_{-i}} \int_i |A_{D^0}(m^2_+, m^2_-)| |A_{D^0}(m^2_-, m^2_+)| \cos\left(\Delta\delta_D(m^2_+, m^2_-)\right) dm^2_+ dm^2_-
        s_i &= \frac{1}{F_i F_{-i}} \int_i |A_{D^0}(m^2_+, m^2_-)| |A_{D^0}(m^2_-, m^2_+)| \sin\left(\Delta\delta_D(m^2_+, m^2_-)\right) dm^2_+ dm^2_-
        \end{align}

    Args:
        x (float, float, int): data point with (m^2_+, m^2_-, bin_id)

    Returns:
        array: array with the strong phase differences for each bin
    """    
    # Sort data by bin_id for faster processing
    sort_idx = np.argsort(x[:, 3])
    x_sorted = x[sort_idx]
    
    # Calculate the amplitude on sorted data
    phsp_data = x_sorted[:, :2]
    ampl_dz = Af(phsp_data)
    ampl_dzb = Afbar(phsp_data)
    
    # Calculate phase difference: delta = arg(A_D0) - arg(A_D0bar)
    # This is equivalent to arg(A_D0 / A_D0bar) or arg(A_D0 * conj(A_D0bar))
    #delta_strong = atfi.atan2(atfi.imaginary(ampl_dz * atfi.conjugate(ampl_dzb)), 
    #                          atfi.real(ampl_dz * atfi.conjugate(ampl_dzb)))
    delta_strong = atfi.atan2(atfi.imaginary(ampl_dz), atfi.real(ampl_dz)) - atfi.atan2(atfi.imaginary(ampl_dzb), atfi.real(ampl_dzb))
    
    # Convert to numpy for faster bin operations
    delta_strong_np = delta_strong.numpy()
    bin_ids = x_sorted[:, 3].astype(int)
    
    # Calculate cb and sb for the binning scheme using vectorized operations
    unique_bins, bin_starts = np.unique(bin_ids, return_index=True)
    bin_starts = np.append(bin_starts, len(bin_ids))
    
    N = x.shape[0]
    for i in range(1, 9):
        Fpi = (bin_starts[8+i+1]-bin_starts[8+i])/N
        Fmi = (bin_starts[8-i+1]-bin_starts[8-i])/N
        Ddd = delta_strong_np[bin_starts[8+i]:bin_starts[8+i+1]]
        Adz = np.abs(ampl_dz.numpy()[bin_starts[8+i]:bin_starts[8+i+1]])/(Fpi+Fmi)/N
        Adzb = np.abs(ampl_dzb.numpy()[bin_starts[8+i]:bin_starts[8+i+1]])/(Fpi+Fmi)/N
        ci = np.sum(Adz * Adzb * np.cos(Ddd)) / np.sqrt(Fpi * Fmi)
        si = np.sum(Adz * Adzb * np.sin(Ddd)) / np.sqrt(Fpi * Fmi)
        print(f"Bin {i}: c = {ci:.6f}, s = {si:.6f}, N+ = {Fpi}, N- = {Fmi}") # NORMALISATION ISSUE
    
    # Return in original order
    delta_strong_original_order = np.empty_like(delta_strong_np)
    delta_strong_original_order[sort_idx] = delta_strong_np
    
    return delta_strong_original_order


def load_data():
    data = np.load(args.input)
    data = add_binflip_binid(data)
    return data


def get_uniform_phase_space_sample(n_events):
    data = phsp.uniform_sample(n_events)
    time = np.random.exponential(scale=1.0, size=len(data))
    data = np.insert(data, 2, time, axis=1)
    data = add_binflip_binid(data)
    return data


def main():

    # Compare the strong phases calculated from the cache with those calculated from the model
    belle_cache = get_belle_cache()
    xbins = belle_cache[3]
    ybins = belle_cache[4]
    model_cache = get_model_cache(Afbar, xbins, ybins) # need to swap x and y bins for the model cache to match the belle cache
    figs, axs = [],[]
    titles = ["Amplitude Difference (Model - Belle Cache)", "Amplitude Squared Difference (Model - Belle Cache)", "Phase Difference (Model - Belle Cache)"]
    for i in range(3):
        fig, ax = plot_2d_binned_array(model_cache[i]-belle_cache[i], xbins, ybins, title=titles[i], xlabel="$m^2_+$", ylabel="$m^2_-$")
        figs.append(fig)
        axs.append(ax)
    plt.show()

    # data = load_data()
    
    # # Calculate strong phase differences
    # print(f"Calculating strong phases for {len(data)} events...")
    # delta_strong_np = calculate_strong_phases(data)
    
    # # Print statistics
    # print(f"\nStrong phase difference statistics:")
    # print(f"  Mean: {np.mean(delta_strong_np):.4f} rad ({np.degrees(np.mean(delta_strong_np)):.2f}°)")
    # print(f"  Std Dev: {np.std(delta_strong_np):.4f} rad ({np.degrees(np.std(delta_strong_np)):.2f}°)")
    # print(f"  Min: {np.min(delta_strong_np):.4f} rad ({np.degrees(np.min(delta_strong_np)):.2f}°)")
    # print(f"  Max: {np.max(delta_strong_np):.4f} rad ({np.degrees(np.max(delta_strong_np)):.2f}°)")
    
    # # Optionally save results
    # if args.binning:
    #     output_file = f"strong_phases_{args.binning}.npy"
    #     np.save(output_file, delta_strong_np)
    #     print(f"\nSaved strong phases to {output_file}")
    
    return 

if __name__ == "__main__":
    if not args.dryrun:
        main()


### TODO:
# Add a function to apply reconstruction efficiencies to the toy MC sample
# Add a function to generate generator-level data on-demand