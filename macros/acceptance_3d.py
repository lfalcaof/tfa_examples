# acceptance_3d.py
# 3D acceptance based on a TH3F histogram
# Axes convention:
#   X = m2(pipi)
#   Y = cos_theta(pipi)
#   Z = t / tal
#
# Manual trilinear interpolation using bin contents

# Requires PyROOT available in the environment (LCG).
# To load the environment, run:
# source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_108 x86_64-el9-gcc13-opt


import math
import ROOT

class Acceptance3D:
    def __init__(self, root_path, hist_name):
        self.f = ROOT.TFile.Open(root_path)
        if not self.f or self.f.IsZombie():
            raise RuntimeError(f"[Acceptance3D] Could not open file: {root_path}")
        self.h = self.f.Get(hist_name)
        if not self.h:
            raise RuntimeError(f"[Acceptance3D] Histogram not found: {hist_name}")
        if not isinstance(self.h, ROOT.TH3):
            raise TypeError(f"[Acceptance3D] {hist_name} is not a TH3")

        # Cache axes
        self.xa = self.h.GetXaxis()
        self.ya = self.h.GetYaxis()
        self.za = self.h.GetZaxis()

        # Number of bins
        self.nx = self.xa.GetNbins()
        self.ny = self.ya.GetNbins()
        self.nz = self.za.GetNbins()

        # Axis ranges
        self.xmin, self.xmax = self.xa.GetXmin(), self.xa.GetXmax()
        self.ymin, self.ymax = self.ya.GetXmin(), self.ya.GetXmax()
        self.zmin, self.zmax = self.za.GetXmin(), self.za.GetXmax()

        # Maximum value used for normalization
        self.max_val = self.h.GetMaximum()
        if self.max_val <= 0:
            raise RuntimeError(
                "[Acceptance3D] Histogram maximum <= 0; please check contents."
            )

        print("[Acceptance3D] Axis titles:",
              f"X='{self.xa.GetTitle()}', Y='{self.ya.GetTitle()}', Z='{self.za.GetTitle()}'")
        print("[Acceptance3D] Axis ranges:",
              f"X=[{self.xmin:.6g},{self.xmax:.6g}]",
              f"Y=[{self.ymin:.6g},{self.ymax:.6g}]",
              f"Z=[{self.zmin:.6g},{self.zmax:.6g}]")

    @staticmethod
    def _finite_or(v, mid):
        """
        Return v if it is finite, otherwise return the provided midpoint.
        """
        try:
            v = float(v)
        except Exception:
            return mid
        if math.isnan(v) or math.isinf(v):
            return mid
        return v

    def _clamp(self, v, vmin, vmax):
        """
        Clamp value v to the interval [vmin, vmax].
        """
        if v < vmin:
            return vmin
        if v > vmax:
            return vmax
        return v

    def _bin_edges(self, ax, i):
        """
        Return (low, up) bin edges for bin i (1..Nbins).
        """
        low = ax.GetBinLowEdge(i)
        up  = ax.GetBinUpEdge(i)
        return low, up

    def _find_bin_pair(self, ax, v):
        """
        Return (i0, i1, t) for 1D linear interpolation between adjacent bins:
          value ≈ (1 - t) * bin(i0) + t * bin(i1),
        with i0, i1 ∈ [1..Nbins] and t ∈ [0,1].

        Underflow/overflow values are clamped to the nearest valid bin pair.
        """
        nb = ax.GetNbins()

        # Underflow / overflow → clamp to edges
        if v <= ax.GetBinLowEdge(1):
            return 1, 1 if nb == 1 else 2, 0.0
        if v >= ax.GetBinUpEdge(nb):
            return (nb - 1 if nb > 1 else 1), nb, 1.0

        ib = ax.FindFixBin(v)
        if ib < 1:
            ib = 1
        if ib > nb:
            ib = nb

        low, up = self._bin_edges(ax, ib)

        # Protection against invalid bin width
        if up - low <= 0:
            return ib, ib, 0.0

        # If inside bin ib, interpolate between ib and ib+1 (if possible)
        if ib < nb and v < up:
            i0, i1 = ib, ib + 1
            t = (v - low) / (up - low)
            # Note: interpolation is based on distance inside bin ib.
            # If desired, one could instead interpolate between bin centers.
            return i0, i1, max(0.0, min(1.0, t))
        else:
            # Special case: last bin → use ib-1 and ib
            if ib > 1:
                lowm1, upm1 = self._bin_edges(ax, ib - 1)
                t = (v - lowm1) / (up - lowm1)
                return ib - 1, ib, max(0.0, min(1.0, t))
            return ib, ib, 0.0

    def _trilinear(self, x, y, z):
        """
        Manual trilinear interpolation using bin contents.
        """
        ix0, ix1, tx = self._find_bin_pair(self.xa, x)
        iy0, iy1, ty = self._find_bin_pair(self.ya, y)
        iz0, iz1, tz = self._find_bin_pair(self.za, z)

        def C(ix, iy, iz):
            return self.h.GetBinContent(ix, iy, iz)

        # Values at the 8 surrounding vertices
        c000 = C(ix0, iy0, iz0)
        c100 = C(ix1, iy0, iz0)
        c010 = C(ix0, iy1, iz0)
        c110 = C(ix1, iy1, iz0)
        c001 = C(ix0, iy0, iz1)
        c101 = C(ix1, iy0, iz1)
        c011 = C(ix0, iy1, iz1)
        c111 = C(ix1, iy1, iz1)

        # Trilinear interpolation
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0  = c00 * (1 - ty) + c10 * ty
        c1  = c01 * (1 - ty) + c11 * ty

        c   = c0 * (1 - tz) + c1 * tz
        return max(0.0, c)  # enforce non-negative efficiency

    def A(self, m2, cosTh, tOverTau, assume_cos_in_01=True):
        """
        Return normalized efficiency in [0,1] using manual trilinear interpolation.

        - If the acceptance map does not cover cosθ < 0, |cosθ| is used
          (assume_cos_in_01=True).
        - Inputs are safely clamped to [xmin, xmax], etc., before bin lookup.
        """
        # Midpoints used as fallback for invalid inputs
        xm = 0.5 * (self.xmin + self.xmax)
        ym = 0.5 * (self.ymin + self.ymax)
        zm = 0.5 * (self.zmin + self.zmax)

        x = self._finite_or(m2, xm)
        y = self._finite_or(cosTh, ym)
        z = self._finite_or(tOverTau, zm)

        # If Y-axis does not include negative values, use |cosθ|
        if assume_cos_in_01 and self.ymin >= 0.0 and y < 0.0:
            y = -y
        # Alternative mapping could be: y = 0.5 * (y + 1.0)

        # Broad clamping to histogram ranges
        x = self._clamp(x, self.xmin, self.xmax)
        y = self._clamp(y, self.ymin, self.ymax)
        z = self._clamp(z, self.zmin, self.zmax)

        val = self._trilinear(x, y, z)
        return val / self.max_val if self.max_val > 0 else 0.0
