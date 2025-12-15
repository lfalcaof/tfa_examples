# acceptance_3d.py
# Aceitação 3D baseada em TH3F (X=m2(pipi), Y=cosθ(pipi)∈[0,1], Z=t/τ)
# Interpolação trilinear MANUAL via conteúdos dos bins (sem TH3::Interpolate).

# Requer PyROOT carregado no ambiente (LCG).
# Para carregar o ambiente faca 
# source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_108 x86_64-el9-gcc13-opt


import math
import ROOT

class Acceptance3D:
    def __init__(self, root_path, hist_name):
        self.f = ROOT.TFile.Open(root_path)
        if not self.f or self.f.IsZombie():
            raise RuntimeError(f"[Acceptance3D] Não foi possível abrir: {root_path}")
        self.h = self.f.Get(hist_name)
        if not self.h:
            raise RuntimeError(f"[Acceptance3D] Histograma não encontrado: {hist_name}")
        if not isinstance(self.h, ROOT.TH3):
            raise TypeError(f"[Acceptance3D] {hist_name} não é TH3")

        self.xa = self.h.GetXaxis()
        self.ya = self.h.GetYaxis()
        self.za = self.h.GetZaxis()

        self.nx = self.xa.GetNbins()
        self.ny = self.ya.GetNbins()
        self.nz = self.za.GetNbins()

        self.xmin, self.xmax = self.xa.GetXmin(), self.xa.GetXmax()
        self.ymin, self.ymax = self.ya.GetXmin(), self.ya.GetXmax()
        self.zmin, self.zmax = self.za.GetXmin(), self.za.GetXmax()

        self.max_val = self.h.GetMaximum()
        if self.max_val <= 0:
            raise RuntimeError("[Acceptance3D] Histograma com máximo <= 0; verifique conteúdos.")

        print("[Acceptance3D] Axis titles:",
              f"X='{self.xa.GetTitle()}', Y='{self.ya.GetTitle()}', Z='{self.za.GetTitle()}'")
        print("[Acceptance3D] Axes ranges:",
              f"X=[{self.xmin:.6g},{self.xmax:.6g}]",
              f"Y=[{self.ymin:.6g},{self.ymax:.6g}]",
              f"Z=[{self.zmin:.6g},{self.zmax:.6g}]")

    @staticmethod
    def _finite_or(v, mid):
        try:
            v = float(v)
        except Exception:
            return mid
        if math.isnan(v) or math.isinf(v):
            return mid
        return v

    def _clamp(self, v, vmin, vmax):
        if v < vmin: return vmin
        if v > vmax: return vmax
        return v

    def _bin_edges(self, ax, i):
        """Retorna (low, up) das bordas do bin i (1..Nbins)."""
        low = ax.GetBinLowEdge(i)
        up  = ax.GetBinUpEdge(i)
        return low, up

    def _find_bin_pair(self, ax, v):
        """
        Retorna (i0, i1, t) para interpolação linear 1D entre bins adjacentes:
        valor ~ (1-t)*bin(i0) + t*bin(i1), com i0,i1 em [1..Nbins], t∈[0,1].
        Em bordas, cola no último par válido.
        """
        nb = ax.GetNbins()
        # Under/overflow -> cola na borda
        if v <= ax.GetBinLowEdge(1):
            return 1, 1 if nb == 1 else 2, 0.0
        if v >= ax.GetBinUpEdge(nb):
            return (nb-1 if nb > 1 else 1), nb, 1.0

        ib = ax.FindFixBin(v)
        if ib < 1: ib = 1
        if ib > nb: ib = nb
        low, up = self._bin_edges(ax, ib)

        # Se v é “perto” do low/up, calcula com esse bin e o vizinho apropriado
        if up - low <= 0:  # proteção
            return ib, ib, 0.0

        # Se v está no interior do bin ib, interpolamos entre ib e ib+1 (se possível)
        if ib < nb and v < up:
            i0, i1 = ib, ib+1
            low0, up0 = low, up
            low1, up1 = self._bin_edges(ax, i1)
            # t com base em distância dentro do bin ib (aproximação linear)
            t = (v - low0) / (up0 - low0)
            # Se preferir ponderar entre centros dos bins vizinhos, pode-se ajustar aqui.
            return i0, i1, max(0.0, min(1.0, t))
        else:
            # caso especial no último bin: use ib-1 e ib
            if ib > 1:
                lowm1, upm1 = self._bin_edges(ax, ib-1)
                t = (v - lowm1) / (up - lowm1)
                return ib-1, ib, max(0.0, min(1.0, t))
            return ib, ib, 0.0

    def _trilinear(self, x, y, z):
        """
        Interpolação trilinear manual nos conteúdos dos bins.
        Nunca chama TH3::Interpolate -> sem erros de domínio.
        """
        ix0, ix1, tx = self._find_bin_pair(self.xa, x)
        iy0, iy1, ty = self._find_bin_pair(self.ya, y)
        iz0, iz1, tz = self._find_bin_pair(self.za, z)

        def C(ix, iy, iz):
            return self.h.GetBinContent(ix, iy, iz)

        # 8 vértices
        c000 = C(ix0, iy0, iz0)
        c100 = C(ix1, iy0, iz0)
        c010 = C(ix0, iy1, iz0)
        c110 = C(ix1, iy1, iz0)
        c001 = C(ix0, iy0, iz1)
        c101 = C(ix1, iy0, iz1)
        c011 = C(ix0, iy1, iz1)
        c111 = C(ix1, iy1, iz1)

        # Interpolação
        c00 = c000*(1-tx) + c100*tx
        c10 = c010*(1-tx) + c110*tx
        c01 = c001*(1-tx) + c101*tx
        c11 = c011*(1-tx) + c111*tx

        c0  = c00*(1-ty) + c10*ty
        c1  = c01*(1-ty) + c11*ty

        c   = c0*(1-tz) + c1*tz
        return max(0.0, c)  # garante não-negativo

    def A(self, m2, cosTh, tOverTau, assume_cos_in_01=True):
        """
        Retorna eficiência normalizada em [0,1] usando trilinear manual.
        - Se o mapa não cobre cos<0, usa |cos| (assume_cos_in_01=True).
        - Clamps seguros aos ranges [xmin,xmax], etc., antes de achar os bins.
        """
        # sanitização + midpoints
        xm = 0.5*(self.xmin + self.xmax)
        ym = 0.5*(self.ymin + self.ymax)
        zm = 0.5*(self.zmin + self.zmax)

        x = self._finite_or(m2, xm)
        y = self._finite_or(cosTh, ym)
        z = self._finite_or(tOverTau, zm)

        # se Y não cobre negativos, use |cos|
        if assume_cos_in_01 and self.ymin >= 0.0 and y < 0.0:
            y = -y
        # (alternativa: y = 0.5*(y+1.0))

        # clamp “amplo” aos ranges do hist
        x = self._clamp(x, self.xmin, self.xmax)
        y = self._clamp(y, self.ymin, self.ymax)
        z = self._clamp(z, self.zmin, self.zmax)

        val = self._trilinear(x, y, z)
        return val / self.max_val if self.max_val > 0 else 0.0
