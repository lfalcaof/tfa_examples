import os
import numpy as np
import uproot


def add_binflip_binid(data):
    hf = uproot.open(os.environ['TFAEX_ROOT']+'/notebooks/files/BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD.root')
    hnp = hf['h_BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD'].to_numpy()
    ix = np.digitize(data[:,0], hnp[1]) - 1
    iy = np.digitize(data[:,1], hnp[2]) - 1
    binid = hnp[0][ix, iy] * (2 * (ix>iy) - 1)  # Flip sign if ix > iy
    data = np.column_stack((data, binid))
    return data