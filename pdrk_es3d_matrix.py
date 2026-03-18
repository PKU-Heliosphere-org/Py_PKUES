"""
pdrk_es3d_matrix.py
Converted from pdrk_es3d_matrix.m
18-10-06 08:23 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Ackn.: Richard Denton (Dartmouth), Xin Tao (USTC), Jin-song Zhao (PMO), etc.
This file sets the pdrk kernel matrix elements for the electrostatic 3D case.
18-10-13 10:37 update to with loss-cone distribution
18-10-20 18:14 electrostatic 3D case
"""

import numpy as np
from scipy import sparse
from scipy.special import ive as besseli_scaled


def pdrk_es3d_matrix(kz, kx, S, N, J, NN, SNJ,
                      rhocsab, vtzs, vds, wcs, kDs,
                      czj, bzj, lmdTab, rsab):
    """
    Build the electrostatic 3D dispersion matrix M.

    Parameters
    ----------
    kz, kx : float
        Parallel and perpendicular wavenumbers [m^-1].
    S : int
        Number of species.
    N : int
        Number of harmonics.
    J : int
        Number of J-poles.
    NN : int
        Total matrix size (= SNJ + 1).
    SNJ : int
        S * (2*N+1) * J.
    rhocsab : ndarray, shape (2, S)
        Cyclotron radii for core/beam.
    vtzs : ndarray, shape (S,)
        Parallel thermal velocities.
    vds : ndarray, shape (S,)
        Parallel drift velocities.
    wcs : ndarray, shape (S,)
        Cyclotron frequencies.
    kDs : ndarray, shape (S,)
        Inverse Debye lengths (1/lambda_D).
    czj, bzj : ndarray, shape (J,)
        J-pole expansion coefficients.
    lmdTab : ndarray, shape (2, S)
        T_z / T_perp for core/beam.
    rsab : ndarray, shape (2, S)
        Core/beam density fractions.

    Returns
    -------
    M : sparse matrix, shape (NN, NN)
        The dispersion eigenvalue matrix.
    """
    k = np.sqrt(kz**2 + kx**2)
    bsab = kx * rhocsab
    bsab[np.abs(bsab) < 1e-50] = 1e-50  # avoid singular when k_perp=0
    bsab2 = bsab**2

    M = sparse.lil_matrix((NN, NN), dtype=complex)
    snj = 0

    csnj = np.zeros(3 * SNJ, dtype=complex)
    bsnj = np.zeros(3 * SNJ, dtype=complex)

    for s in range(S):
        for n in range(-N, N + 1):
            for j in range(J):
                for iab in range(2):
                    Gamn = besseli_scaled(n, bsab2[iab, s])

                    cnj_val = czj[j] * kz * vtzs[s] + kz * vds[s] + n * wcs[s]
                    csnj[snj] = cnj_val

                    bsnj[snj] += (rsab[iab, s] * Gamn * kDs[s]**2 / k**2 *
                                  bzj[j] * (lmdTab[iab, s] * n * wcs[s] +
                                            czj[j] * kz * vtzs[s]))

                snj += 1

    # Assemble the eigenvalue matrix
    for snj_idx in range(SNJ):
        # (n_snj, n_snj) diagonal
        M[snj_idx, snj_idx] += csnj[snj_idx]

        # (n_snj, E) -- last column
        M[snj_idx, NN - 1] += bsnj[snj_idx]

        # (E, n_snj) -- last row
        M[NN - 1, snj_idx] += -csnj[snj_idx]

    # (E, E) -- bottom-right corner
    M[NN - 1, NN - 1] += -np.sum(bsnj[:SNJ])

    return M.tocsr()

