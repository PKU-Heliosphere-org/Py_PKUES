"""
pdrk_em3d_matrix.py
Converted from pdrk_em3d_matrix.m
18-10-06 08:23 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Ackn.: Richard Denton (Dartmouth), Xin Tao (USTC), Jin-song Zhao (PMO), etc.
This file sets the pdrk kernel matrix elements for the electromagnetic 3D case.
18-10-13 10:37 update to with loss-cone distribution
18-10-21 17:59 fixed a bug of rsab(iab), which should be rsab(iab,s)
"""

import numpy as np
from scipy import sparse
from scipy.special import ive as besseli_scaled  # besseli(n, x, 1) = ive(n, x)


def pdrk_em3d_matrix(kz, kx, S, N, J, NN, SNJ, SNJ1, SNJ3,
                      rhocsab, vtzs, vds, wcs, wps2,
                      czj, bzj, lmdTab, rsab, c2):
    """
    Build the electromagnetic 3D dispersion matrix M.

    The eigenvalue problem is: lambda * X = M * X,
    where lambda = omega (complex frequency).

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
        Total matrix size.
    SNJ : int
        S * (2*N+1) * J.
    SNJ1 : int
        SNJ + 1.
    SNJ3 : int
        3 * SNJ1.
    rhocsab : ndarray, shape (2, S)
        Cyclotron radii for core/beam.
    vtzs : ndarray, shape (S,)
        Parallel thermal velocities.
    vds : ndarray, shape (S,)
        Parallel drift velocities.
    wcs : ndarray, shape (S,)
        Cyclotron frequencies.
    wps2 : ndarray, shape (S,)
        Squared plasma frequencies.
    czj, bzj : ndarray, shape (J,)
        J-pole expansion coefficients.
    lmdTab : ndarray, shape (2, S)
        Temperature anisotropy ratios T_z / T_perp for core/beam.
    rsab : ndarray, shape (2, S)
        Core/beam density fractions.
    c2 : float
        Speed of light squared.

    Returns
    -------
    M : sparse matrix, shape (NN, NN)
        The dispersion eigenvalue matrix.
    """
    k = np.sqrt(kz**2 + kx**2)
    bsab = kx * rhocsab                  # dimensionless, shape (2, S)
    bsab[np.abs(bsab) < 1e-50] = 1e-50  # avoid singular when k_perp=0
    bsab2 = bsab**2

    # Use lil_matrix for efficient element-by-element construction
    M = sparse.lil_matrix((NN, NN), dtype=complex)
    snj = 0  # 0-based counter

    # Accumulated b-terms (scalar sums over all species/harmonics/poles)
    b11 = 0.0 + 0j
    b12 = 0.0 + 0j
    b13 = 0.0 + 0j
    b21 = 0.0 + 0j
    b22 = 0.0 + 0j
    b23 = 0.0 + 0j
    b31 = 0.0 + 0j
    b32 = 0.0 + 0j
    b33 = 0.0 + 0j

    # Per-(s,n,j) arrays
    csnj = np.zeros(3 * SNJ, dtype=complex)   # MATLAB uses 3*SNJ but only SNJ entries used
    b11snj = np.zeros(3 * SNJ, dtype=complex)
    b12snj = np.zeros(3 * SNJ, dtype=complex)
    b13snj = np.zeros(3 * SNJ, dtype=complex)
    b21snj = np.zeros(3 * SNJ, dtype=complex)
    b22snj = np.zeros(3 * SNJ, dtype=complex)
    b23snj = np.zeros(3 * SNJ, dtype=complex)
    b31snj = np.zeros(3 * SNJ, dtype=complex)
    b32snj = np.zeros(3 * SNJ, dtype=complex)
    b33snj = np.zeros(3 * SNJ, dtype=complex)

    for s in range(S):
        for n in range(-N, N + 1):
            for j in range(J):
                # snj is 0-based; in MATLAB it starts from 1
                for iab in range(2):  # core (0) and beam (1)
                    # Scaled Bessel functions
                    Gamn = besseli_scaled(n, bsab2[iab, s])
                    Gamnp = (besseli_scaled(n + 1, bsab2[iab, s]) +
                             besseli_scaled(n - 1, bsab2[iab, s]) -
                             2 * besseli_scaled(n, bsab2[iab, s])) / 2

                    cnj_val = czj[j] * kz * vtzs[s] + kz * vds[s] + n * wcs[s]
                    csnj[snj] = cnj_val
                    bj0ab = vds[s] + (1 - 1.0 / lmdTab[iab, s]) * czj[j] * vtzs[s]

                    # --- For A_nj ---
                    if n == 0:
                        bnj1 = bj0ab / (czj[j] * vtzs[s] + vds[s])  # avoid cnj=0
                    else:
                        bnj1 = kz * bj0ab / cnj_val
                    bnj2 = 1.0 - bnj1

                    tmp = wps2[s] * bzj[j]

                    b11snj[snj] += rsab[iab, s] * tmp * bnj2 * n**2 * Gamn / bsab2[iab, s]
                    b11 += rsab[iab, s] * tmp * bnj1 * n**2 * Gamn / bsab2[iab, s]

                    b12snj[snj] += rsab[iab, s] * tmp * bnj2 * 1j * n * Gamnp
                    b12 += rsab[iab, s] * tmp * bnj1 * 1j * n * Gamnp
                    b21snj[snj] = -b12snj[snj]
                    b21 = -b12

                    b22snj[snj] += rsab[iab, s] * tmp * bnj2 * (
                        n**2 * Gamn / bsab2[iab, s] - 2 * bsab2[iab, s] * Gamnp)
                    b22 += rsab[iab, s] * tmp * bnj1 * (
                        n**2 * Gamn / bsab2[iab, s] - 2 * bsab2[iab, s] * Gamnp)

                    # --- For eta_n * A_nj ---
                    if n == 0:
                        bnj1_eta = 0.0  # avoid cnj=0 when kz=0
                    else:
                        bnj1_eta = n * wcs[s] * bj0ab / cnj_val / vtzs[s]
                    bnj2_eta = czj[j] / lmdTab[iab, s] + bnj1_eta

                    b13snj[snj] += rsab[iab, s] * tmp * bnj2_eta * n * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamn / bsab[iab, s]
                    b13 += -rsab[iab, s] * tmp * bnj1_eta * n * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamn / bsab[iab, s]
                    b31snj[snj] = b13snj[snj]
                    b31 = b13

                    b23snj[snj] += -rsab[iab, s] * 1j * tmp * bnj2_eta * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamnp * bsab[iab, s]
                    b23 += rsab[iab, s] * 1j * tmp * bnj1_eta * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamnp * bsab[iab, s]
                    b32snj[snj] = -b23snj[snj]
                    b32 = -b23

                    # --- For eta_n^2 * A_nj ---
                    if bj0ab == 0 or kz == 0:
                        bnj1_eta2 = 0.0
                        bnj2_eta2 = czj[j] * czj[j]
                    else:
                        bnj1_eta2 = n**2 * wcs[s]**2 * bj0ab / cnj_val / vtzs[s]**2 / kz
                        bnj2_eta2 = ((vds[s] / vtzs[s] + czj[j]) * czj[j] / lmdTab[iab, s] +
                                     n * wcs[s] * bj0ab * (1 - n * wcs[s] / cnj_val) /
                                     vtzs[s]**2 / kz)

                    b33snj[snj] += rsab[iab, s] * tmp * bnj2_eta2 * 2 * lmdTab[iab, s] * Gamn
                    b33 += rsab[iab, s] * tmp * bnj1_eta2 * 2 * lmdTab[iab, s] * Gamn

                snj += 1  # increment after j-loop (after both iab iterations)

    # ---------------------------------------------------------------
    # Assemble the eigenvalue matrix M
    # ---------------------------------------------------------------
    for snj_idx in range(SNJ):
        # MATLAB: jjx = snj + 0*SNJ1, jjy = snj + 1*SNJ1, jjz = snj + 2*SNJ1
        # In MATLAB these are 1-based; in Python 0-based.
        jjx = snj_idx + 0 * SNJ1
        jjy = snj_idx + 1 * SNJ1
        jjz = snj_idx + 2 * SNJ1

        # v_snjx row
        M[jjx, jjx] += csnj[snj_idx]
        M[jjx, SNJ3 + 0] += b11snj[snj_idx]  # Ex column
        M[jjx, SNJ3 + 1] += b12snj[snj_idx]  # Ey column
        M[jjx, SNJ3 + 2] += b13snj[snj_idx]  # Ez column

        # v_snjy row
        M[jjy, jjy] += csnj[snj_idx]
        M[jjy, SNJ3 + 0] += b21snj[snj_idx]
        M[jjy, SNJ3 + 1] += b22snj[snj_idx]
        M[jjy, SNJ3 + 2] += b23snj[snj_idx]

        # v_snjz row
        M[jjz, jjz] += csnj[snj_idx]
        M[jjz, SNJ3 + 0] += b31snj[snj_idx]
        M[jjz, SNJ3 + 1] += b32snj[snj_idx]
        M[jjz, SNJ3 + 2] += b33snj[snj_idx]

    # E(J) rows: J_{x,y,z} = j_{x,y,z} + sum(v_snj{x,y,z})
    tp = -1.0
    for idx in range(SNJ1):
        # x-component
        M[SNJ3 + 0, 0 * SNJ1 + idx] += tp
        # y-component
        M[SNJ3 + 1, 1 * SNJ1 + idx] += tp
        # z-component
        M[SNJ3 + 2, 2 * SNJ1 + idx] += tp

    # j_x(E), j_y(E), j_z(E) -- the SNJ1-th row in each block
    # MATLAB: 1*SNJ1 (1-based) -> SNJ1-1 (0-based)
    M[1 * SNJ1 - 1, SNJ3 + 0] += b11
    M[1 * SNJ1 - 1, SNJ3 + 1] += b12
    M[1 * SNJ1 - 1, SNJ3 + 2] += b13
    M[2 * SNJ1 - 1, SNJ3 + 0] += b21
    M[2 * SNJ1 - 1, SNJ3 + 1] += b22
    M[2 * SNJ1 - 1, SNJ3 + 2] += b23
    M[3 * SNJ1 - 1, SNJ3 + 0] += b31
    M[3 * SNJ1 - 1, SNJ3 + 1] += b32
    M[3 * SNJ1 - 1, SNJ3 + 2] += b33

    # E(B): Maxwell's equations coupling E to B
    # MATLAB (1-based): SNJ3+1 -> SNJ3+0, SNJ3+5 -> SNJ3+4, etc.
    M[SNJ3 + 0, SNJ3 + 4] += c2 * kz       # Ex <- By
    M[SNJ3 + 1, SNJ3 + 3] += -c2 * kz      # Ey <- Bx
    M[SNJ3 + 1, SNJ3 + 5] += c2 * kx        # Ey <- Bz
    M[SNJ3 + 2, SNJ3 + 4] += -c2 * kx       # Ez <- By

    # B(E): Faraday's law coupling B to E
    M[SNJ3 + 3, SNJ3 + 1] += -kz            # Bx <- Ey
    M[SNJ3 + 4, SNJ3 + 0] += kz             # By <- Ex
    M[SNJ3 + 4, SNJ3 + 2] += -kx            # By <- Ez
    M[SNJ3 + 5, SNJ3 + 1] += kx             # Bz <- Ey

    return M.tocsr()

