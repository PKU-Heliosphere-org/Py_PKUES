"""
pkues_velocity.py
Converted from pkues_velocity.m
20-05-14 08:23 Coded By Xingyu Zhu and Jiansen He
This file calculates the current/velocity of each component
"""

import numpy as np
from scipy.special import ive as besseli_scaled  # besseli(n, x, 1) = ive(n, x)


def pkues_velocity(jpa, jpl, wws2, Pola_SI, kz, kx, rhocsab,
                   vtzs, vds, wcs, wps2, czj, bzj, lmdTab, rsab,
                   ns0, qs, vA, epsilon0, mu0, wcs1,
                   S, N, J, SNJ, NN,
                   Js, dV, dVnorm, xinorm, JE):
    """
    Calculate the current/velocity of each component for a given (jpa, jpl).

    Parameters
    ----------
    jpa : int
        Index for parameter a scan.
    jpl : int
        Index for selected wave/polarization.
    wws2, Pola_SI : arrays
        Wave solutions and polarization in SI units.
    kz, kx : float
        Parallel and perpendicular wavenumbers.
    rhocsab : array (2, S)
        Cyclotron radii for core/beam of each species.
    vtzs : array (S,)
        Parallel thermal velocities.
    vds : array (S,)
        Parallel drift velocities.
    wcs : array (S,)
        Cyclotron frequencies.
    wps2 : array (S,)
        Squared plasma frequencies.
    czj, bzj : arrays (J,)
        J-pole expansion coefficients.
    lmdTab : array (2, S)
        Temperature anisotropy ratios for core/beam.
    rsab : array (2, S)
        Core/beam density ratios.
    ns0 : array (S,)
        Number densities.
    qs : array (S,)
        Charges.
    vA : float
        Alfven speed.
    epsilon0, mu0 : float
        Vacuum permittivity and permeability.
    wcs1 : float
        1st species cyclotron frequency.
    S, N, J, SNJ, NN : int
        Number of species, harmonics, poles, etc.
    Js, dV, dVnorm, xinorm, JE : arrays (modified in place)
        Output arrays for current, velocity, normalized velocity,
        density fluctuation, and dissipation.
    """
    k = np.sqrt(kz**2 + kx**2)
    bsab = kx * rhocsab  # shape (2, S)
    bsab[np.abs(bsab) < 1e-50] = 1e-50  # avoid singular when k_perp=0
    bsab2 = bsab**2

    # Initialize local per-species sums
    b11_s = np.zeros(S, dtype=complex)
    b12_s = np.zeros(S, dtype=complex)
    b13_s = np.zeros(S, dtype=complex)
    b21_s = np.zeros(S, dtype=complex)
    b22_s = np.zeros(S, dtype=complex)
    b23_s = np.zeros(S, dtype=complex)
    b31_s = np.zeros(S, dtype=complex)
    b32_s = np.zeros(S, dtype=complex)
    b33_s = np.zeros(S, dtype=complex)

    total_nj = (2 * N + 1) * J  # number of (n,j) combinations
    cnj_s = np.zeros((total_nj, S), dtype=complex)
    b11nj_s = np.zeros((total_nj, S), dtype=complex)
    b12nj_s = np.zeros((total_nj, S), dtype=complex)
    b13nj_s = np.zeros((total_nj, S), dtype=complex)
    b21nj_s = np.zeros((total_nj, S), dtype=complex)
    b22nj_s = np.zeros((total_nj, S), dtype=complex)
    b23nj_s = np.zeros((total_nj, S), dtype=complex)
    b31nj_s = np.zeros((total_nj, S), dtype=complex)
    b32nj_s = np.zeros((total_nj, S), dtype=complex)
    b33nj_s = np.zeros((total_nj, S), dtype=complex)

    for s in range(S):
        nj = 0
        for n in range(-N, N + 1):
            for j in range(J):
                for iab in range(2):  # core (0) and beam (1)
                    # Scaled Bessel functions: besseli(n, x, 1) in MATLAB = ive(n, x) in scipy
                    Gamn = besseli_scaled(n, bsab2[iab, s])
                    Gamnp = (besseli_scaled(n + 1, bsab2[iab, s]) +
                             besseli_scaled(n - 1, bsab2[iab, s]) -
                             2 * besseli_scaled(n, bsab2[iab, s])) / 2

                    cnj_val = czj[j] * kz * vtzs[s] + kz * vds[s] + n * wcs[s]
                    cnj_s[nj, s] = cnj_val
                    bj0ab = vds[s] + (1 - 1 / lmdTab[iab, s]) * czj[j] * vtzs[s]

                    # For A_nj
                    if n == 0:
                        bnj1 = bj0ab / (czj[j] * vtzs[s] + vds[s])  # avoid cnj=0
                    else:
                        bnj1 = kz * bj0ab / cnj_val
                    bnj2 = 1 - bnj1

                    tmp = wps2[s] * bzj[j]

                    b11nj_s[nj, s] += rsab[iab, s] * tmp * bnj2 * n**2 * Gamn / bsab2[iab, s]
                    b11_s[s] += rsab[iab, s] * tmp * bnj1 * n**2 * Gamn / bsab2[iab, s]

                    b12nj_s[nj, s] += rsab[iab, s] * tmp * bnj2 * 1j * n * Gamnp
                    b12_s[s] += rsab[iab, s] * tmp * bnj1 * 1j * n * Gamnp
                    b21nj_s[nj, s] = -b12nj_s[nj, s]
                    b21_s[s] = -b12_s[s]

                    b22nj_s[nj, s] += rsab[iab, s] * tmp * bnj2 * (
                        n**2 * Gamn / bsab2[iab, s] - 2 * bsab2[iab, s] * Gamnp)
                    b22_s[s] += rsab[iab, s] * tmp * bnj1 * (
                        n**2 * Gamn / bsab2[iab, s] - 2 * bsab2[iab, s] * Gamnp)

                    # For eta_n * A_nj
                    if n == 0:
                        bnj1_eta = 0  # avoid cnj=0 when kz=0
                    else:
                        bnj1_eta = n * wcs[s] * bj0ab / cnj_val / vtzs[s]
                    bnj2_eta = czj[j] / lmdTab[iab, s] + bnj1_eta

                    b13nj_s[nj, s] += rsab[iab, s] * tmp * bnj2_eta * n * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamn / bsab[iab, s]
                    b13_s[s] += -rsab[iab, s] * tmp * bnj1_eta * n * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamn / bsab[iab, s]
                    b31nj_s[nj, s] = b13nj_s[nj, s]
                    b31_s[s] = b13_s[s]

                    b23nj_s[nj, s] += -rsab[iab, s] * 1j * tmp * bnj2_eta * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamnp * bsab[iab, s]
                    b23_s[s] += rsab[iab, s] * 1j * tmp * bnj1_eta * \
                        np.sqrt(2 * lmdTab[iab, s]) * Gamnp * bsab[iab, s]
                    b32nj_s[nj, s] = -b23nj_s[nj, s]
                    b32_s[s] = -b23_s[s]

                    # For eta_n^2 * A_nj
                    if bj0ab == 0 or kz == 0:
                        bnj1_eta2 = 0
                        bnj2_eta2 = czj[j] * czj[j]
                    else:
                        bnj1_eta2 = n**2 * wcs[s]**2 * bj0ab / cnj_val / vtzs[s]**2 / kz
                        bnj2_eta2 = (vds[s] / vtzs[s] + czj[j]) * czj[j] / lmdTab[iab, s] + \
                            n * wcs[s] * bj0ab * (1 - n * wcs[s] / cnj_val) / vtzs[s]**2 / kz

                    b33nj_s[nj, s] += rsab[iab, s] * tmp * bnj2_eta2 * 2 * lmdTab[iab, s] * Gamn
                    b33_s[s] += rsab[iab, s] * tmp * bnj1_eta2 * 2 * lmdTab[iab, s] * Gamn

                nj += 1  # increment after j loop (for both iab)

    # Calculate current and velocity
    wtmp = wws2[jpa, 0, jpl] * wcs1
    dEx_tmp = Pola_SI[jpa, 0, jpl, 0]
    dEy_tmp = Pola_SI[jpa, 0, jpl, 1]
    dEz_tmp = Pola_SI[jpa, 0, jpl, 2]
    dBx_tmp = Pola_SI[jpa, 0, jpl, 3]
    dBy_tmp = Pola_SI[jpa, 0, jpl, 4]
    dBz_tmp = Pola_SI[jpa, 0, jpl, 5]

    for s in range(S):
        Jxs = 0.0 + 0j
        Jys = 0.0 + 0j
        Jzs = 0.0 + 0j
        nj = 0
        for n in range(-N, N + 1):
            for j in range(J):
                denom = wtmp - cnj_s[nj, s]
                Jxs += (b11nj_s[nj, s] / denom * dEx_tmp +
                        b12nj_s[nj, s] / denom * dEy_tmp +
                        b13nj_s[nj, s] / denom * dEz_tmp)
                Jys += (b21nj_s[nj, s] / denom * dEx_tmp +
                        b22nj_s[nj, s] / denom * dEy_tmp +
                        b23nj_s[nj, s] / denom * dEz_tmp)
                Jzs += (b31nj_s[nj, s] / denom * dEx_tmp +
                        b32nj_s[nj, s] / denom * dEy_tmp +
                        b33nj_s[nj, s] / denom * dEz_tmp)
                nj += 1

        Js[jpa, 0, s, jpl] = -1j * epsilon0 * (
            Jxs + b11_s[s] / wtmp * dEx_tmp +
            b12_s[s] / wtmp * dEy_tmp + b13_s[s] / wtmp * dEz_tmp)
        Js[jpa, 1, s, jpl] = -1j * epsilon0 * (
            Jys + b21_s[s] / wtmp * dEx_tmp +
            b22_s[s] / wtmp * dEy_tmp + b23_s[s] / wtmp * dEz_tmp)
        Js[jpa, 2, s, jpl] = -1j * epsilon0 * (
            Jzs + b31_s[s] / wtmp * dEx_tmp +
            b32_s[s] / wtmp * dEy_tmp + b33_s[s] / wtmp * dEz_tmp)

        dV[jpa, 0, s, jpl] = Js[jpa, 0, s, jpl] / (qs[s] * ns0[s])
        dV[jpa, 1, s, jpl] = Js[jpa, 1, s, jpl] / (qs[s] * ns0[s])
        dV[jpa, 2, s, jpl] = Js[jpa, 2, s, jpl] / (qs[s] * ns0[s])

        dVnorm[jpa, 0, s, jpl] = dV[jpa, 0, s, jpl] / vA
        dVnorm[jpa, 1, s, jpl] = dV[jpa, 1, s, jpl] / vA
        dVnorm[jpa, 2, s, jpl] = dV[jpa, 2, s, jpl] / vA

        # Density fluctuation
        xinorm[jpa, s, jpl] = (dV[jpa, 0, s, jpl] * kx +
                                dV[jpa, 2, s, jpl] * kz) / wtmp

        # Dissipation: JE = -Re(J · E*) / (EB_energy) / 2
        JE[jpa, 0, s, jpl] = -(Js[jpa, 0, s, jpl] * np.conj(dEx_tmp) +
                                np.conj(Js[jpa, 0, s, jpl]) * dEx_tmp) / 4
        JE[jpa, 1, s, jpl] = -(Js[jpa, 1, s, jpl] * np.conj(dEy_tmp) +
                                np.conj(Js[jpa, 1, s, jpl]) * dEy_tmp) / 4
        JE[jpa, 2, s, jpl] = -(Js[jpa, 2, s, jpl] * np.conj(dEz_tmp) +
                                np.conj(Js[jpa, 2, s, jpl]) * dEz_tmp) / 4

        EBenergy = ((dEx_tmp * np.conj(dEx_tmp) + dEy_tmp * np.conj(dEy_tmp) +
                     dEz_tmp * np.conj(dEz_tmp)) * epsilon0 * 0.5 +
                    (dBx_tmp * np.conj(dBx_tmp) + dBy_tmp * np.conj(dBy_tmp) +
                     dBz_tmp * np.conj(dBz_tmp)) / mu0 * 0.5)

        JE[jpa, 0, s, jpl] = JE[jpa, 0, s, jpl] / EBenergy / 2
        JE[jpa, 1, s, jpl] = JE[jpa, 1, s, jpl] / EBenergy / 2
        JE[jpa, 2, s, jpl] = JE[jpa, 2, s, jpl] / EBenergy / 2

    return Js, dV, dVnorm, xinorm, JE

