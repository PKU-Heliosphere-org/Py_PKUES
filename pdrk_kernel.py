"""
pdrk_kernel.py
Converted from pdrk_kernel.m
Hua-sheng XIE, huashengxie@gmail.com, IFTS-ZJU, 2014-06-01 17:00
pdrk_em3d.m, Plasma Dispersion Relation solver, kinetic, EM3D,
bi-Maxwellian equilibrium distribution with parallel drift.
Transform to matrix eigenvalue problem lambda*X = M*X.
J-pole approximation for Z(zeta) = sum(b_j / (zeta - c_j))

Ref:
  [Xie2016] H. S. Xie & Y. Xiao, PDRK: A General Kinetic Dispersion
    Relation Solver for Magnetized Plasma, Plasma Science and Technology,
    Vol.18, No.2, p97 (2016).

18-10-03 13:14 Bugs free version.
18-10-13 10:37 Update to with loss-cone distribution.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs as sparse_eigs

from pdrk_em3d_matrix import pdrk_em3d_matrix
from pdrk_es3d_matrix import pdrk_es3d_matrix
from pkues_velocity import pkues_velocity


def pdrk_kernel(init, icalp=0, wws=None, wws2=None, Pola=None,
                Pola_norm=None, Pola_SI=None, jpl=0,
                Js=None, dV=None, dVnorm=None, xinorm=None, JE=None,
                Zp_norm=None, Zm_norm=None, scaling=None,
                idf=0, jpa_df=0, jpb_df=0):
    """
    Core kernel: scan over (pa, pb) and solve the dispersion eigenvalue problem.

    Parameters
    ----------
    init : dict
        Initialized parameters from pdrk_initialize.
    icalp : int
        =0 for first run (solve all roots), =1 for second run (polarization).
    wws : ndarray or None
        Previous solutions (needed if icalp=1).
    wws2, Pola, Pola_norm, Pola_SI : ndarrays or None
        Output arrays (needed if icalp=1).
    jpl : int
        Index of selected wave (for icalp=1).
    Js, dV, dVnorm, xinorm, JE : ndarrays or None
        Velocity/dissipation arrays (for icalp=1).
    Zp_norm, Zm_norm : ndarrays or None
        Elsasser variable arrays (for icalp=1).
    scaling : ndarray or None
        Scaling array (for icalp=1).
    idf : int
        Whether to calculate distribution function.
    jpa_df, jpb_df : int
        Indices for distribution function calculation.

    Returns
    -------
    result : dict
        Dictionary with keys: 'ww', 'kk', 'kxx', 'kzz', 'tt'
        (and polarization data if icalp=1).
    """
    # Unpack init
    S = init['S']
    N = init['N']
    J = init['J']
    NN = init['NN']
    SNJ = init['SNJ']
    SNJ1 = init['SNJ1']
    iem = init['iem']
    sp = init['sp']
    nw = init['nw']
    npa = init['npa']
    npb = init['npb']
    pa = init['pa']
    pb = init['pb']
    ipa = init['ipa']
    ipb = init['ipb']
    iloga = init['iloga']
    ilogb = init['ilogb']
    iout = init['iout']
    par = init['par'].copy()
    cwp = init['cwp']
    wcs1 = init['wcs1']
    c2 = init['c2']
    epsilon0 = init['epsilon0']
    mu0 = init['mu0']
    B0 = init['B0']
    vA = init['vA']

    rhocsab = init['rhocsab'].copy()
    vtzs = init['vtzs']
    vds = init['vds']
    wcs = init['wcs']
    wps2 = init['wps2']
    czj = init['czj']
    bzj = init['bzj']
    lmdTab = init['lmdTab']
    rsab = init['rsab']
    kDs = 1.0 / init['lambdaDs']
    ns0 = init['ns0']
    qs = init['qs']

    if iem == 1:
        SNJ3 = init['SNJ3']

    wg0_local = init['wg0'] * wcs1 if sp != 0 else 0.0
    wg = wg0_local

    # Allocate output arrays
    wwp = np.zeros((npa, npb, nw), dtype=complex)
    ppa, ppb = np.meshgrid(pa, pb, indexing='ij')  # ndgrid equivalent
    kxx = np.zeros((npa, npb))
    kzz = np.zeros((npa, npb))
    kk = np.zeros((npa, npb))
    tt = np.zeros((npa, npb))

    for jpa in range(npa):
        for jpb in range(npb):

            # Update scan parameter(s)
            if iloga == 0:
                par[ipa - 1] = pa[jpa]  # ipa is 1-based index
            else:
                par[ipa - 1] = 10.0**pa[jpa]

            if ipa != ipb:
                if ilogb == 0:
                    par[ipb - 1] = pb[jpb]
                else:
                    par[ipb - 1] = 10.0**pb[jpb]

            # Compute k, theta, kz, kx
            k = np.abs(par[0]) / cwp  # k >= 0, [m^-1]
            theta = par[1]            # angle, [degree]

            if ipa > 2 and ipb > 2:
                kz = par[2] / cwp     # [m^-1]
                kx = par[3] / cwp     # [m^-1]
            else:
                kz = np.cos(theta * np.pi / 180) * k
                kx = np.sin(theta * np.pi / 180) * k

            k = np.sqrt(kx**2 + kz**2)                      # update k
            theta = np.angle(kz + 1j * kx) * 180 / np.pi    # update theta

            # Build the dispersion matrix
            if iem == 1:
                M = pdrk_em3d_matrix(kz, kx, S, N, J, NN, SNJ, SNJ1, SNJ3,
                                      rhocsab.copy(), vtzs, vds, wcs, wps2,
                                      czj, bzj, lmdTab, rsab, c2)
            else:
                M = pdrk_es3d_matrix(kz, kx, S, N, J, NN, SNJ,
                                      rhocsab.copy(), vtzs, vds, wcs, kDs,
                                      czj, bzj, lmdTab, rsab)

            # Solve eigenvalue problem
            if sp == 0:
                # Solve all roots
                d = np.linalg.eigvals(M.toarray())
            else:
                # Sparse solver: only nw solutions around wg
                if iem == 1 and iout == 2 and icalp == 1:
                    # Polarization run
                    print(f'jpa={jpa}, jpb={jpb}')
                    eps = 1j * 1e-16
                    wg_local = wws[jpa, jpb, jpl] * wcs1 + eps

                    _nan_c = np.nan + 0j * np.nan
                    eigvec_reliable = True

                    try:
                        eigenvalues, eigenvectors = sparse_eigs(
                            M.astype(complex), k=nw, sigma=wg_local)
                    except Exception as e:
                        print(f"[WARN] sparse_eigs failed at jpa={jpa}, jpb={jpb}, jpl={jpl}: {e}")
                        eigvec_reliable = False
                        eigenvalues = np.array([wg_local])
                        eigenvectors = np.zeros((NN, 1), dtype=complex)

                    d = eigenvalues

                    # Extract polarization (dE, dB) from eigenvector
                    V = eigenvectors
                    D_val = eigenvalues[0]
                    v0 = V[:, 0]
                    dEB = v0[(NN - 6):NN]  # last 6 rows of first eigenvector

                    # ============================================================
                    # 特征向量可靠性检查：
                    # 计算残差 r = ||M*v - lambda*v|| / |lambda|
                    # 当 k→0 矩阵近奇异时，shift-invert 求出的特征向量可能
                    # 完全不可信（残差 >> 1）。此时该 k 点的极化量无物理意义，
                    # 全部设为 NaN，绘图时自动跳过。
                    # ============================================================
                    if eigvec_reliable:
                        M_dense = M.toarray() if sparse.issparse(M) else M
                        residual_vec = M_dense @ v0 - D_val * v0
                        res_norm = np.linalg.norm(residual_vec)
                        denom_norm = np.abs(D_val)
                        if denom_norm > 0:
                            rel_residual = res_norm / denom_norm
                        else:
                            rel_residual = res_norm
                        # 阈值：相对残差超过 1e-3 则认为不可信
                        if rel_residual > 1e-10:
                            print(f"  [SKIP] jpa={jpa}: eigenvector unreliable "
                                  f"(rel_residual={rel_residual:.2e}), set to NaN")
                            eigvec_reliable = False

                    if not eigvec_reliable:
                        # 该 k 点全部极化量设为 NaN
                        wws2[jpa, jpb, jpl] = _nan_c
                        Pola_SI[jpa, jpb, jpl, :] = _nan_c
                        Pola_norm[jpa, jpb, jpl, :] = _nan_c
                        Pola[jpa, jpb, jpl, :] = _nan_c
                        scaling[jpa, jpb, jpl] = _nan_c
                        for s_idx in range(S):
                            for comp in range(3):
                                Js[jpa, comp, s_idx, jpl] = _nan_c
                                dV[jpa, comp, s_idx, jpl] = _nan_c
                                dVnorm[jpa, comp, s_idx, jpl] = _nan_c
                                JE[jpa, comp, s_idx, jpl] = _nan_c
                                Zp_norm[jpa, comp, s_idx, jpl] = _nan_c
                                Zm_norm[jpa, comp, s_idx, jpl] = _nan_c
                            xinorm[jpa, s_idx, jpl] = _nan_c
                    else:
                        # 特征向量可信，正常计算极化量
                        wws2[jpa, jpb, jpl] = D_val / wcs1

                        # Scaling: 优先按 dBy 归一化，用相对阈值判断
                        scaling_factor = 1.0
                        max_dB = np.max(np.abs(dEB[3:6]))
                        max_dEB = np.max(np.abs(dEB))
                        den = dEB[4]
                        if max_dB > 0 and np.abs(den) < max_dB * 1e-6:
                            idx_ref = 3 + np.argmax(np.abs(dEB[3:6]))
                            den = dEB[idx_ref]
                        if np.abs(den) == 0:
                            if max_dEB > 0:
                                den = dEB[np.argmax(np.abs(dEB))]
                            else:
                                den = 1.0 + 0.0j

                        scaling[jpa, jpb, jpl] = scaling_factor * B0 / den

                        Pola_SI[jpa, jpb, jpl, 0:6] = dEB * scaling[jpa, jpb, jpl]
                        Pola_norm[jpa, jpb, jpl, 0:6] = Pola_SI[jpa, jpb, jpl, 0:6] / B0
                        Pola_norm[jpa, jpb, jpl, 0:3] = Pola_norm[jpa, jpb, jpl, 0:3] / vA

                        # Energy densities
                        dE = Pola_SI[jpa, jpb, jpl, 0:3]
                        dB = Pola_SI[jpa, jpb, jpl, 3:6]
                        Pola_SI[jpa, jpb, jpl, 6] = (
                            np.sum(dE * np.conj(dE)).real * epsilon0 * 0.5)
                        Pola_SI[jpa, jpb, jpl, 7] = (
                            np.sum(dB * np.conj(dB)).real / mu0 * 0.5)

                        # Current density from Ampere's law
                        Pola_SI[jpa, jpb, jpl, 8] = (
                            c2 * kz * Pola_SI[jpa, jpb, jpl, 4] -
                            D_val * Pola_SI[jpa, jpb, jpl, 0]) * epsilon0 / 1j
                        Pola_SI[jpa, jpb, jpl, 9] = (
                            c2 * kx * Pola_SI[jpa, jpb, jpl, 5] -
                            c2 * kz * Pola_SI[jpa, jpb, jpl, 3] -
                            D_val * Pola_SI[jpa, jpb, jpl, 1]) * epsilon0 / 1j
                        Pola_SI[jpa, jpb, jpl, 10] = (
                            -c2 * kx * Pola_SI[jpa, jpb, jpl, 4] -
                            D_val * Pola_SI[jpa, jpb, jpl, 2]) * epsilon0 / 1j

                        # Normalized current
                        Pola_norm[jpa, jpb, jpl, 8] = (
                            Pola_SI[jpa, jpb, jpl, 8] / 1j / (qs[0] * ns0[0] * vA))
                        Pola_norm[jpa, jpb, jpl, 9] = (
                            Pola_SI[jpa, jpb, jpl, 9] / 1j / (qs[0] * ns0[0] * vA))
                        Pola_norm[jpa, jpb, jpl, 10] = (
                            Pola_SI[jpa, jpb, jpl, 10] / 1j / (qs[0] * ns0[0] * vA))

                        # Calculate velocity
                        Js, dV, dVnorm, xinorm, JE = pkues_velocity(
                            jpa, jpl, wws2, Pola_SI, kz, kx, rhocsab,
                            vtzs, vds, wcs, wps2, czj, bzj, lmdTab, rsab,
                            ns0, qs, vA, epsilon0, mu0, wcs1,
                            S, N, J, SNJ, NN,
                            Js, dV, dVnorm, xinorm, JE)

                        # Elsasser variables
                        for s_idx in range(S):
                            density_ratio = np.sqrt(ns0[0] / ns0[s_idx])
                            for comp in range(3):
                                b_comp = Pola_norm[jpa, jpb, jpl, 3 + comp] * density_ratio
                                Zp_norm[jpa, comp, s_idx, jpl] = (
                                    dVnorm[jpa, comp, s_idx, jpl] + b_comp)
                                Zm_norm[jpa, comp, s_idx, jpl] = (
                                    dVnorm[jpa, comp, s_idx, jpl] - b_comp)

                        # Normalize dEB for storage
                        dEB = dEB / dEB[0]
                        ctmp = (np.sqrt(np.real(dEB[0])**2 + np.real(dEB[1])**2 +
                                        np.real(dEB[2])**2) +
                                np.imag(dEB[0])**2 + np.imag(dEB[1])**2 +
                                np.imag(dEB[2])**2)
                        dEB = dEB / ctmp
                        Pola[jpa, jpb, jpl, 0:6] = dEB
                        wws2[jpa, jpb, jpl] = D_val / wcs1

                        # Energy
                        dEx, dEy, dEz = dEB[0], dEB[1], dEB[2]
                        dBx, dBy, dBz = dEB[3], dEB[4], dEB[5]
                        UE = (dEx * np.conj(dEx) + dEy * np.conj(dEy) +
                              dEz * np.conj(dEz)) * epsilon0 * 0.5
                        UB = (dBx * np.conj(dBx) + dBy * np.conj(dBy) +
                              dBz * np.conj(dBz)) / mu0 * 0.5
                        Pola[jpa, jpb, jpl, 6] = UE
                        Pola[jpa, jpb, jpl, 7] = UB

                else:
                    # Standard sparse solve
                    try:
                        d = sparse_eigs(M.astype(complex), k=nw, sigma=wg,
                                        return_eigenvectors=False)
                    except Exception as e:
                        print(f"[WARN] sparse_eigs failed at jpa={jpa}, jpb={jpb}, jpl={jpl}: {e}")
                        d = np.array([wg])

            # Sort by growth rate (descending imaginary part)
            omega = d
            ind = np.argsort(-np.imag(omega))
            w = omega[ind]

            # Update initial guess
            wg = w[0]
            if jpb == 0:
                wg0_local = wg

            nw_store = min(nw, len(w))
            wwp[jpa, jpb, :nw_store] = w[:nw_store]

            # Store k, theta in first run
            if icalp == 0:
                kxx[jpa, jpb] = kx
                kzz[jpa, jpb] = kz
                kk[jpa, jpb] = k
                tt[jpa, jpb] = theta

        # Reset initial guess for next jpa
        wg = wg0_local

    # Build result
    result = {
        'wwp': wwp,
        'kk': kk,
        'kxx': kxx,
        'kzz': kzz,
        'tt': tt,
        'ppa': ppa,
        'ppb': ppb,
    }

    if icalp == 0:
        result['ww'] = wwp.copy()
    else:
        result['wws2'] = wws2
        result['Pola'] = Pola
        result['Pola_norm'] = Pola_norm
        result['Pola_SI'] = Pola_SI
        result['Js'] = Js
        result['dV'] = dV
        result['dVnorm'] = dVnorm
        result['xinorm'] = xinorm
        result['JE'] = JE
        result['Zp_norm'] = Zp_norm
        result['Zm_norm'] = Zm_norm
        result['scaling'] = scaling

    return result

