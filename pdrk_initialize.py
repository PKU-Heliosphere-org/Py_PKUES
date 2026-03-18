"""
pdrk_initialize.py
Converted from pdrk_initialize.m
18-10-06 07:36 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Ackn.: Richard Denton (Dartmouth), Xin Tao (USTC), Jin-song Zhao (PMO), etc.
This file initializes the parameters for pdrk_kernel.py
"""

import numpy as np
import os


def pdrk_initialize(config):
    """
    Initialize all plasma parameters from the input file and config.

    Parameters
    ----------
    config : dict
        Configuration from pkues_setup.

    Returns
    -------
    init : dict
        Dictionary containing all initialized plasma and scan parameters.
    """
    # Physical constants
    c2 = (2.9979e8)**2            # speed of light c^2, [m^2/s^2]
    epsilon0 = 8.8542e-12         # vacuum permittivity, [F/m]
    mu0 = 1.0 / (c2 * epsilon0)  # vacuum permeability, [H/m]
    kB = 1.3807e-23               # Boltzmann constant, [J/K]

    # Extract config
    iem = config['iem']
    N = config['N']
    J = config['J']
    sp = config['sp']
    wg0 = config['wg0']
    B0 = config['B0']
    par = config['par'].copy()
    ipa = config['ipa']
    ipb = config['ipb']
    iloga = config['iloga']
    ilogb = config['ilogb']
    pa1 = config['pa1']
    pa2 = config['pa2']
    dpa = config['dpa']
    pb1 = config['pb1']
    pb2 = config['pb2']
    dpb = config['dpb']
    savepath = config['savepath']

    # ---------------------------------------------------------------
    # Read input parameters from file
    # ---------------------------------------------------------------
    input_file = '../input/pdrk.in'
    if not os.path.isfile(input_file):
        # Try alternative paths
        for alt in ['input/pdrk.in', 'pdrk.in', '../input/pdrk_em3d.in']:
            if os.path.isfile(alt):
                input_file = alt
                break

    pardat = _read_input(input_file)
    S = pardat.shape[0]  # number of species
    col = pardat.shape[1]
    if col != 8:
        print('Wrong input data !!!')

    # Species parameters (0-indexed arrays)
    qs0 = pardat[:, 0]       # charge, q/e
    ms0 = pardat[:, 1]       # mass, m/mp
    ns0 = pardat[:, 2]       # density, m^-3
    Tzs0 = pardat[:, 3]      # parallel temperature, eV
    Tps0 = pardat[:, 4]      # perp temperature, eV
    alphas = pardat[:, 5]    # loss-cone size/anisotropic
    Deltas = pardat[:, 6]    # loss-cone depth, =0 (max) to 1 (no)
    vds0 = pardat[:, 7]      # parallel drift velocity, vds/c

    # Core/beam (loss-cone) distribution ratios
    rsab = np.zeros((2, S))
    for s in range(S):
        if alphas[s] == 1:
            rsab[0, s] = 1.0
            rsab[1, s] = 0.0
        else:
            # sigma=a, core fv ratio
            rsab[0, s] = (1 - alphas[s] * Deltas[s]) / (1 - alphas[s])
            # sigma=b, loss cone fv ratio
            rsab[1, s] = alphas[s] * (Deltas[s] - 1) / (1 - alphas[s])

    # Check charge and current neutrality
    Qtotal = np.sum(qs0 * ns0)
    Jtotal = np.sum(qs0 * ns0 * vds0)
    if Qtotal != 0 or Jtotal != 0:
        print('Warning: Total charge or current not zero !!!')

    # Convert to SI units
    e_charge = 1.6022e-19
    m_proton = 1.6726e-27

    qs = qs0 * e_charge                    # [C] (coulomb)
    ms = ms0 * m_proton                    # [kg]
    Tzs = Tzs0 * e_charge / kB            # T_parallel, eV -> [K]
    Tps = Tps0 * e_charge / kB            # T_perp, eV -> [K]
    vds = vds0 * np.sqrt(c2)              # [m/s]

    # Derived quantities
    vtzs = np.sqrt(2 * kB * Tzs / ms)                     # parallel thermal velocity [m/s]
    lambdaDs = np.sqrt(epsilon0 * kB * Tzs / (ns0 * qs**2))  # Debye length [m]
    kDs = 1.0 / lambdaDs                                   # [m^-1]
    wps = np.sqrt(ns0 * qs**2 / (ms * epsilon0))          # plasma frequency [rad/s]
    wcs = B0 * qs / ms                                     # cyclotron frequency [rad/s]
    rhocs = np.sqrt(kB * Tps / ms) / wcs                   # cyclotron radius [m]

    wps2 = wps**2                          # squared plasma frequency [(rad/s)^2]
    lmdT = Tzs / Tps                       # T_parallel / T_perp, dimensionless

    # Loss-cone two perpendicular temperatures
    Tpsab = np.zeros((2, S))
    Tpsab[0, :] = Tps
    Tpsab[1, :] = alphas * Tps             # [K]

    lmdTab = np.zeros((2, S))
    lmdTab[0, :] = Tzs / Tpsab[0, :]
    lmdTab[1, :] = Tzs / Tpsab[1, :]      # dimensionless

    rhocsab = np.zeros((2, S))
    rhocsab[0, :] = np.sqrt(kB * Tpsab[0, :] / ms) / wcs   # [m]
    rhocsab[1, :] = np.sqrt(kB * Tpsab[1, :] / ms) / wcs   # [m]

    # Beta and Alfven speed
    betasz = 2 * mu0 * kB * ns0 * Tzs / B0**2              # beta_parallel
    betasp = 2 * mu0 * kB * ns0 * Tps / B0**2              # beta_perp
    vA = B0 / np.sqrt(mu0 * np.sum(ms * ns0))              # total Alfven speed [m/s]
    vAs = B0 / np.sqrt(mu0 * ms * ns0)                     # species Alfven speed [m/s]

    # Normalized by omega_c and omega_p of the first species
    cSs1 = np.sqrt(2 * kB * Tzs[0] / ms[0])               # sound speed [m/s]
    wcs1 = np.abs(wcs[0])                                   # |omega_{c1}| [rad/s]
    wps1 = np.sqrt(ns0[0] * qs[0]**2 / (ms[0] * epsilon0))  # omega_{p1} [rad/s]

    # Thermal speeds (added by Xingyu Zhu 2020-06-23)
    vthzS = np.sqrt(2 * kB * Tzs / ms)                     # [m/s]
    vthpS = np.sqrt(2 * kB * Tps / ms)                     # [m/s]

    cwp = np.sqrt(c2) / wps1             # c/omega_{p1} [m]
    vAwp = vA / wcs1                     # v_A/omega_{c1} [m]

    # ---------------------------------------------------------------
    # J-pole expansion coefficients
    # ---------------------------------------------------------------
    bzj, czj = _get_jpole_coefficients(J)

    # ---------------------------------------------------------------
    # Matrix dimensions
    # ---------------------------------------------------------------
    SNJ = S * (2 * N + 1) * J
    SNJ1 = SNJ + 1

    if iem == 1:  # electromagnetic
        SNJ3 = 3 * SNJ1
        NN = SNJ3 + 6
    else:         # electrostatic
        NN = SNJ1

    # Number of roots
    if sp == 0:
        nw = NN
    else:
        nw = 1
        wg = wg0 * wcs1   # [rad/s]

    nw0 = nw // 3

    # ---------------------------------------------------------------
    # Scan parameter arrays
    # ---------------------------------------------------------------
    npa = round((pa2 - pa1) / dpa) + 1
    pa = pa1 + np.arange(npa) * dpa

    if ipa == ipb:  # 1D scan
        npb = 1
        pb = np.array([pa[0]])
    else:           # 2D scan
        npb = round((pb2 - pb1) / dpb) + 1
        pb = pb1 + np.arange(npb) * dpb

    # Scan type strings
    ireturn = 0
    ipbtmp = 0
    if ipa == 1 and ipb == 1:
        strpa = r'$kc/\omega_p$'
        strpb = r'$\theta°$'
        strscan = '1. (1,1) scan k, fixed theta'
        ipbtmp = 1  # index 1 (0-based) = theta
    elif ipa == 2 and ipb == 2:
        strpa = r'$\theta$'
        strpb = r'$kc/\omega_p$'
        strscan = '2. (2,2) scan theta, fixed k'
        ipbtmp = 0  # index 0 (0-based) = k
    elif ipa == 3 and ipb == 3:
        strpa = r'$k_zc/\omega_p$'
        strpb = r'$k_xc/\omega_p$'
        strscan = '4. (3,3) scan kz, fixed kx'
        ipbtmp = 3  # index 3 (0-based) = kx
    elif ipa == 4 and ipb == 4:
        strpa = r'$k_xc/\omega_p$'
        strpb = r'$k_zc/\omega_p$'
        strscan = '5. (4,4) scan kx, fixed kz'
        ipbtmp = 2  # index 2 (0-based) = kz
    elif ipa == 1 and ipb == 2:
        strpa = r'$kc/\omega_p$'
        strpb = r'$\theta$'
        strscan = '3. (1,2) scan 2D (k, theta)'
    elif ipa == 3 and ipb == 4:
        strpa = r'$k_zc/\omega_p$'
        strpb = r'$k_xc/\omega_p$'
        strscan = '6. (3,4) scan 2D (kz,kx)'
    else:
        strpa = '--'
        strpb = '--'
        strscan = ("7. (..,..), not support this scan (ipa,ipb) yet, "
                   "please modify 'pdrk_kernel.py'!!!")
        ireturn = 1

    # Create output directory
    os.makedirs(savepath, exist_ok=True)

    # ---------------------------------------------------------------
    # Pack into dict
    # ---------------------------------------------------------------
    init = {
        # Physical constants
        'c2': c2, 'epsilon0': epsilon0, 'mu0': mu0, 'kB': kB,
        # Species parameters (original units)
        'S': S, 'qs0': qs0, 'ms0': ms0, 'ns0': ns0,
        'Tzs0': Tzs0, 'Tps0': Tps0, 'alphas': alphas, 'Deltas': Deltas, 'vds0': vds0,
        # SI units
        'qs': qs, 'ms': ms, 'Tzs': Tzs, 'Tps': Tps, 'vds': vds,
        # Derived
        'vtzs': vtzs, 'lambdaDs': lambdaDs, 'kDs': kDs,
        'wps': wps, 'wcs': wcs, 'rhocs': rhocs,
        'wps2': wps2, 'lmdT': lmdT,
        # Loss-cone
        'rsab': rsab, 'Tpsab': Tpsab, 'lmdTab': lmdTab, 'rhocsab': rhocsab,
        # Beta and speeds
        'betasz': betasz, 'betasp': betasp, 'vA': vA, 'vAs': vAs,
        'cSs1': cSs1, 'wcs1': wcs1, 'wps1': wps1,
        'vthzS': vthzS, 'vthpS': vthpS,
        'cwp': cwp, 'vAwp': vAwp,
        # J-pole
        'bzj': bzj, 'czj': czj, 'J': J,
        # Matrix dimensions
        'N': N, 'SNJ': SNJ, 'SNJ1': SNJ1, 'NN': NN,
        'iem': iem, 'sp': sp, 'sp0': sp,
        'nw': nw, 'nw0': nw0,
        # Scan parameters
        'npa': npa, 'npb': npb, 'pa': pa, 'pb': pb,
        'ipa': ipa, 'ipb': ipb,
        'iloga': iloga, 'ilogb': ilogb,
        'strpa': strpa, 'strpb': strpb,
        'strscan': strscan, 'ireturn': ireturn, 'ipbtmp': ipbtmp,
        # Config pass-through
        'par': par, 'B0': B0, 'iout': config['iout'],
        'wg0': wg0, 'savepath': savepath,
    }

    if sp != 0:
        init['wg'] = wg

    if iem == 1:
        init['SNJ3'] = 3 * SNJ1

    return init


def _read_input(filename):
    """
    Read the PDRK input file (skip 1 header line, space-delimited).

    Returns
    -------
    data : ndarray, shape (S, 8)
        Each row: [qs0, ms0, ns0, Tzs0, Tps0, alphas, Deltas, vds0]
    """
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Skip header line(s)
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('%') or line.startswith('#'):
            continue
        vals = line.split()
        if len(vals) >= 8:
            data.append([float(v) for v in vals[:8]])
    return np.array(data)


def _get_jpole_coefficients(J):
    """
    Return the J-pole expansion coefficients (bzj, czj) for the plasma
    dispersion function Z(zeta) = sum(b_j / (zeta - c_j)).

    Parameters
    ----------
    J : int
        Number of poles (4, 8, or 12).

    Returns
    -------
    bzj : ndarray, shape (J,)
    czj : ndarray, shape (J,)
    """
    if J == 8:
        # Ronnmark1982, 8-pole
        bzj = np.zeros(8, dtype=complex)
        czj = np.zeros(8, dtype=complex)
        bzj[0] = -1.734012457471826e-2 - 4.630639291680322e-2j
        bzj[1] = -7.399169923225014e-1 + 8.395179978099844e-1j
        bzj[2] = 5.840628642184073 + 9.536009057643667e-1j
        bzj[3] = -5.583371525286853 - 1.120854319126599e1j
        czj[0] = 2.237687789201900 - 1.625940856173727j
        czj[1] = 1.465234126106004 - 1.789620129162444j
        czj[2] = 0.8392539817232638 - 1.891995045765206j
        czj[3] = 0.2739362226285564 - 1.941786875844713j
        bzj[4:8] = np.conj(bzj[0:4])
        czj[4:8] = -np.conj(czj[0:4])

    elif J == 12:
        # from Cal_J_pole_bjcj.m, Xie2016
        bzj = np.zeros(12, dtype=complex)
        czj = np.zeros(12, dtype=complex)
        bzj[0] = -0.00454786121654587 - 0.000621096230229454j
        bzj[1] = 0.215155729087593 + 0.201505401672306j
        bzj[2] = 0.439545042119629 + 4.16108468348292j
        bzj[3] = -20.2169673323552 - 12.8855035482440j
        bzj[4] = 67.0814882450356 + 20.8463458499504j
        bzj[5] = -48.0146738250076 + 107.275614092570j
        czj[0] = -2.97842916245164 - 2.04969666644050j
        czj[1] = 2.25678378396682 - 2.20861841189542j
        czj[2] = -1.67379985617161 - 2.32408519416336j
        czj[3] = -1.15903203380422 - 2.40673940954718j
        czj[4] = 0.682287636603418 - 2.46036501461004j
        czj[5] = -0.225365375071350 - 2.48677941704753j
        bzj[6:12] = np.conj(bzj[0:6])
        czj[6:12] = -np.conj(czj[0:6])

    elif J == 4:
        # Martin1980
        bzj = np.zeros(4, dtype=complex)
        czj = np.zeros(4, dtype=complex)
        bzj[0] = 0.546796859834032 + 0.037196505239277j
        bzj[1] = -1.046796859834027 + 2.101852568038518j
        czj[0] = 1.23588765343592 - 1.21498213255731j
        czj[1] = -0.378611612386277 - 1.350943585432730j
        bzj[2:4] = np.conj(bzj[0:2])
        czj[2:4] = -np.conj(czj[0:2])

    else:
        raise ValueError(f"Unsupported J={J}. Choose J=4, 8, or 12.")

    return bzj, czj

