"""
pkues_main_modified.py
Main driver program for PDRK (Plasma Dispersion Relation solver, Kinetic).

Workflow:
  1. Read configuration (pkues_setup)
  2. Initialize plasma parameters (pdrk_initialize)
  3. Solve eigenvalue problem across scan parameters (pdrk_kernel)
  4. Interactive visualization and mode selection (pdrk_plot_all)
  5. Optionally compute polarization and other wave properties

Converted from the MATLAB pdrk suite by Hua-sheng XIE.
Enhanced with Hungarian-algorithm robust mode tracking.

Usage:
  python pkues_main_modified.py
  # or import and call run_pdrk(config) programmatically.
"""

import numpy as np
import time
import os
import sys

from pdrk_initialize import pdrk_initialize
from pdrk_kernel import pdrk_kernel
from pkues_plot_all import pdrk_plot_all_interactive, pdrk_plot_select


def pkues_setup_default():
    """
    Default configuration dict (equivalent to MATLAB pkues_setup.m).

    Users should modify this or provide their own config dict.

    Returns
    -------
    config : dict
    """
    config = {
        # ----- Solver settings -----
        'iem': 1,         # 1 = electromagnetic, 0 = electrostatic
        'N': 3,           # Number of harmonics (-N..N for Bessel functions)
        'J': 8,           # Number of J-poles for Z function (4, 8, or 12)
        'sp': 0,          # 0 = eig (all roots), 1 = eigs (sparse, nw roots near wg0)
        'wg0': 0.0,       # Initial guess for omega / wcs1 (used when sp=1)
        'iout': 2,        # Output level: 0 = basic, 1 = save data, 2 = polarization

        # ----- Magnetic field -----
        'B0': 1.21e-7,     # Background magnetic field [T]

        # ----- Scan parameters -----
        # par = [k*c/wp, theta, kz*c/wp, kx*c/wp] (the scan variables)
        # par[0]: k*c/omega_p1,  par[1]: theta (degree)
        # par[2]: kz*c/omega_p1, par[3]: kx*c/omega_p1
        'par': np.array([0.1, 10.0, 0.0, 0.0]),

        # ipa, ipb: which parameter to scan
        #   (1,1): scan k, fixed theta
        #   (2,2): scan theta, fixed k
        #   (1,2): 2D scan (k, theta)
        #   (3,3): scan kz, fixed kx
        #   (4,4): scan kx, fixed kz
        #   (3,4): 2D scan (kz, kx)
        'ipa': 1,
        'ipb': 1,

        # Log scale: 0=linear, 1=log10
        'iloga': 0,
        'ilogb': 0,

        # Scan range for parameter a
        'pa1': 0.01,      # start
        'pa2': 1.0,       # end
        'dpa': 0.002,      # step

        # Scan range for parameter b (used only when ipa != ipb)
        'pb1': 0.0,
        'pb2': 90.0,
        'dpb': 1.0,

        # ----- Output settings -----
        'savepath': './output/',
    }
    return config


def run_pdrk(config=None, wpdat=None, interactive=True,
             use_hungarian=True, pred_weight=0.70,
             rex=1.0, rey=1.0, rez=1.0):
    """
    Run the complete PDRK workflow.

    Parameters
    ----------
    config : dict or None
        Configuration dictionary. If None, uses default.
    wpdat : ndarray or None, shape (npl, 3)
        Pre-defined starting points for mode selection.
        If None and interactive=True, user selects interactively.
    interactive : bool
        If True, launch interactive mode selector.
        If False and wpdat is not None, run batch mode.
    use_hungarian : bool
        Use Hungarian-algorithm robust tracking (default True).
    pred_weight : float
        Prediction weight for Hungarian cost matrix (0 to 1).
    rex, rey, rez : float
        Rescaling factors for plot axes.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'ww':   all eigenvalue solutions, shape (npa, npb, nw) [rad/s]
        - 'wwn':  normalized solutions (omega / wcs1)
        - 'wws':  traced dispersion surfaces (normalized), or None
        - 'wpdat': starting points used for tracing, or None
        - 'pa', 'pb': scan parameter arrays
        - 'init': full initialization dict
        - 'runtime': computation time [s]
    """
    if config is None:
        config = pkues_setup_default()

    # ------------------------------------------------------------------
    # Step 1: Initialize
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  PDRK - Plasma Dispersion Relation solver (Kinetic)")
    print("  Python version with Hungarian-algorithm mode tracking")
    print("=" * 65)

    t0 = time.time()
    init = pdrk_initialize(config)

    if init.get('ireturn', 0) == 1:
        print(f"ERROR: Unsupported scan type (ipa={config['ipa']}, "
              f"ipb={config['ipb']}). Exiting.")
        return None

    print(f"\n  Species: S={init['S']}")
    print(f"  Harmonics: N={init['N']}, J-poles: J={init['J']}")
    print(f"  Mode: {'EM3D' if init['iem']==1 else 'ES3D'}")
    print(f"  Scan: {init['strscan']}")
    print(f"  Matrix size: NN={init['NN']}, nw={init['nw']}")
    print(f"  Scan points: npa={init['npa']}, npb={init['npb']}")
    print(f"  B0={config['B0']:.2e} T, vA/c={init['vA']/np.sqrt(init['c2']):.4e}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Solve eigenvalue problem (kernel)
    # ------------------------------------------------------------------
    print("  Computing dispersion solutions...")
    t1 = time.time()

    kernel_result = pdrk_kernel(init, icalp=0)

    runtime = time.time() - t1
    print(f"  Done. Runtime: {runtime:.2f} s")

    ww = kernel_result['ww']          # shape (npa, npb, nw), [rad/s]
    wwn = ww / init['wcs1']           # normalized
    pa = init['pa']
    pb = init['pb']
    npa = init['npa']
    npb = init['npb']

    # Figure filename string
    figstr = f"iem{init['iem']}_S{init['S']}_N{init['N']}_J{init['J']}"

    # ------------------------------------------------------------------
    # Step 3: Visualization and mode selection
    # ------------------------------------------------------------------
    wws = None
    wpdat_out = None

    if interactive and wpdat is None:
        # Interactive mode: user clicks to select modes
        wws, wpdat_out = pdrk_plot_all_interactive(
            ww, pa, pb, npa, npb,
            init['ipa'], init['ipb'],
            iloga=init['iloga'], ilogb=init['ilogb'],
            wcs1=init['wcs1'],
            strpa=init['strpa'], strpb=init['strpb'],
            betasz=init['betasz'], betasp=init['betasp'],
            alphas=init['alphas'], Deltas=init['Deltas'],
            vA=init['vA'], c2=init['c2'],
            S=init['S'], N=init['N'], J=init['J'], iem=init['iem'],
            par=init['par'], ipbtmp=init['ipbtmp'],
            runtime=runtime,
            savepath=init['savepath'], figstr=figstr,
            rex=rex, rey=rey, rez=rez,
            pred_weight=pred_weight,
            init=init,
            run_pkues_output=True)

    elif wpdat is not None:
        # Batch mode: trace given wpdat
        wws = pdrk_plot_select(
            wwn, pa, pb, npa, npb,
            init['ipa'], init['ipb'], wpdat,
            iloga=init['iloga'], ilogb=init['ilogb'],
            wcs1=init['wcs1'],
            strpa=init['strpa'], strpb=init['strpb'],
            betasz=init['betasz'], betasp=init['betasp'],
            alphas=init['alphas'], Deltas=init['Deltas'],
            vA=init['vA'], c2=init['c2'],
            S=init['S'], N=init['N'], J=init['J'], iem=init['iem'],
            par=init['par'], ipbtmp=init['ipbtmp'],
            runtime=runtime,
            savepath=init['savepath'], figstr=figstr,
            rex=rex, rey=rey, rez=rez,
            use_hungarian=use_hungarian, pred_weight=pred_weight)
        wpdat_out = wpdat

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.2f} s")

    savepath = init['savepath']
    os.makedirs(savepath, exist_ok=True)

    # Save data
    save_dict = {
        'ww': ww,
        'wwn': wwn,
        'pa': pa,
        'pb': pb,
        'npa': npa,
        'npb': npb,
        'wcs1': init['wcs1'],
        'runtime': runtime,
    }
    if wws is not None:
        save_dict['wws'] = wws
    if wpdat_out is not None:
        save_dict['wpdat'] = wpdat_out

    np.savez(os.path.join(savepath, f'pdrk_{figstr}_data.npz'), **save_dict)
    print(f"  Data saved to: {savepath}pdrk_{figstr}_data.npz")

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------
    result = {
        'ww': ww,
        'wwn': wwn,
        'wws': wws,
        'wpdat': wpdat_out,
        'pa': pa,
        'pb': pb,
        'init': init,
        'runtime': runtime,
        'kernel_result': kernel_result,
    }

    return result


def load_and_reselect(npz_path, config=None,
                      wpdat=None, interactive=True,
                      use_hungarian=True, pred_weight=0.70,
                      rex=1.0, rey=1.0, rez=1.0):
    """
    Load previously saved PDRK results and re-run the mode selection
    (without re-computing the eigenvalue problem).

    This is useful when:
      - The computation took a long time and you want to re-select modes
      - You want to try different tracking methods on the same data

    Parameters
    ----------
    npz_path : str
        Path to the saved .npz file from a previous run.
    config : dict or None
        Configuration (needed for display parameters). If None, uses default.
    wpdat : ndarray or None
        Pre-set starting points. If None, interactive selection.
    interactive : bool
        Launch interactive mode selector.
    use_hungarian : bool
        Use Hungarian tracking.
    pred_weight : float
        Prediction weight.
    rex, rey, rez : float
        Rescaling factors.

    Returns
    -------
    wws : ndarray
        Traced dispersion surfaces.
    wpdat_out : ndarray
        Starting points used.
    """
    data = np.load(npz_path, allow_pickle=True)
    ww = data['ww']
    wwn = data['wwn']
    pa = data['pa']
    pb = data['pb']
    npa = int(data['npa'])
    npb = int(data['npb'])
    wcs1 = float(data['wcs1'])
    runtime = float(data['runtime'])

    if config is None:
        config = pkues_setup_default()

    init = pdrk_initialize(config)

    figstr = f"iem{init['iem']}_S{init['S']}_N{init['N']}_J{init['J']}"

    if interactive and wpdat is None:
        wws, wpdat_out = pdrk_plot_all_interactive(
            ww, pa, pb, npa, npb,
            init['ipa'], init['ipb'],
            iloga=init['iloga'], ilogb=init['ilogb'],
            wcs1=wcs1,
            strpa=init['strpa'], strpb=init['strpb'],
            betasz=init['betasz'], betasp=init['betasp'],
            alphas=init['alphas'], Deltas=init['Deltas'],
            vA=init['vA'], c2=init['c2'],
            S=init['S'], N=init['N'], J=init['J'], iem=init['iem'],
            par=init['par'], ipbtmp=init['ipbtmp'],
            runtime=runtime,
            savepath=init['savepath'], figstr=figstr,
            rex=rex, rey=rey, rez=rez,
            wpdat=None, pred_weight=pred_weight)
    else:
        wws = pdrk_plot_select(
            wwn, pa, pb, npa, npb,
            init['ipa'], init['ipb'], wpdat,
            iloga=init['iloga'], ilogb=init['ilogb'],
            wcs1=wcs1,
            strpa=init['strpa'], strpb=init['strpb'],
            betasz=init['betasz'], betasp=init['betasp'],
            alphas=init['alphas'], Deltas=init['Deltas'],
            vA=init['vA'], c2=init['c2'],
            S=init['S'], N=init['N'], J=init['J'], iem=init['iem'],
            par=init['par'], ipbtmp=init['ipbtmp'],
            runtime=runtime,
            savepath=init['savepath'], figstr=figstr,
            rex=rex, rey=rey, rez=rez,
            use_hungarian=use_hungarian, pred_weight=pred_weight)
        wpdat_out = wpdat

    return wws, wpdat_out


# ======================================================================
# Command-line entry point
# ======================================================================
if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     PDRK — Plasma Dispersion Relation solver (Kinetic)        ║
║     Python version with interactive mode selection            ║
║     Hungarian-algorithm robust tracking at mode crossings     ║
╚═══════════════════════════════════════════════════════════════╝

Usage examples:
---------------

  # 1. Run with default config (interactive)
  python pkues_main.py

  # 2. Run programmatically with custom config
  from pkues_main_modified import run_pdrk, pkues_setup_default
  config = pkues_setup_default()
  config['B0'] = 1e-4          # modify B0
  config['pa1'] = 0.01         # scan k from 0.01
  config['pa2'] = 2.0          #            to 2.0
  config['dpa'] = 0.02         # step 0.02
  result = run_pdrk(config)

  # 3. Re-select modes from saved data (no re-computation)
  from pkues_main_modified import load_and_reselect
  wws, wpdat = load_and_reselect('./output/pdrk_xxx_data.npz')

  # 4. Batch mode with known wpdat
  import numpy as np
  wpdat = np.array([
      [0.5, 0.0, 1.2+0.1j],   # mode 1 start
      [0.3, 0.0, -0.8+0.05j], # mode 2 start
  ])
  result = run_pdrk(config, wpdat=wpdat, interactive=False)

Input file:
-----------
  Species data is read from '../input/pdrk.in'
  Format: qs0  ms0  ns0  Tzs0  Tps0  alphas  Deltas  vds0
  (one line per species, first line is header)

Tracking methods:
-----------------
  Hungarian (default): Multi-point prediction + cost matrix + global
    optimal assignment. Robust at mode crossings.
  Simple: Original PCHIP + nearest-neighbor (per-mode independent).
    Faster but may misidentify modes at crossings.
""")

    # Default run
    config = pkues_setup_default()
    result = run_pdrk(config, interactive=True)

