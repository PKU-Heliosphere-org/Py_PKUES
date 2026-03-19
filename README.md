# Py_PKUES

A Python implementation of **PDRK** (Plasma Dispersion Relation solver, Kinetic) and **PKUES** — a general kinetic dispersion relation solver for magnetized plasma with bi-Maxwellian equilibrium distribution.

Originally developed in MATLAB by Hua-sheng Xie (IFTS-ZJU). extended by Xingyu Zhu, Die Duan, and Jiansen He, and converted to Python by Xie Haoen (Peking University).

This Python version add some new functions.

Version1 (2026.3.19)

## Overview

PDRK solves the linearized Vlasov–Maxwell system for an arbitrary number of plasma species with bi-Maxwellian (or loss-cone) equilibrium distributions and parallel drifts. The dispersion equation is reformulated as a matrix eigenvalue problem using J-pole approximations of the plasma dispersion function Z(ζ), enabling simultaneous extraction of all wave modes at each (k, θ) point.  

Key capabilities:

- **Electromagnetic (EM3D) and electrostatic (ES3D) modes** in a uniform magnetized plasma
- **Arbitrary number of particle species** with independent temperatures, anisotropies, and drift velocities
- **Loss-cone distributions** (controlled by α and Δ parameters)
- **1D or 2D parameter scans** over wave number k, propagation angle θ, or parallel/perpendicular components k∥, k⊥
- **Interactive mode selection** with robust Hungarian-algorithm-based tracking across avoided crossings
- **Full polarization analysis**: δE, δB, δV, δJ components, magnetic helicity, compressibility, Elsässer variables
- **Velocity distribution function (VDF) perturbation**: 3D δf computation with VTK output for ParaView visualization

## References

> H. S. Xie & Y. Xiao, *PDRK: A General Kinetic Dispersion Relation Solver for Magnetized Plasma*, Plasma Science and Technology, Vol. 18, No. 2, p97 (2016). [DOI:10.1088/1009-0630/18/2/01](https://doi.org/10.1088/1009-0630/18/2/01)

> Luo, Q., Zhu, X., He, J., Cui, J., Lai, H., Verscharen, D., & Duan, D. (2022). Coherence of Ion Cyclotron Resonance in Damped Ion Cyclotron Waves in Space Plasmas. The Astrophysical Journal, 928(1), 36.

> He, J., Zhu, X., Luo, Q., Hou, C., Verscharen, D., Duan, D., ... & Yao, Z. (2022). Observations of rapidly growing whistler waves in front of space plasma shock due to resonance interaction between fluctuating electron velocity distributions and electromagnetic fields. The Astrophysical Journal, 941(2), 147.

> Zhu, X., He, J., Duan, D., Verscharen, D., Owen, C. J., Fedorov, A., ... & Horbury, T. S. (2023). Non-field-aligned Proton Beams and Their Roles in the Growth of Fast Magnetosonic/Whistler Waves: Solar Orbiter Observations. The Astrophysical Journal, 953(2), 161.

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib

## Installation

Clone the repository and ensure the dependencies are installed:

```bash
git clone https://github.com/<your-username>/PDRK-Python.git
cd PDRK-Python
pip install numpy scipy matplotlib
```

No build step is required — all modules are pure Python.

## File Structure

```
PDRK-Python/
│   |── pdrk.in                    # Species input file
│   ├── pkues_main.py              # Main driver program
│   ├── pkues_setup.m              # MATLAB setup reference
│   ├── pdrk_initialize.py         # Parameter initialization
│   ├── pdrk_kernel.py             # Core eigenvalue solver
│   ├── pdrk_em3d_matrix.py        # EM3D dispersion matrix constructor
│   ├── pdrk_es3d_matrix.py        # ES3D dispersion matrix constructor
│   ├── pkues_velocity.py          # Velocity perturbation computation
│   ├── pkues_plot_all.py          # Interactive mode selector & tracer
│   ├── pkues_output.py            # Polarization plotting orchestrator
│   ├── pkues_plot_growth_rate.py  # Growth rate plots
│   ├── pkues_plot_comp_velocity.py# Component velocity plots
│   ├── pkues_add_polarization_1.py# Electron phase difference plots
│   ├── pkues_add_polarization_2.py# Ion phase difference plots
│   └── pkues_write_f_SI.py        # VDF computation & VTK output
├── output/                        # Generated figures and data
└── README.md
```

## Input File Format

The species parameters are read from `input/pdrk.in`. Each row defines one particle species (the first row is a header and is skipped):

```
qs0    ms0    ns0        Tzs0   Tps0   alphas  Deltas  vds0
1.0    1.0    5.0e6      50.0   50.0   1.0     1.0     0.0
-1.0   0.00054 5.0e6    50.0   50.0   1.0     1.0     0.0
```

| Column   | Description                              | Unit / Normalization       |
|----------|------------------------------------------|----------------------------|
| `qs0`    | Charge                                   | in units of proton charge *e* |
| `ms0`    | Mass                                     | in units of proton mass *mₚ* |
| `ns0`    | Number density                           | m⁻³                       |
| `Tzs0`   | Parallel temperature                     | eV                         |
| `Tps0`   | Perpendicular temperature                | eV                         |
| `alphas` | Loss-cone size parameter (1 = no cone)   | dimensionless              |
| `Deltas` | Loss-cone depth (0 = max, 1 = none)      | dimensionless              |
| `vds0`   | Parallel drift velocity                  | in units of *c*            |

## Quick Start

### Interactive Mode (Recommended)

```python
from pkues_main import run_pdrk, pkues_setup_default

config = pkues_setup_default()
config['B0'] = 1.21e-7        # Background magnetic field [T]
config['pa1'] = 0.01          # Scan k*c/ωp from 0.01
config['pa2'] = 1.5           #                  to 1.5
config['dpa'] = 0.01          # Step size
config['ipa'] = 1             # Scan parameter: k
config['ipb'] = 1             # Fixed parameter: same (1D scan)
config['par'][1] = 20.0       # Propagation angle θ = 20°

result = run_pdrk(config, interactive=True)
```

A scatter plot of all eigenvalue solutions appears. Click to select wave modes of interest, then press **Enter** to trace the selected dispersion surfaces and compute full polarization properties.

**Interactive controls:**

| Key / Action | Effect |
|:---:|---|
| Left click | Select nearest solution point |
| Right click | Remove last selected point |
| `Enter` | Confirm selection, trace & plot |
| `r` | Remove last point |
| `c` | Clear all selections |
| `q` | Quit without tracing |
| `a` | Auto-select the most unstable mode |
| `t` | Toggle tracking method (Simple ↔ Hungarian) |

### Batch Mode

```python
import numpy as np
from pkues_main import run_pdrk, pkues_setup_default

config = pkues_setup_default()
# ... set config parameters ...

wpdat = np.array([
    [0.5, 0.0, 1.2 + 0.1j],    # Mode 1: [pa_start, pb_start, ω/ωc1]
    [0.3, 0.0, -0.8 + 0.05j],   # Mode 2
])

result = run_pdrk(config, wpdat=wpdat, interactive=False)
```

### Re-select Modes from Saved Data

```python
from pkues_main import load_and_reselect

wws, wpdat = load_and_reselect('./output/pdrk_xxx_data.npz')
```

## Scan Parameter Options

The scan type is controlled by `(ipa, ipb)`:

| `(ipa, ipb)` | Scan mode |
|:---:|---|
| (1, 1) | Scan *k*, fixed θ |
| (2, 2) | Scan θ, fixed *k* |
| (1, 2) | 2D scan (*k*, θ) |
| (3, 3) | Scan *k*∥, fixed *k*⊥ |
| (4, 4) | Scan *k*⊥, fixed *k*∥ |
| (3, 4) | 2D scan (*k*∥, *k*⊥) |

Set `iloga=1` / `ilogb=1` for logarithmic scan grids (10^pa1 to 10^pa2).

## Output: Polarization Properties

After mode selection, the code computes and plots the following quantities as functions of the scan parameter:

- **Dispersion relation**: ω_r/ω_c1 and γ/ω_c1 vs. *k*
- **Electric field polarization**: δE_x, δE_y, δE_z (normalized by B₀V_A)
- **Magnetic field polarization**: δB_x, δB_y, δB_z (normalized by B₀)
- **Magnetic helicity**: σ_m = 2Im(δB_x* · δB_y) / (|δB_x|² + |δB_y|²)
- **Magnetic compressibility**: C_B = |δB∥|² / |δB|²
- **Current density**: δJ_x, δJ_y, δJ_z
- **Velocity perturbation**: δV per species, with Elsässer variables Z⁺, Z⁻
- **Phase differences**: between E/B components and velocity components (ions and electrons)
- **Energy partition**: electric vs. magnetic energy density

## Velocity Distribution Function (VDF)

The code can compute the perturbed velocity distribution function δf for any selected wave mode and particle species, and output 3D VTK files for visualization in ParaView.

### Configuration

```python
config['idf'] = 1              # Enable VDF computation
config['jpa_df'] = 59          # k-point index for VDF
config['jpb_df'] = 0           # Second scan parameter index
config['jpl_df'] = 0           # Wave mode index (0-based)
config['s_df'] = 0             # Species index (0-based)
config['vdf_config'] = {
    'ampl': 0.01,              # Perturbation amplitude (dimensionless)
    'vxrange': (-3, 3),        # Velocity range in vA units
    'vyrange': (-3, 3),
    'vzrange': (-3, 3),
    'vxsteps': 50,             # Grid resolution per axis
    'vysteps': 50,
    'vzsteps': 50,
    'damping': False,          # Include γ in time evolution
    'const_r': True,           # Use constant ω_r for time evolution
    'periods': True,           # Time unit = wave period
    'num_periods': 1,          # Number of periods to simulate
    'timesteps': 20,           # Time steps per period
}
```

### VTK Output

For each time step, two VTK files are generated:

1. **`pdrk_dist_deltaf_(kdi=X.XXX)(ampl=X.XX)_NNN.vtk`** — Structured grid in (v_x/v_A, v_y/v_A, v_z/v_A) space with scalar fields `f0+deltaf` and `deltaf`
2. **`pdrk_normEB_(kdi=X.XXX)(ampl=X.XX)_NNN.vtk`** — Polydata with normalized δE and δB direction lines

Open the `.vtk` files in [ParaView](https://www.paraview.org/) to render isosurfaces (Contour filter), slices, volume renderings, or animations of the distribution function evolution.

## Solver Details

- The plasma dispersion function Z(ζ) is approximated by a J-pole rational expansion: Z(ζ) ≈ Σⱼ bⱼ/(ζ − cⱼ), with J = 4, 8, or 12 poles
- The linearized Vlasov–Maxwell system is cast as a matrix eigenvalue problem M·X = ω·X, where the matrix dimension is NN = 3×(S×(2N+1)×J + 1) + 6
- For the first pass (all roots), `numpy.linalg.eigvals` or `scipy.sparse.linalg.eigs` is used
- For the polarization pass, `sparse_eigs` with shift-invert (σ = ω_guess) extracts the single eigenvalue closest to the traced solution, along with its eigenvector
- The physical fields (δE, δB) occupy the last 6 elements of the eigenvector; all other elements are auxiliary J-pole expansion coefficients
- An eigenvector reliability check (residual ‖Mv − λv‖/|λ| > 10⁻³ → NaN) prevents spurious results near k → 0 where the matrix becomes ill-conditioned

## Acknowledgments

- **Hua-sheng Xie** (IFTS-ZJU / FRI-ENN) — Original MATLAB PDRK code
- **Xingyu Zhu, Die Duan, Jiansen He** (Peking University) — PKUES extensions: polarization analysis, VDF computation, VTK output
- **Richard Denton** (Dartmouth), **Xin Tao** (USTC), **Jin-song Zhao** (PMO) — Contributions to the original PDRK framework
- **Haoen Xie** (Peking University) -- convert to Python version

## License

Please cite the PDRK paper (Xie & Xiao, 2016) if you use this code in published research.

