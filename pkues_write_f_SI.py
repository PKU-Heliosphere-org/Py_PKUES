"""
pkues_write_f_SI.py
Velocity Distribution Function (VDF) computation and VTK output.

Computes:
  - f0: background bi-Maxwellian distribution
  - δf: perturbed distribution from linear kinetic theory
  - f = f0 + δf: total distribution

Outputs:
  - VTK structured grid file: f0+deltaf, deltaf on 3D velocity grid
  - VTK polydata file: normalized E and B field direction lines

Based on the MATLAB code by Xingyu Zhu and Jiansen He (2020-06-01).
Converted to Python with numpy vectorization.

Reference:
  H. S. Xie & Y. Xiao, PDRK, Plasma Science and Technology, 18(2), p97 (2016).
"""

import numpy as np
from scipy.special import jv as besselj
import os


# ======================================================================
#  VTK writers — byte-level faithful reproduction of MATLAB vtkwrite.m
# ======================================================================

def _vtkwrite_structured_grid(filename, X, Y, Z, scalar_dict):
    """
    Write a VTK STRUCTURED_GRID file (BINARY, big-endian float32).

    Reproduces the exact byte-level output of MATLAB vtkwrite.m:
        vtkwrite(filename, 'structured_grid', X, Y, Z,
                 'scalars', title1, s1, 'scalars', title2, s2,
                 'Precision', 20, 'BINARY')

    Key implementation notes
    ------------------------
    1) MATLAB vtkwrite writes DIMENSIONS as size(X,1) size(X,2) size(X,3).
       With our indexing='ij' meshgrid convention, X.shape = (nx, ny, nz),
       so DIMENSIONS = nx ny nz.  VTK interprets DIMENSIONS i j k with i
       varying fastest in the point list.

    2) MATLAB builds  output = [x(:)'; y(:)'; z(:)']  (3×N matrix)
       then calls  fwrite(fid, output, 'float', 'b')  which writes in
       column-major order → interleaved triples: x0,y0,z0, x1,y1,z1, ...
       where x(:) flattens in Fortran (column-major) order.

       In Python this is:
           x_flat = X.ravel(order='F')     # Fortran-order flatten
           y_flat = Y.ravel(order='F')
           z_flat = Z.ravel(order='F')
           coords = np.column_stack([x_flat, y_flat, z_flat])  # (N, 3)
           fid.write(coords.astype('>f4').tobytes())
       The (N,3) C-order layout naturally gives x0,y0,z0, x1,y1,z1, ...

    3) Scalar data is flattened in Fortran order: data.ravel(order='F').

    Parameters
    ----------
    filename : str
        Output VTK file path.
    X, Y, Z : ndarray, shape (nx, ny, nz)
        3D coordinate arrays (from meshgrid with indexing='ij').
    scalar_dict : dict
        {name: data_array} for each scalar field.  Same shape as X.
    """
    nx, ny, nz = X.shape
    n_points = nx * ny * nz

    with open(filename, 'wb') as fid:
        # ---- Header (ASCII portion) ----
        fid.write(b'# vtk DataFile Version 2.0\n')
        fid.write(b'VTK from Matlab\n')
        fid.write(b'BINARY\n')

        # ---- Dataset structure ----
        fid.write(b'DATASET STRUCTURED_GRID\n')
        fid.write(f'DIMENSIONS {nx} {ny} {nz}\n'.encode())
        fid.write(f'POINTS {n_points} float\n'.encode())

        # ---- Coordinate data (interleaved, Fortran-order flatten) ----
        x_flat = X.ravel(order='F').astype(np.float32)
        y_flat = Y.ravel(order='F').astype(np.float32)
        z_flat = Z.ravel(order='F').astype(np.float32)
        # column_stack → (N, 3) → C-order bytes = x0,y0,z0, x1,y1,z1, ...
        coords = np.column_stack([x_flat, y_flat, z_flat]).astype('>f4')
        fid.write(coords.tobytes())

        # ---- Scalar fields ----
        fid.write(f'\nPOINT_DATA {n_points}'.encode())

        for title, data in scalar_dict.items():
            fid.write(f'\nSCALARS {title} float\n'.encode())
            fid.write(b'LOOKUP_TABLE default\n')
            # Flatten in Fortran order, same as MATLAB data(:)
            flat = data.ravel(order='F').astype('>f4')
            fid.write(flat.tobytes())


def _vtkwrite_EB_polydata(filename, E_direc, B_direc):
    """
    Write a VTK POLYDATA file (ASCII) with E and B direction lines.

    Reproduces the MATLAB code in pkues_write_f_SI_unit.m lines 163-187.

    Parameters
    ----------
    filename : str
        Output VTK file path.
    E_direc : array-like, shape (3,)
        Normalized E-field direction (real-valued, scaled to vxrange).
    B_direc : array-like, shape (3,)
        Normalized B-field direction (real-valued, scaled to vxrange).
    """
    E_direc = np.real(np.asarray(E_direc, dtype=float))
    B_direc = np.real(np.asarray(B_direc, dtype=float))

    with open(filename, 'w') as fid:
        fid.write('# vtk DataFile Version 2.0\n')
        fid.write('VTK from Matlab\n')
        fid.write('ASCII\n')
        fid.write('DATASET POLYDATA\n')
        fid.write('POINTS 3 float\n')

        fmt = '%.15f %.15f %.15f\n'
        fid.write(fmt % (0.0, 0.0, 0.0))
        fid.write(fmt % (E_direc[0], E_direc[1], E_direc[2]))
        fid.write(fmt % (B_direc[0], B_direc[1], B_direc[2]))

        fid.write('\nLINES 2 6\n')
        fid.write('2 0 1\n')
        fid.write('2 0 2\n')

        fid.write('\nCELL_DATA 2\n')
        fid.write('SCALARS cell_scalars float 1\n')
        fid.write('LOOKUP_TABLE my_table\n')
        fid.write('0.0\n')
        fid.write('1.0\n')
        fid.write('\nLOOKUP_TABLE my_table 2\n')
        fid.write('0.0 1.0 0.0 1.0\n')
        fid.write('1.0 1.0 0.0 1.0')


# ======================================================================
#  VDF computation core
# ======================================================================

def _compute_vdf_snapshot(omega, kx, kz, dEx, dEy, dEz,
                          s_idx, N, qs, ms, ns0, wcs, vds, vthzS, vthpS,
                          vA, ampl,
                          vx_si, vy_si, vz_si,
                          time, damping, const_r, timesteps, time_idx):
    """
    Compute f0, Re(δf), Im(δf) on a 3D velocity grid for one time snapshot.

    Physics is identical to MATLAB pkues_write_f_SI_unit.m.
    All velocity inputs are in SI [m/s].

    Returns
    -------
    f0 : ndarray, shape (nvx, nvy, nvz)
    deltaf_real : ndarray, same shape
    deltaf_imag : ndarray, same shape
    """
    j = s_idx
    x = omega  # complex frequency [rad/s]

    # 3D meshgrids  (indexing='ij': first dim = vx, second = vy, third = vz)
    VX, VY, VZ = np.meshgrid(vx_si, vy_si, vz_si, indexing='ij')

    vpar = VZ
    vperp = np.sqrt(VX**2 + VY**2)

    # Azimuthal angle  (matching MATLAB: acos(vx/vperp), flip if vy<0)
    phi = np.zeros_like(VX)
    nonzero = vperp > 0
    phi[nonzero] = np.arccos(np.clip(VX[nonzero] / vperp[nonzero], -1, 1))
    phi[VY < 0] = 2.0 * np.pi - phi[VY < 0]

    # Species parameters
    vth_perp = vthpS[j]
    vth_para = vthzS[j]
    v_drift = vds[j]
    wc_s = wcs[j]
    q_s = qs[j]
    m_s = ms[j]
    n_s = ns0[j]

    # Background distribution f0  (bi-Maxwellian)
    exp_perp = np.exp(-vperp**2 / vth_perp**2)
    exp_para = np.exp(-(vpar - v_drift)**2 / vth_para**2)
    norm_f = n_s / (np.pi**1.5 * vth_perp**2 * vth_para)
    f0 = norm_f * exp_perp * exp_para

    # df/dv_parallel  (MATLAB line 68-70)
    dfdvpar = (-2.0 * (vpar - v_drift) * exp_perp * exp_para *
               n_s / (np.pi**1.5 * vth_perp**2 * vth_para**3))

    # df/dv_perp  (MATLAB line 72-74)
    dfdvperp = (-2.0 * vperp * exp_perp * exp_para *
                n_s / (np.pi**1.5 * vth_perp**4 * vth_para))

    # Bessel argument
    z = kx * vperp / wc_s

    # Sum over harmonics m = -N..N  (MATLAB line 83-109)
    deltaf = np.zeros_like(VX, dtype=complex)

    for m in range(-N, N + 1):
        a = float(m) * wc_s - x + kz * vpar

        UStix = dfdvperp + kz * (vperp * dfdvpar - vpar * dfdvperp) / x
        VStix = kx * (vperp * dfdvpar - vpar * dfdvperp) / x

        # Regularize denominators
        denom_cyc = wc_s**2 - a**2
        denom_cyc[np.abs(denom_cyc) < 1e-30] = 1e-30
        a_safe = a.copy()
        a_safe[np.abs(a_safe) < 1e-30] = 1e-30

        comp1 = dEx * UStix * (a * 1j * np.cos(phi) - wc_s * np.sin(phi)) / denom_cyc
        comp2 = dEy * UStix * (a * 1j * np.sin(phi) + wc_s * np.cos(phi)) / denom_cyc
        comp3 = -1j * dEz * dfdvpar / a_safe
        comp4 = -dEz * VStix * (a * 1j * np.cos(phi) - wc_s * np.sin(phi)) / denom_cyc

        Jm = besselj(m, z)
        phase = np.exp(1j * z * np.sin(phi)) * np.exp(-1j * float(m) * phi)

        deltaf -= (q_s / m_s) * phase * Jm * ampl * (comp1 + comp2 + comp3 + comp4)

    # Time evolution  (MATLAB line 111-125)
    if const_r:
        if damping:
            deltaf *= np.exp(-1j * time * x)
        else:
            deltaf *= np.exp(-1j * time * np.real(x))
    else:
        deltaf *= np.exp(-1j * 2.0 * np.pi * float(time_idx) / float(timesteps))

    return f0, np.real(deltaf), np.imag(deltaf)


# ======================================================================
#  Main entry: compute VDF time series and write VTK files
# ======================================================================

def run_vdf_for_mode(jpa, jpb, jpl, s_idx, init, wws, Pola_SI, scaling,
                     kx, kz, vdf_config, savepath='./', figstr=''):
    """
    Compute VDF for all time snapshots and write VTK files.

    Equivalent to MATLAB's ``run ../modules/pkues_write_f_SI_unit``.

    For each time step, writes:
      1. ``pdrk_dist_deltaf_(kdi=...)(ampl=...)_NNN.vtk``
         STRUCTURED_GRID with scalars 'f0+deltaf' and 'deltaf'
      2. ``pdrk_normEB_(kdi=...)(ampl=...)_NNN.vtk``
         POLYDATA with normalized E and B direction lines

    Parameters
    ----------
    jpa, jpb, jpl : int
        Indices for scan parameter and mode.
    s_idx : int
        Species index (0-based).
    init : dict
        Initialization parameters from pdrk_initialize.
    wws : ndarray
        Traced dispersion surfaces (normalized by wcs1).
    Pola_SI : ndarray
        SI-unit polarization array.
    scaling : ndarray
        Scaling factor array.
    kx, kz : float
        Wave numbers [m^-1].
    vdf_config : dict
        VDF configuration dict with keys:
        'ampl', 'vxrange', 'vyrange', 'vzrange',
        'vxsteps', 'vysteps', 'vzsteps',
        'damping', 'const_r', 'periods',
        'num_periods', 'timesteps'
    savepath : str
        Output directory for VTK files.
    figstr : str
        Figure string identifier (unused, kept for interface compatibility).
    """
    os.makedirs(savepath, exist_ok=True)

    wcs1 = init['wcs1']
    vA = init['vA']
    cwp = init['cwp']
    N = init['N']

    # Complex frequency [rad/s]
    omega = wws[jpa, jpb, jpl] * wcs1
    k = np.sqrt(kx**2 + kz**2)
    kdi = k * cwp  # normalized wave number

    # UN-scaled field perturbations  (MATLAB: Pola_SI / scaling_factor)
    scaling_factor = 1.0
    dEx = Pola_SI[jpa, jpb, jpl, 0] / scaling_factor
    dEy = Pola_SI[jpa, jpb, jpl, 1] / scaling_factor
    dEz = Pola_SI[jpa, jpb, jpl, 2] / scaling_factor
    dBx = Pola_SI[jpa, jpb, jpl, 3] / scaling_factor
    dBy = Pola_SI[jpa, jpb, jpl, 4] / scaling_factor
    dBz = Pola_SI[jpa, jpb, jpl, 5] / scaling_factor

    # VDF configuration
    ampl = vdf_config['ampl']
    vxrange = vdf_config['vxrange']
    vyrange = vdf_config['vyrange']
    vzrange = vdf_config['vzrange']
    vxsteps = vdf_config['vxsteps']
    vysteps = vdf_config['vysteps']
    vzsteps = vdf_config['vzsteps']
    damping = vdf_config['damping']
    const_r = vdf_config['const_r']
    periods = vdf_config['periods']
    num_periods = vdf_config['num_periods']
    timesteps = vdf_config['timesteps']

    # Build 1D velocity grids  (MATLAB line 53-57: linspace-equivalent)
    nvx = vxsteps + 1
    nvy = vysteps + 1
    nvz = vzsteps + 1
    vx_va = np.linspace(vxrange[0], vxrange[1], nvx)
    vy_va = np.linspace(vyrange[0], vyrange[1], nvy)
    vz_va = np.linspace(vzrange[0], vzrange[1], nvz)

    # SI velocity arrays for physics computation
    vx_si = vx_va * vA
    vy_si = vy_va * vA
    vz_si = vz_va * vA

    # 3D meshgrids in vA units (for VTK coordinate output)
    # indexing='ij': shape = (nvx, nvy, nvz), first dim = vx
    # This matches MATLAB convention: VXarray(ii+1, kk+1, ll+1) = vx
    VX_va, VY_va, VZ_va = np.meshgrid(vx_va, vy_va, vz_va, indexing='ij')

    # Track max/min  (MATLAB line 24-27)
    maxf = -999.0
    minf = 999.0
    maxdfim = -999.0
    mindfim = 999.0

    x = omega

    print(f"\n  ========== VDF Computation ==========")
    print(f"  jpa={jpa}, jpb={jpb}, jpl={jpl}, species={s_idx + 1}")
    print(f"  kdi = {kdi:.4f}")
    print(f"  wr/wci = {np.real(x / wcs1):.4f}")
    print(f"  wi/wci = {np.imag(x / wcs1):.4f}")
    print(f"  ampl = {ampl}")

    total_steps = num_periods * timesteps

    for timerun in range(total_steps + 1):

        # Compute time  (MATLAB line 33-37)
        if periods:
            time = 2.0 * np.pi * float(timerun) / (np.abs(np.real(x)) * float(timesteps))
        else:
            time = 20.0 * np.pi * float(timerun) / (1.0 * float(timesteps))

        print(f"  time check {time:.6e}")

        # File names matching MATLAB convention
        vtk_fname = os.path.join(
            savepath,
            f'pdrk_dist_deltaf_(kdi={kdi:.3f})(ampl={ampl:.2f})_{timerun:03d}.vtk')
        eb_fname = os.path.join(
            savepath,
            f'pdrk_normEB_(kdi={kdi:.3f})(ampl={ampl:.2f})_{timerun:03d}.vtk')

        print(f"  kdi={kdi:.3f};wr/wci={np.real(x / wcs1):.3f};"
              f"wi/wci={np.imag(x / wcs1):.3f}")

        # Compute VDF
        f0, df_real, df_imag = _compute_vdf_snapshot(
            omega, kx, kz, dEx, dEy, dEz,
            s_idx, N,
            init['qs'], init['ms'], init['ns0'],
            init['wcs'], init['vds'],
            init['vthzS'], init['vthpS'],
            vA, ampl,
            vx_si, vy_si, vz_si,
            time, damping, const_r, timesteps, timerun)

        farray = f0 + df_real
        dfarray = df_real

        # Update max/min  (MATLAB line 127-130, but vectorized)
        maxf = max(maxf, np.max(farray))
        minf = min(minf, np.min(farray))
        maxdfim = max(maxdfim, np.max(df_imag))
        mindfim = min(mindfim, np.min(df_imag))

        print(f"  # Maximum value of f: {maxf}")
        print(f"  # Minimum value of f: {minf}")
        print(f"  # Maximum value of Im(delta f): {maxdfim}")
        print(f"  # Minimum value of Im(delta f): {mindfim}")

        # ---- Write VTK: structured grid with VDF ----
        # MATLAB: vtkwrite(filename, 'structured_grid', VXarray/vA, VYarray/vA,
        #                  VZarray/vA, 'scalars', 'f0+deltaf', farray,
        #                  'scalars', 'deltaf', dfarray, 'Precision', 20, 'BINARY')
        _vtkwrite_structured_grid(
            vtk_fname, VX_va, VY_va, VZ_va,
            {'f0+deltaf': farray, 'deltaf': dfarray})
        print(f"    VTK saved: {vtk_fname}")

        # ---- Write VTK: E and B field directions ----
        # MATLAB line 156-157:
        # E_direc = vxrange(2) * real([dEx,dEy,dEz]*exp(-i*time*real(x))
        #                              / norm(real([dEx,dEy,dEz])))
        dE_vec = np.array([dEx, dEy, dEz])
        dB_vec = np.array([dBx, dBy, dBz])

        norm_E = np.linalg.norm(np.real(dE_vec))
        norm_B = np.linalg.norm(np.real(dB_vec))
        if norm_E < 1e-30:
            norm_E = 1.0
        if norm_B < 1e-30:
            norm_B = 1.0

        E_direc = vxrange[1] * np.real(
            dE_vec * np.exp(-1j * time * np.real(x))) / norm_E
        B_direc = vxrange[1] * np.real(
            dB_vec * np.exp(-1j * time * np.real(x))) / norm_B

        _vtkwrite_EB_polydata(eb_fname, E_direc, B_direc)
        print(f"    VTK saved: {eb_fname}")

    print(f"  ========== VDF Done ==========\n")

