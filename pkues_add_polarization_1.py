"""
pkues_add_polarization_1.py
Converted from pkues_add_polarization_1.m
20-05-20 Coded by Xingyu Zhu and Jiansen He
Plot pseudo growing rate according to Eq.(3) of He et al. (2019)
Phase differences between E/B field and electron velocity components.
"""

import numpy as np
import matplotlib.pyplot as plt


def _plot_phase_grid(fig, axes, pas, pa, npa, phi_data, labels, pltc, jpl, strpa):
    """Helper to plot a 3x3 grid of phase difference panels."""
    color = pltc[jpl]
    ref_lines = [0, 90, 180, 270, 360]

    for idx, (ax, phi, label) in enumerate(zip(axes.flat, phi_data, labels)):
        ax.plot(pas, phi, '-', color=color, linewidth=2)
        for ref in ref_lines:
            ax.plot(pas, np.full(npa, ref), '--', color='b', linewidth=0.5)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(label)
        ax.set_box_aspect(None)


def add_polarization_electron(S, pas, pa, npa, Pola_norm, dV, jpl, pltc,
                               strpa, savepath, figstr):
    """
    Plot phase differences between B/E fields and electron velocity.

    Parameters
    ----------
    S : int
        Number of species.
    pas : array
        Scan parameter array.
    pa : array
        Parameter range.
    npa : int
        Number of scan points.
    Pola_norm : array
        Normalized polarization, shape (npa, npb, npl, 6+).
        Indices 0-2: Ex,Ey,Ez; 3-5: Bx,By,Bz.
    dV : array
        Velocity perturbation, shape (npa, 3, S, npl).
    jpl : int
        Index for selected wave.
    pltc : array
        Plot colors.
    strpa : str
        X-axis label.
    savepath : str
        Path to save figures.
    figstr : str
        Figure filename string.
    """
    # Determine electron species index based on S
    if S == 3:
        s_elec = 2   # 3rd species index (0-based), MATLAB dV(:,x,3,1) -> dV[:,x,2,0]
    elif S == 2:
        s_elec = 1   # 2nd species index (0-based), MATLAB dV(:,x,2,1) -> dV[:,x,1,0]
    else:
        return

    # Compute phase differences (in degrees, wrapped to [0, 360])
    # def phase_diff(a, b, eps=1e-12):
    #     phi = np.full(a.shape, np.nan, dtype=float)
    #     mask = (np.abs(a) > eps) & (np.abs(b) > eps)
    #     phi[mask] = np.mod((np.angle(a[mask]) - np.angle(b[mask])) * 180 / np.pi + 360, 360)
    #     return phi
    def phase_diff(a, b):
        return np.mod((np.angle(a) - np.angle(b)) * 180 / np.pi + 360, 360)

    phi_Bx_Vex = phase_diff(Pola_norm[:, 0, jpl, 3], dV[:, 0, s_elec, jpl])
    phi_By_Vey = phase_diff(Pola_norm[:, 0, jpl, 4], dV[:, 1, s_elec, jpl])
    phi_Bz_Vez = phase_diff(Pola_norm[:, 0, jpl, 5], dV[:, 2, s_elec, jpl])
    phi_Ex_Vex = phase_diff(Pola_norm[:, 0, jpl, 0], dV[:, 0, s_elec, jpl])
    phi_Ey_Vey = phase_diff(Pola_norm[:, 0, jpl, 1], dV[:, 1, s_elec, jpl])
    phi_Ez_Vez = phase_diff(Pola_norm[:, 0, jpl, 2], dV[:, 2, s_elec, jpl])
    phi_Bx_Ex = phase_diff(Pola_norm[:, 0, jpl, 3], Pola_norm[:, 0, jpl, 0])
    phi_By_Ey = phase_diff(Pola_norm[:, 0, jpl, 4], Pola_norm[:, 0, jpl, 1])
    phi_Bz_Ez = phase_diff(Pola_norm[:, 0, jpl, 5], Pola_norm[:, 0, jpl, 2])

    phi_data = [phi_Bx_Vex, phi_By_Vey, phi_Bz_Vez,
                phi_Ex_Vex, phi_Ey_Vey, phi_Ez_Vez,
                phi_Bx_Ex, phi_By_Ey, phi_Bz_Ez]

    labels = [r'$\phi(Bx-Vex)\;(°)$', r'$\phi(By-Vey)\;(°)$', r'$\phi(Bz-Vez)\;(°)$',
              r'$\phi(Ex-Vex)\;(°)$', r'$\phi(Ey-Vey)\;(°)$', r'$\phi(Ez-Vez)\;(°)$',
              r'$\phi(Bx-Ex)\;(°)$',  r'$\phi(By-Ey)\;(°)$',  r'$\phi(Bz-Ez)\;(°)$']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    _plot_phase_grid(fig, axes, pas, pa, npa, phi_data, labels, pltc, jpl, strpa)

    plt.tight_layout()
    fig.savefig(f"{savepath}fig_pdrk_{figstr}_BEVepola.png", dpi=150)
    plt.close(fig)
    return fig

