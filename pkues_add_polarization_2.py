"""
pkues_add_polarization_2.py
Converted from pkues_add_polarization_2.m
20-05-20 added by Xingyu Zhu and Jiansen He
Plot growing rate according to Eq.(3) of He et al. (2019)
Phase differences between E/B field and core ion velocity components.
"""

import numpy as np
import matplotlib.pyplot as plt


def add_polarization_ion(S, pas, pa, npa, Pola_norm, dV, jpl, pltc,
                          strpa, savepath, figstr):
    """
    Plot phase differences between B/E fields and core ion velocity.

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
    if S not in (2, 3):
        return

    # Ion species index = 0 (1st species, core ions) for both S==2 and S==3
    s_ion = 0

    # def phase_diff(a, b, eps=1e-12):
    #     phi = np.full(a.shape, np.nan, dtype=float)
    #     mask = (np.abs(a) > eps) & (np.abs(b) > eps)
    #     phi[mask] = np.mod((np.angle(a[mask]) - np.angle(b[mask])) * 180 / np.pi + 360, 360)
    #     return phi
    def phase_diff(a, b):
        return np.mod((np.angle(a) - np.angle(b)) * 180 / np.pi + 360, 360)

    phi_Bx_Vix = phase_diff(Pola_norm[:, 0, jpl, 3], dV[:, 0, s_ion, jpl])
    phi_By_Viy = phase_diff(Pola_norm[:, 0, jpl, 4], dV[:, 1, s_ion, jpl])
    phi_Bz_Viz = phase_diff(Pola_norm[:, 0, jpl, 5], dV[:, 2, s_ion, jpl])
    phi_Ex_Vix = phase_diff(Pola_norm[:, 0, jpl, 0], dV[:, 0, s_ion, jpl])
    phi_Ey_Viy = phase_diff(Pola_norm[:, 0, jpl, 1], dV[:, 1, s_ion, jpl])
    phi_Ez_Viz = phase_diff(Pola_norm[:, 0, jpl, 2], dV[:, 2, s_ion, jpl])
    phi_Bx_Ex = phase_diff(Pola_norm[:, 0, jpl, 3], Pola_norm[:, 0, jpl, 0])
    phi_By_Ey = phase_diff(Pola_norm[:, 0, jpl, 4], Pola_norm[:, 0, jpl, 1])
    phi_Bz_Ez = phase_diff(Pola_norm[:, 0, jpl, 5], Pola_norm[:, 0, jpl, 2])

    phi_data = [phi_Bx_Vix, phi_By_Viy, phi_Bz_Viz,
                phi_Ex_Vix, phi_Ey_Viy, phi_Ez_Viz,
                phi_Bx_Ex, phi_By_Ey, phi_Bz_Ez]

    labels = [r'$\phi(Bx-Vix)\;(°)$', r'$\phi(By-Viy)\;(°)$', r'$\phi(Bz-Viz)\;(°)$',
              r'$\phi(Ex-Vix)\;(°)$', r'$\phi(Ey-Viy)\;(°)$', r'$\phi(Ez-Viz)\;(°)$',
              r'$\phi(Bx-Ex)\;(°)$',  r'$\phi(By-Ey)\;(°)$',  r'$\phi(Bz-Ez)\;(°)$']

    ref_lines = [0, 90, 180, 270, 360]
    color = pltc[jpl]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for ax, phi, label in zip(axes.flat, phi_data, labels):
        ax.plot(pas, phi, '-', color=color, linewidth=2)
        for ref in ref_lines:
            ax.plot(pas, np.full(npa, ref), '--', color='b', linewidth=0.5)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(label)

    plt.tight_layout()
    fig.savefig(f"{savepath}fig_pdrk_{figstr}_BEVipola.png", dpi=150)
    plt.close(fig)
    return fig

