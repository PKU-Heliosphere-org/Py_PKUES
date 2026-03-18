"""
pkues_plot_comp_velocity.py
Converted from pkues_plot_comp_velocity.m
20-05-14 coded by Xingyu Zhu and Jiansen He
Plot velocity polarizations: Comp1: Vy/iVx; Comp1/Comp2: Vx1/Vx2
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_comp_velocity(S, pas, pa, dVnorm, jpl, pltc, strpa, npa, savepath, figstr):
    """
    Plot velocity component polarizations for each species.

    Parameters
    ----------
    S : int
        Number of species.
    pas : array
        Parameter array for x-axis.
    pa : array
        Parameter range.
    dVnorm : array
        Normalized velocity perturbation, shape (npa, 3, S, npl).
    jpl : int
        Index for selected wave.
    pltc : array
        Plot colors.
    strpa : str
        Label string for x-axis parameter.
    npa : int
        Number of scan points.
    savepath : str
        Path to save figures.
    figstr : str
        Figure filename string.
    """
    color = pltc[jpl]

    if S == 3:  # two ion components (core + beam)
        fig, axes = plt.subplots(3, 6, figsize=(22, 14))
        fig.subplots_adjust(hspace=0.4, wspace=0.5)

        # --- Row 1: Core ions ---
        ax = axes[0, 0]
        ax.plot(pas, np.real(dVnorm[:, 1, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 1, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{iy,\mathrm{core}} / (iV_{ix,\mathrm{core}})$')

        ax = axes[0, 1]
        ax.plot(pas, np.real(dVnorm[:, 2, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 2, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{iz,\mathrm{core}} / (iV_{ix,\mathrm{core}})$')

        ax = axes[0, 2]
        ax.plot(pas, np.abs(dVnorm[:, 0, 0, jpl]) / np.abs(dVnorm[:, 0, 1, jpl]),
                '-', color=color, linewidth=2)
        ax.plot(pas, np.ones(npa), '--', color='b')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim([0, np.percentile(np.abs(dVnorm[:, 0, 0, jpl]) / np.abs(dVnorm[:, 0, 1, jpl]), 95) * 1.2])
        ax.set_xlabel(strpa); ax.set_ylabel(r'$|V_{ix,\mathrm{core}}| / |V_{ix,\mathrm{beam}}|$')

        for idx, (comp, label) in enumerate([(0, 'Vx_{core}'), (1, 'Vy_{core}'), (2, 'Vz_{core}')]):
            ax = axes[0, 3 + idx]
            ax.plot(pas, np.real(dVnorm[:, comp, 0, jpl]), '-', color=color, linewidth=2, label='Re')
            ax.plot(pas, np.imag(dVnorm[:, comp, 0, jpl]), '--', color=color, linewidth=2, label='Im')
            ax.set_xlim([np.min(pa), np.max(pa)])
            ax.set_ylim(np.percentile(np.real(dVnorm[:, comp, 0, jpl]), [5, 95]))
            ax.set_xlabel(strpa); ax.set_ylabel(f'${label}$')
            ax.legend(frameon=False, fontsize=8)

        # --- Row 2: Beam ions ---
        ax = axes[1, 0]
        ax.plot(pas, np.real(dVnorm[:, 1, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 1, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{iy,\mathrm{beam}} / (iV_{ix,\mathrm{beam}})$')

        ax = axes[1, 1]
        ax.plot(pas, np.real(dVnorm[:, 2, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 2, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{iz,\mathrm{beam}} / (iV_{ix,\mathrm{beam}})$')

        ax = axes[1, 2]
        ax.plot(pas, np.abs(dVnorm[:, 0, 1, jpl]) / np.abs(dVnorm[:, 0, 2, jpl]),
                '-', color=color, linewidth=2)
        ax.plot(pas, np.ones(npa), '--', color='b')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim([0, np.percentile(np.abs(dVnorm[:, 0, 1, jpl]) / np.abs(dVnorm[:, 0, 2, jpl]), 95) * 1.2])
        ax.set_xlabel(strpa); ax.set_ylabel(r'$|V_{ix,\mathrm{beam}}| / |V_{ex}|$')

        for idx, (comp, label) in enumerate([(0, 'Vx_{beam}'), (1, 'Vy_{beam}'), (2, 'Vz_{beam}')]):
            ax = axes[1, 3 + idx]
            ax.plot(pas, np.real(dVnorm[:, comp, 1, jpl]), '-', color=color, linewidth=2, label='Re')
            ax.plot(pas, np.imag(dVnorm[:, comp, 1, jpl]), '--', color=color, linewidth=2, label='Im')
            ax.set_xlim([np.min(pa), np.max(pa)])
            ax.set_ylim(np.percentile(np.real(dVnorm[:, comp, 1, jpl]), [5, 95]))
            ax.set_xlabel(strpa); ax.set_ylabel(f'${label}$')
            ax.legend(frameon=False, fontsize=8)

        # --- Row 3: Electrons ---
        ax = axes[2, 0]
        ax.plot(pas, np.real(dVnorm[:, 1, 2, jpl] / (1j * dVnorm[:, 0, 2, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 1, 2, jpl] / (1j * dVnorm[:, 0, 2, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{ey} / (iV_{ex})$')

        ax = axes[2, 1]
        ax.plot(pas, np.real(dVnorm[:, 2, 2, jpl] / (1j * dVnorm[:, 0, 2, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 2, 2, jpl] / (1j * dVnorm[:, 0, 2, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{ez} / (iV_{ex})$')

        ax = axes[2, 2]
        ax.plot(pas, np.abs(dVnorm[:, 0, 2, jpl]) / np.abs(dVnorm[:, 0, 0, jpl]),
                '-', color=color, linewidth=2)
        ax.plot(pas, np.ones(npa), '--', color='b')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.abs(dVnorm[:, 0, 2, jpl]) / np.abs(dVnorm[:, 0, 0, jpl]), [5, 95]) * 1.2)
        ax.set_xlabel(strpa); ax.set_ylabel(r'$|V_{ex}| / |V_{ix,\mathrm{core}}|$')

        for idx, (comp, label) in enumerate([(0, 'Vx_e'), (1, 'Vy_e'), (2, 'Vz_e')]):
            ax = axes[2, 3 + idx]
            ax.plot(pas, np.real(dVnorm[:, comp, 2, jpl]), '-', color=color, linewidth=2, label='Re')
            ax.plot(pas, np.imag(dVnorm[:, comp, 2, jpl]), '--', color=color, linewidth=2, label='Im')
            ax.set_xlim([np.min(pa), np.max(pa)])
            ax.set_ylim(np.percentile(np.real(dVnorm[:, comp, 2, jpl]), [5, 95]))
            ax.set_xlabel(strpa); ax.set_ylabel(f'${label}$')
            ax.legend(frameon=False, fontsize=8)

    elif S == 2:  # one ion component
        fig, axes = plt.subplots(2, 6, figsize=(22, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.5)

        # --- Row 1: Protons ---
        ax = axes[0, 0]
        ax.plot(pas, np.real(dVnorm[:, 1, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 1, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{i,y} / (iV_{i,x})$')

        ax = axes[0, 1]
        ax.plot(pas, np.real(dVnorm[:, 2, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 2, 0, jpl] / (1j * dVnorm[:, 0, 0, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{i,z} / (iV_{i,x})$')

        ax = axes[0, 2]
        ax.plot(pas, np.abs(dVnorm[:, 0, 0, jpl]) / np.abs(dVnorm[:, 0, 1, jpl]),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim([0, np.percentile(np.abs(dVnorm[:, 0, 0, jpl]) / np.abs(dVnorm[:, 0, 1, jpl]), 95) * 1.2])
        ax.set_xlabel(strpa); ax.set_ylabel(r'$|V_{i,x}| / |V_{e,x}|$')

        for idx, (comp, label) in enumerate([(0, 'V_{i,x}'), (1, 'V_{i,y}'), (2, 'V_{i,z}')]):
            ax = axes[0, 3 + idx]
            ax.plot(pas, np.real(dVnorm[:, comp, 0, jpl]), '-', color=color, linewidth=2, label='Re')
            ax.plot(pas, np.imag(dVnorm[:, comp, 0, jpl]), '--', color=color, linewidth=2, label='Im')
            ax.set_xlim([np.min(pa), np.max(pa)])
            ylim_min = min(np.percentile(np.real(dVnorm[:, comp, 0, jpl]), 5), np.percentile(np.imag(dVnorm[:, comp, 0, jpl]), 5))
            ylim_max = max(np.percentile(np.real(dVnorm[:, comp, 0, jpl]), 95), np.percentile(np.imag(dVnorm[:, comp, 0, jpl]), 95))
            ax.set_ylim([ylim_min, ylim_max])
            ax.set_xlabel(strpa); ax.set_ylabel(f'${label}$')
            ax.legend(frameon=False, fontsize=8)

        # --- Row 2: Electrons ---
        ax = axes[1, 0]
        ax.plot(pas, np.real(dVnorm[:, 1, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 1, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{e,y} / (iV_{e,x})$')

        ax = axes[1, 1]
        ax.plot(pas, np.real(dVnorm[:, 2, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim(np.percentile(np.real(dVnorm[:, 2, 1, jpl] / (1j * dVnorm[:, 0, 1, jpl])), [5, 95]))
        ax.set_xlabel(strpa); ax.set_ylabel(r'$V_{e,z} / (iV_{e,x})$')

        ax = axes[1, 2]
        ax.plot(pas, np.abs(dVnorm[:, 0, 0, jpl]) / np.abs(dVnorm[:, 0, 1, jpl]),
                '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_ylim([0, np.percentile(np.abs(dVnorm[:, 0, 0, jpl]) / np.abs(dVnorm[:, 0, 1, jpl]), 95) * 1.2])
        ax.set_xlabel(strpa); ax.set_ylabel(r'$|V_{i,y}| / |V_{e,y}|$')

        for idx, (comp, label) in enumerate([(0, 'V_{e,x}'), (1, 'V_{e,y}'), (2, 'V_{e,z}')]):
            ax = axes[1, 3 + idx]
            ax.plot(pas, np.real(dVnorm[:, comp, 1, jpl]), '-', color=color, linewidth=2, label='Re')
            ax.plot(pas, np.imag(dVnorm[:, comp, 1, jpl]), '--', color=color, linewidth=2, label='Im')
            ax.set_xlim([np.min(pa), np.max(pa)])
            ylim_min = min(np.percentile(np.real(dVnorm[:, comp, 1, jpl]), 5), np.percentile(np.imag(dVnorm[:, comp, 1, jpl]), 5))
            ylim_max = max(np.percentile(np.real(dVnorm[:, comp, 1, jpl]), 95), np.percentile(np.imag(dVnorm[:, comp, 1, jpl]), 95))
            ax.set_ylim([ylim_min, ylim_max])
            ax.set_xlabel(strpa); ax.set_ylabel(f'${label}$')
            ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{savepath}fig_pdrk_{figstr}_velocity.png", dpi=150)
    plt.close(fig)
    return fig

