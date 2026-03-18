"""
pkues_plot_growth_rate.py
Converted from pkues_plot_growth_rate.m
20-05-20 added by Xingyu Zhu and Jiansen He
Plot growth/damping rate according to Eq.(3) of He et al. (2019)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator


def symlog_axis(ax, axis='y', linthresh=1e-10):
    """Apply symmetric log scale to an axis (mimics MATLAB symlog)."""
    if axis == 'y':
        ax.set_yscale('symlog', linthresh=linthresh)
    elif axis == 'x':
        ax.set_xscale('symlog', linthresh=linthresh)


def plot_growth_rate(S, pas, pa, JE, jpl, pltc, strpa, npa, savepath, figstr):
    """
    Plot growth/damping rate for each species component.

    Parameters
    ----------
    S : int
        Number of species.
    pas : array
        Parameter array for x-axis.
    pa : array
        Parameter range.
    JE : array
        Dissipation rate array, shape (npa, 3, S, npl).
    jpl : int
        Index for selected wave.
    pltc : array
        Plot colors.
    strpa : str
        Label string for x-axis parameter.
    npa : int
        Number of parameter a points.
    savepath : str
        Path to save figures.
    figstr : str
        Figure filename string.
    """
    color = pltc[jpl]

    if S == 3:  # two ion components (core + beam)
        fig, axes = plt.subplots(3, 4, figsize=(18, 14))
        fig.suptitle('')

        # Row 1: core ions
        ax = axes[0, 0]
        ax.plot(pas, JE[:, 0, 0, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{x,\mathrm{core}}$ [s$^{-1}$]')
        ax.set_title(r'$-J \cdot E / (dB^2 + dE^2) / 2$')

        ax = axes[0, 1]
        ax.plot(pas, JE[:, 1, 0, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{y,\mathrm{core}}$')

        ax = axes[0, 2]
        ax.plot(pas, JE[:, 2, 0, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{z,\mathrm{core}}$')

        ax = axes[0, 3]
        core_trace = JE[:, 0, 0, jpl] + JE[:, 1, 0, jpl] + JE[:, 2, 0, jpl]
        beam_trace = JE[:, 0, 1, jpl] + JE[:, 1, 1, jpl] + JE[:, 2, 1, jpl]
        ax.plot(pas, core_trace / beam_trace, '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{\mathrm{trace,core}} / \gamma_{\mathrm{trace,beam}}$')

        # Row 2: beam ions
        ax = axes[1, 0]
        ax.plot(pas, JE[:, 0, 1, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{x,\mathrm{beam}}$')

        ax = axes[1, 1]
        ax.plot(pas, JE[:, 1, 1, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{y,\mathrm{beam}}$')

        ax = axes[1, 2]
        ax.plot(pas, JE[:, 2, 1, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{z,\mathrm{beam}}$')

        ax = axes[1, 3]
        e_trace = JE[:, 0, 2, jpl] + JE[:, 1, 2, jpl] + JE[:, 2, 2, jpl]
        ax.plot(pas, beam_trace / e_trace, '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{\mathrm{trace,beam}} / \gamma_{\mathrm{trace,e}}$')

        # Row 3: electrons
        ax = axes[2, 0]
        ax.plot(pas, JE[:, 0, 2, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{x,\mathrm{electron}}$')

        ax = axes[2, 1]
        ax.plot(pas, JE[:, 1, 2, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{y,\mathrm{electron}}$')

        ax = axes[2, 2]
        ax.plot(pas, JE[:, 2, 2, jpl], '-', color=color, linewidth=2)
        symlog_axis(ax, 'y')
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{z,\mathrm{electron}}$')

        ax = axes[2, 3]
        ax.plot(pas, core_trace / e_trace, '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{\mathrm{trace,core}} / \gamma_{\mathrm{trace,e}}$')

    elif S == 2:  # one ion component
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))

        # Row 1: protons
        ax = axes[0, 0]
        ax.plot(pas, JE[:, 0, 0, jpl], '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{x,\mathrm{proton}}$ [s$^{-1}$]')
        ax.set_title(r'$-J \cdot E / (dB^2 + dE^2) / 2$')

        ax = axes[0, 1]
        ax.plot(pas, JE[:, 1, 0, jpl], '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{y,\mathrm{proton}}$')

        ax = axes[0, 2]
        ax.plot(pas, JE[:, 2, 0, jpl], '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{z,\mathrm{proton}}$')

        ax = axes[0, 3]
        i_trace = JE[:, 0, 0, jpl] + JE[:, 1, 0, jpl] + JE[:, 2, 0, jpl]
        e_trace = JE[:, 0, 1, jpl] + JE[:, 1, 1, jpl] + JE[:, 2, 1, jpl]
        ax.plot(pas, i_trace / e_trace, '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{\mathrm{trace,i}} / \gamma_{\mathrm{trace,e}}$')

        # Row 2: electrons
        ax = axes[1, 0]
        ax.plot(pas, JE[:, 0, 1, jpl], '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{x,\mathrm{electron}}$')

        ax = axes[1, 1]
        ax.plot(pas, JE[:, 1, 1, jpl], '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{y,\mathrm{electron}}$')

        ax = axes[1, 2]
        ax.plot(pas, JE[:, 2, 1, jpl], '-', color=color, linewidth=2)
        ax.set_xlim([np.min(pa), np.max(pa)])
        ax.set_xlabel(strpa)
        ax.set_ylabel(r'$\gamma_{z,\mathrm{electron}}$')

        axes[1, 3].axis('off')

    plt.tight_layout()
    fig.savefig(f"{savepath}fig_pdrk_{figstr}_growingrate.png", dpi=150)
    plt.close(fig)
    return fig

