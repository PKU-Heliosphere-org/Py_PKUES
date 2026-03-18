"""
pkues_output.py
Converted from pkues_output.m
18-10-06 07:05 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Modified by Xingyu Zhu, Die Duan and Jiansen He to plot polarizations of
different components, magnetic helicity and magnetic compressibility as
a function of k.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from pkues_plot_comp_velocity import plot_comp_velocity
from pkues_plot_growth_rate import plot_growth_rate
from pkues_add_polarization_1 import add_polarization_electron
from pkues_add_polarization_2 import add_polarization_ion


def pkues_output(wws, wws2, Pola, Pola_norm, Pola_SI, dV, dVnorm, JE,
                 Zp_norm, Zm_norm, ns0,
                 npa, npb, npl, nw0, ipa, ipb, iloga, ilogb,
                 pa, pas, ppa, ppb,
                 strpa, strpb, S, N, J, B0, theta,
                 pltc, savepath, figstr,
                 betasz, betasp, jpb=0):
    """
    Output and plot PDRK results including polarization, helicity,
    compressibility, velocity, growth rate, and phase differences.

    Parameters
    ----------
    (See individual module docstrings for parameter descriptions)
    """

    os.makedirs(savepath, exist_ok=True)

    for jpl in range(npl):
        if ipa == ipb:  # 1D polarizations
            
            matrix = np.column_stack([
                pas, np.real(wws[:, 0, jpl]), np.imag(wws[:, 0, jpl]),
                np.real(Pola_norm[:, 0, jpl, 0]), np.imag(Pola_norm[:, 0, jpl, 0]),
                np.real(Pola_norm[:, 0, jpl, 1]), np.imag(Pola_norm[:, 0, jpl, 1]),
                np.real(Pola_norm[:, 0, jpl, 2]), np.imag(Pola_norm[:, 0, jpl, 2]),
                np.real(Pola_norm[:, 0, jpl, 3]), np.imag(Pola_norm[:, 0, jpl, 3]),
                np.real(Pola_norm[:, 0, jpl, 4]), np.imag(Pola_norm[:, 0, jpl, 4]),
                np.real(Pola_norm[:, 0, jpl, 5]), np.imag(Pola_norm[:, 0, jpl, 5]),
                np.real(dVnorm[:, 0, 1, jpl]), np.imag(dVnorm[:, 0, 1, jpl]),
                np.real(dVnorm[:, 1, 1, jpl]), np.imag(dVnorm[:, 1, 1, jpl]),
                np.real(dVnorm[:, 2, 1, jpl]), np.imag(dVnorm[:, 2, 1, jpl]),
                np.real(dVnorm[:, 0, 0, jpl]), np.imag(dVnorm[:, 0, 0, jpl]),
                np.real(dVnorm[:, 1, 0, jpl]), np.imag(dVnorm[:, 1, 0, jpl]),
                np.real(dVnorm[:, 2, 0, jpl]), np.imag(dVnorm[:, 2, 0, jpl]),
                JE[:, 0, 0, jpl], JE[:, 1, 0, jpl], JE[:, 2, 0, jpl],
                JE[:, 0, 1, jpl], JE[:, 1, 1, jpl], JE[:, 2, 1, jpl]
                ])

            np.savetxt(os.path.join(savepath, f'polarization_mode_{jpl+1}.dat'), matrix)
            _pas = 10.0**pas if iloga == 1 else pas.copy()
            _pa = _pas.copy()

            # --- Figure 1: E, B polarizations (4x5 layout) ---
            fig1, axes1 = plt.subplots(4, 5, figsize=(20, 16))
            fig1.subplots_adjust(hspace=0.5, wspace=0.5)
            color = pltc[jpl]

            # (1,1) omega_r
            ax = axes1[0, 0]
            ax.plot(_pas, np.real(wws2[:, 0, jpl]), '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$\omega_r / \omega_{c1}$')

            # (1,2) omega_i
            ax = axes1[0, 1]
            ax.plot(_pas, np.imag(wws2[:, 0, jpl]), '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$\omega_i / \omega_{c1}$')

            # (1,3) Energy E / Energy B
            ax = axes1[0, 2]
            ax.plot(_pas, Pola_SI[:, 0, jpl, 6] / Pola_SI[:, 0, jpl, 7],
                    '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel('Energy E / Energy B')

            # (2,1) Ey/(iEx)
            ax = axes1[1, 0]
            val = np.real(Pola_norm[:, 0, jpl, 1] / (1j * Pola_norm[:, 0, jpl, 0]))
            ax.plot(_pas, val, '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$E_y / (iE_x)$')

            # (2,2) |Ez|/|Ex|
            ax = axes1[1, 1]
            ax.plot(_pas, np.abs(Pola_norm[:, 0, jpl, 2]) / np.abs(Pola_norm[:, 0, jpl, 0]),
                    '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$|E_z| / |E_x|$')

            # (2,3)-(2,5) Ex, Ey, Ez components Re/Im
            for ci, (col, lbl) in enumerate([(0, r'$E_x / B_0 V_A$'),
                                              (1, r'$E_y / B_0 V_A$'),
                                              (2, r'$E_z / B_0 V_A$')]):
                ax = axes1[1, 2 + ci]
                ax.plot(_pas, np.real(Pola_norm[:, 0, jpl, col]), '-', color=color, linewidth=2, label='Re')
                ax.plot(_pas, np.imag(Pola_norm[:, 0, jpl, col]), '--', color=color, linewidth=2, label='Im')
                ax.set_xlim([np.min(_pa), np.max(_pa)])
                ax.set_xlabel(strpa); ax.set_ylabel(lbl)
                if ci == 0:
                    ax.legend(frameon=False, fontsize=7)

            # (3,1) By/(iBx)
            ax = axes1[2, 0]
            val = np.real(Pola_norm[:, 0, jpl, 4] / (1j * Pola_norm[:, 0, jpl, 3]))
            ax.plot(_pas, val, '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$B_y / (iB_x)$')

            # (3,2) |Bz|/|Bx|
            ax = axes1[2, 1]
            ax.plot(_pas, np.abs(Pola_norm[:, 0, jpl, 5]) / np.abs(Pola_norm[:, 0, jpl, 3]),
                    '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$|B_z| / |B_x|$')

            # (3,3)-(3,5) Bx, By, Bz components Re/Im
            for ci, (col, lbl) in enumerate([(3, r'$B_x / B_0$'),
                                              (4, r'$B_y / B_0$'),
                                              (5, r'$B_z / B_0$')]):
                ax = axes1[2, 2 + ci]
                ax.plot(_pas, np.real(Pola_norm[:, 0, jpl, col]), '-', color=color, linewidth=2, label='Re')
                ax.plot(_pas, np.imag(Pola_norm[:, 0, jpl, col]), '--', color=color, linewidth=2, label='Im')
                ax.set_xlim([np.min(_pa), np.max(_pa)])
                ax.set_xlabel(strpa); ax.set_ylabel(lbl)
                if ci == 0:
                    ax.legend(frameon=False, fontsize=7)

            # (4,1) Jy/(iJx)
            ax = axes1[3, 0]
            val = np.real(Pola_norm[:, 0, jpl, 9] / (1j * Pola_norm[:, 0, jpl, 8]))
            ax.plot(_pas, val, '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$J_y / (iJ_x)$')

            # (4,2) |Jz|/|Jx|
            ax = axes1[3, 1]
            ax.plot(_pas, np.abs(Pola_norm[:, 0, jpl, 10]) / np.abs(Pola_norm[:, 0, jpl, 8]),
                    '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$|J_z| / |J_x|$')

            # (4,3)-(4,5) Jx, Jy, Jz components Re/Im
            for ci, (col, lbl) in enumerate([(8, r'$J_x / (q_i n_i V_A)$'),
                                              (9, r'$J_y / (q_i n_i V_A)$'),
                                              (10, r'$J_z / (q_i n_i V_A)$')]):
                ax = axes1[3, 2 + ci]
                ax.plot(_pas, np.real(Pola_norm[:, 0, jpl, col]), '-', color=color, linewidth=2, label='Re')
                ax.plot(_pas, np.imag(Pola_norm[:, 0, jpl, col]), '--', color=color, linewidth=2, label='Im')
                ax.set_xlim([np.min(_pa), np.max(_pa)])
                ax.set_xlabel(strpa); ax.set_ylabel(lbl)
                if ci == 0:
                    ax.legend(frameon=False, fontsize=7)

            plt.tight_layout()
            fig1.savefig(f"{savepath}fig_pdrk_{figstr}_pola.png", dpi=150)
            plt.close(fig1)

            # --- Figure 2: Magnetic helicity and compressibility ---
            tempBx = Pola[:, 0, jpl, 3]
            tempBy = Pola[:, 0, jpl, 4]
            tempBz = Pola[:, 0, jpl, 5]

            if ipa + ipb - 2 == 0:  # scan k, fixed theta (0-indexed: ipa=0,ipb=0 => sum=0)
                # Note: in original code ipa=1,ipb=1 (MATLAB 1-based), so condition is ipa+ipb-2==0
                # For Python we check the original condition
                kdirection = np.array([np.sin(theta * np.pi / 180), 0,
                                       np.cos(theta * np.pi / 180)])
                Bpara = (tempBx * kdirection[0] + tempBy * kdirection[1] +
                         tempBz * kdirection[2])
                Bperp1 = tempBy
                kperp2 = np.cross(kdirection, [0, 1, 0])
                kperp2 = kperp2 / np.linalg.norm(kperp2)
                Bperp2 = (tempBx * kperp2[0] + tempBy * kperp2[1] +
                          tempBz * kperp2[2])
            else:
                Bpara = tempBz
                Bperp1 = tempBy
                Bperp2 = tempBx

            Mcompress = (np.abs(tempBz)**2 /
                         (np.abs(Bpara)**2 + np.abs(Bperp1)**2 + np.abs(Bperp2)**2))
            Mhelicity = (2 / np.cos(theta * np.pi / 180) *
                         np.real(1j * tempBx * np.conj(tempBy)) /
                         (np.abs(tempBx)**2 + np.abs(tempBy)**2 + np.abs(tempBz)**2))

            fig2, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(12, 5))
            ax_c.plot(_pas, np.real(Mcompress), '-', color=color, linewidth=2)
            ax_c.set_xlim([np.min(_pa), np.max(_pa)])
            ax_c.set_xlabel(strpa)
            ax_c.set_ylabel(r'$dB_{||}^2 / |dB|^2$')

            ax_h.plot(_pas, np.real(Mhelicity), '-', color=color, linewidth=2)
            ax_h.set_xlim([np.min(_pa), np.max(_pa)])
            ax_h.set_xlabel(strpa)
            ax_h.set_ylabel(r'$\sigma_m$')

            plt.tight_layout()
            fig2.savefig(f"{savepath}fig_pdrk_{figstr}_helicity.png", dpi=150)
            plt.close(fig2)

            # --- Additional plots ---
            plot_comp_velocity(S, _pas, _pa, dVnorm, jpl, pltc, strpa, npa, savepath, figstr)
            plot_growth_rate(S, _pas, _pa, JE, jpl, pltc, strpa, npa, savepath, figstr)
            add_polarization_electron(S, _pas, _pa, npa, Pola_norm, dV, jpl, pltc,
                                      strpa, savepath, figstr)
            add_polarization_ion(S, _pas, _pa, npa, Pola_norm, dV, jpl, pltc,
                                  strpa, savepath, figstr)

        else:  # 2D polarizations
            wwjp = wws2[:, :, jpl]
            Polajp = Pola[:, :, jpl, :]

            fig_2d = plt.figure(figsize=(14, 10))
            ax1 = fig_2d.add_subplot(221, projection='3d')
            ax1.plot_surface(ppa, ppb, np.real(wwjp))
            ax1.set_xlabel(f'{strpa}, ilogx={iloga}')
            ax1.set_ylabel(f'{strpb}, ilogy={ilogb}')
            ax1.set_zlabel(r'$\omega_r / \omega_{c1}$')

            ax2 = fig_2d.add_subplot(222, projection='3d')
            ax2.plot_surface(ppa, ppb, np.imag(wwjp))
            ax2.set_xlabel(strpa); ax2.set_ylabel(strpb)
            ax2.set_zlabel(r'$\omega_i / \omega_{c1}$')

            EyoveriEx = Polajp[:, :, 1] / (1j * Polajp[:, :, 0])

            ax3 = fig_2d.add_subplot(223, projection='3d')
            ax3.plot_surface(ppa, ppb, np.real(EyoveriEx))
            ax3.set_xlabel(strpa); ax3.set_ylabel(strpb)
            ax3.set_zlabel(r'Re[$E_y / (iE_x)$]')

            ax4 = fig_2d.add_subplot(224, projection='3d')
            ax4.plot_surface(ppa, ppb, np.imag(EyoveriEx))
            ax4.set_xlabel(strpa); ax4.set_ylabel(strpb)
            ax4.set_zlabel(r'Im[$E_y / (iE_x)$]')

            plt.tight_layout()
            fig_2d.savefig(f"{savepath}fig_pdrk_{figstr}_pola.png", dpi=150)
            plt.close(fig_2d)

    # Save workspace
    filename = f"out_pdrk_S={S}_J={J}_N={N}_B0={B0}.npz"
    np.savez(os.path.join(savepath, filename),
             wws=wws, wws2=wws2, Pola=Pola, Pola_norm=Pola_norm,
             Pola_SI=Pola_SI, dV=dV, dVnorm=dVnorm, JE=JE)

