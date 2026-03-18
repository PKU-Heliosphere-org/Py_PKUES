"""
pkues_output_simple.py
Converted from pkues_output_simple.m
18-10-06 07:05 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Simplified output including sigma_r, sigma_c calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def pkues_output_simple(wws, wws2, Pola, Pola_norm, Pola_SI, dV, dVnorm, JE,
                        Zp_norm, Zm_norm, ns0,
                        npa, npb, npl, ipa, ipb, iloga,
                        pa, pas, ppa, ppb,
                        strpa, strpb, S, N, J, B0, theta,
                        pltc, savepath, figstr, jpb=0):
    """
    Simplified output for PDRK results.
    Includes sigma_r, sigma_c calculations and associated plots.
    """
    os.makedirs(savepath, exist_ok=True)

    # Save polarization data to ASCII
    matrix = np.column_stack([
        pas,
        np.real(wws[:, 0, 0]), np.imag(wws[:, 0, 0]),
        np.real(Pola_norm[:, 0, 0, 0]), np.imag(Pola_norm[:, 0, 0, 0]),
        np.real(Pola_norm[:, 0, 0, 1]), np.imag(Pola_norm[:, 0, 0, 1]),
        np.real(Pola_norm[:, 0, 0, 2]), np.imag(Pola_norm[:, 0, 0, 2]),
        np.real(Pola_norm[:, 0, 0, 3]), np.imag(Pola_norm[:, 0, 0, 3]),
        np.real(Pola_norm[:, 0, 0, 4]), np.imag(Pola_norm[:, 0, 0, 4]),
        np.real(Pola_norm[:, 0, 0, 5]), np.imag(Pola_norm[:, 0, 0, 5]),
        np.real(dVnorm[:, 0, 1, 0]), np.imag(dVnorm[:, 0, 1, 0]),  # species 2 (beam/electron depending on S)
        np.real(dVnorm[:, 1, 1, 0]), np.imag(dVnorm[:, 1, 1, 0]),
        np.real(dVnorm[:, 2, 1, 0]), np.imag(dVnorm[:, 2, 1, 0]),
        np.real(dVnorm[:, 0, 0, 0]), np.imag(dVnorm[:, 0, 0, 0]),  # species 1 (core)
        np.real(dVnorm[:, 1, 0, 0]), np.imag(dVnorm[:, 1, 0, 0]),
        np.real(dVnorm[:, 2, 0, 0]), np.imag(dVnorm[:, 2, 0, 0]),
        JE[:, 0, 0, 0], JE[:, 1, 0, 0], JE[:, 2, 0, 0],
        JE[:, 0, 1, 0], JE[:, 1, 1, 0], JE[:, 2, 1, 0],
    ])
    np.savetxt('polarization.dat', matrix)

    for jpl in range(npl):
        if ipa == ipb:  # 1D plot
            _pas = 10.0**pas if iloga == 1 else pas.copy()
            _pa = _pas.copy()
            color = pltc[jpl]

            # --- Main figure: omega, helicity, compressibility, sigma_r/sigma_c ---
            fig, axes = plt.subplots(3, 5, figsize=(22, 14))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            # (1,1) omega_r
            ax = axes[0, 0]
            ax.plot(_pas, np.real(wws2[:, 0, jpl]), '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$\omega_r / \omega_{c1}$')

            # (1,2) omega_i
            ax = axes[0, 1]
            ax.plot(_pas, np.imag(wws2[:, 0, jpl]), '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$\omega_i / \omega_{c1}$')

            # Magnetic helicity and compressibility
            tempBx = Pola[:, 0, jpl, 3]
            tempBy = Pola[:, 0, jpl, 4]
            tempBz = Pola[:, 0, jpl, 5]

            if ipa + ipb - 2 == 0:  # scan k, fixed theta
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

            ax = axes[0, 2]
            ax.plot(_pas, np.real(Mcompress), '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_ylim([0, 1])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$dB_{||}^2 / |dB|^2$')

            ax = axes[0, 3]
            ax.plot(_pas, np.real(Mhelicity), '-', color=color, linewidth=2)
            ax.set_xlim([np.min(_pa), np.max(_pa)])
            ax.set_ylim([-1.01, 1.01])
            ax.set_xlabel(strpa); ax.set_ylabel(r'$\sigma_m$')

            axes[0, 4].axis('off')

            # --- Compute sigma_c and sigma_r ---
            # sigma_c: cross helicity based on Elsasser variables
            sigmac = np.zeros((npa, 6, S, npl))
            for s_idx in range(S):
                for comp in range(3):
                    sigmac[:, comp, s_idx, jpl] = (
                        (np.abs(Zp_norm[:, comp, s_idx, jpl])**2 -
                         np.abs(Zm_norm[:, comp, s_idx, jpl])**2) /
                        (np.abs(Zp_norm[:, comp, s_idx, jpl])**2 +
                         np.abs(Zm_norm[:, comp, s_idx, jpl])**2))

                # perp (x+y)
                sigmac[:, 3, s_idx, jpl] = (
                    (np.abs(Zp_norm[:, 0, s_idx, jpl])**2 +
                     np.abs(Zp_norm[:, 1, s_idx, jpl])**2 -
                     np.abs(Zm_norm[:, 0, s_idx, jpl])**2 -
                     np.abs(Zm_norm[:, 1, s_idx, jpl])**2) /
                    (np.abs(Zp_norm[:, 0, s_idx, jpl])**2 +
                     np.abs(Zp_norm[:, 1, s_idx, jpl])**2 +
                     np.abs(Zm_norm[:, 0, s_idx, jpl])**2 +
                     np.abs(Zm_norm[:, 1, s_idx, jpl])**2))

                # trace (x+y+z)
                sigmac[:, 4, s_idx, jpl] = (
                    (np.abs(Zp_norm[:, 0, s_idx, jpl])**2 +
                     np.abs(Zp_norm[:, 1, s_idx, jpl])**2 +
                     np.abs(Zp_norm[:, 2, s_idx, jpl])**2 -
                     np.abs(Zm_norm[:, 0, s_idx, jpl])**2 -
                     np.abs(Zm_norm[:, 1, s_idx, jpl])**2 -
                     np.abs(Zm_norm[:, 2, s_idx, jpl])**2) /
                    (np.abs(Zp_norm[:, 0, s_idx, jpl])**2 +
                     np.abs(Zp_norm[:, 1, s_idx, jpl])**2 +
                     np.abs(Zp_norm[:, 2, s_idx, jpl])**2 +
                     np.abs(Zm_norm[:, 0, s_idx, jpl])**2 +
                     np.abs(Zm_norm[:, 1, s_idx, jpl])**2 +
                     np.abs(Zm_norm[:, 2, s_idx, jpl])**2))

            # sigma_r: residual energy
            sigmar = np.zeros((npa, 6, S, npl))
            for ixx in range(npa):
                for s in range(S):
                    density_ratio = np.sqrt(ns0[0] / ns0[s])
                    for comp_idx, b_idx in enumerate([3, 4, 5]):
                        v_comp = np.abs(dVnorm[ixx, comp_idx, s, jpl])**2
                        b_comp = np.abs(Pola_norm[ixx, jpb, jpl, b_idx] * density_ratio)**2
                        sigmar[ixx, comp_idx, s, jpl] = (v_comp - b_comp) / (v_comp + b_comp)

                    # perp (x+y)
                    v_perp = (np.abs(dVnorm[ixx, 0, s, jpl])**2 +
                              np.abs(dVnorm[ixx, 1, s, jpl])**2)
                    b_perp = (np.abs(Pola_norm[ixx, jpb, jpl, 3] * density_ratio)**2 +
                              np.abs(Pola_norm[ixx, jpb, jpl, 4] * density_ratio)**2)
                    sigmar[ixx, 3, s, jpl] = (v_perp - b_perp) / (v_perp + b_perp)

                    # trace (x+y+z)
                    v_trace = (np.abs(dVnorm[ixx, 0, s, jpl])**2 +
                               np.abs(dVnorm[ixx, 1, s, jpl])**2 +
                               np.abs(dVnorm[ixx, 2, s, jpl])**2)
                    b_trace = (np.abs(Pola_norm[ixx, jpb, jpl, 3] * density_ratio)**2 +
                               np.abs(Pola_norm[ixx, jpb, jpl, 4] * density_ratio)**2 +
                               np.abs(Pola_norm[ixx, jpb, jpl, 5] * density_ratio)**2)
                    sigmar[ixx, 4, s, jpl] = (v_trace - b_trace) / (v_trace + b_trace)

            # --- Plot sigma_r and sigma_c ---
            comp_labels = ['X', 'Y', 'Z', 'perp', 'trace']

            if S == 3:
                fig4, axes4 = plt.subplots(3, 5, figsize=(22, 14))
                fig4.subplots_adjust(hspace=0.5, wspace=0.5)
                species_names = ['proton core', 'proton beam', 'electron']
                for s_row in range(3):
                    for c_col in range(5):
                        ax = axes4[s_row, c_col]
                        ax.plot(_pas, sigmar[:, c_col, s_row, 0], 'r', linewidth=2, label=r'$\sigma_r$')
                        ax.plot(_pas, sigmac[:, c_col, s_row, 0], 'b', linewidth=2, label=r'$\sigma_c$')
                        ax.set_xlim([np.min(_pa), np.max(_pa)])
                        ax.set_ylim([-1, 1])
                        ax.set_xlabel(strpa)
                        ax.set_ylabel(comp_labels[c_col])
                        if c_col == 0:
                            ax.legend(frameon=False, fontsize=8)
                            ax.set_title(species_names[s_row])

            elif S == 2:
                # Plot sigma_r/sigma_c in rows 2 and 3 of the main figure
                species_names = ['proton', 'electron']
                for s_row, s_idx in enumerate([0, 1]):
                    for c_col in range(5):
                        ax = axes[1 + s_row, c_col]
                        ax.plot(_pas, sigmar[:, c_col, s_idx, 0], 'r', linewidth=2, label=r'$\sigma_r$')
                        ax.plot(_pas, sigmac[:, c_col, s_idx, 0], 'b', linewidth=2, label=r'$\sigma_c$')
                        ax.set_xlim([np.min(_pa), np.max(_pa)])
                        ax.set_ylim([-1, 1])
                        ax.set_xlabel(strpa)
                        ax.set_ylabel(comp_labels[c_col])
                        if c_col == 0:
                            ax.legend(frameon=False, fontsize=8)
                            ax.set_title(species_names[s_row])

            plt.tight_layout()
            fig.savefig(f"{savepath}fig_pdrk_{figstr}_pola.png", dpi=150)
            plt.close(fig)

            if S == 3:
                fig4.savefig(f"{savepath}fig_pdrk_{figstr}_sigma.png", dpi=150)
                plt.close(fig4)

    # Save workspace
    filename = f"out_pdrk_S={S}_J={J}_N={N}_B0={B0}.npz"
    np.savez(os.path.join(savepath, filename),
             wws=wws, wws2=wws2, Pola=Pola, Pola_norm=Pola_norm,
             Pola_SI=Pola_SI, dV=dV, dVnorm=dVnorm, JE=JE,
             sigmar=sigmar if 'sigmar' in dir() else None,
             sigmac=sigmac if 'sigmac' in dir() else None)

