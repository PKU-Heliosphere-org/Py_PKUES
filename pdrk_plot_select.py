"""
pdrk_plot_select.py
Converted from pdrk_plot_select.m
18-10-05 16:42 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Try a strategy to select given dispersion surface. Steps:
1. Select an initial point in the solutions, e.g., (k, w_j)
2. Determine the corresponding w_j for other k, using interp or
   least square, or nearest neighbor
3. Store the (k, w_j) as one dispersion surface
4. Repeat for another surface
Better for smaller dk.

18:41 Test 1D nearest neighbor can work, but not well for omega past zero
19:33 Try interp1, works very well!
18-10-06 00:48 2D scan not perfect yet
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def pchip_extrap(x_known, y_known, x_query):
    """
    PCHIP interpolation with extrapolation, equivalent to MATLAB's
    interp1(x, y, xq, 'pchip', 'extrap').

    Parameters
    ----------
    x_known : array_like
        Known x values (must be sorted and unique).
    y_known : array_like
        Known y values (can be complex).
    x_query : float or array_like
        Query point(s).

    Returns
    -------
    y_query : complex
        Interpolated/extrapolated value(s).
    """
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=complex)

    # Sort by x
    sort_idx = np.argsort(x_known)
    x_sorted = x_known[sort_idx]
    y_sorted = y_known[sort_idx]

    # Remove duplicates (keep last)
    _, unique_idx = np.unique(x_sorted, return_index=True)
    x_sorted = x_sorted[unique_idx]
    y_sorted = y_sorted[unique_idx]

    if len(x_sorted) < 2:
        return y_sorted[0] if len(y_sorted) > 0 else 0.0

    # Interpolate real and imag parts separately
    interp_re = PchipInterpolator(x_sorted, np.real(y_sorted), extrapolate=True)
    interp_im = PchipInterpolator(x_sorted, np.imag(y_sorted), extrapolate=True)

    return interp_re(x_query) + 1j * interp_im(x_query)


def pdrk_plot_select(wwn, pa, pb, npa, npb, ipa, ipb, iloga, ilogb,
                     wcs1, strpa, strpb, betasz, betasp, alphas, Deltas,
                     vA, c2, S, N, J, iem, par, ipbtmp,
                     runtime, savepath, figstr, wpdat,
                     rex=1.0, rey=1.0, rez=1.0):
    """
    Select specific dispersion surfaces from the full solution set
    and plot them.

    Parameters
    ----------
    wwn : ndarray, shape (npa, npb, nw)
        All eigenvalue solutions normalized by wcs1 (omega/omega_{c1}).
    pa, pb : ndarray
        Scan parameter arrays.
    npa, npb : int
        Number of scan points along parameter a and b.
    ipa, ipb : int
        Scan parameter type indices.
    iloga, ilogb : int
        Log scale flags (0=linear, 1=log10).
    wcs1 : float
        Absolute cyclotron frequency of 1st species [rad/s].
    strpa, strpb : str
        Axis label strings for parameter a and b.
    betasz, betasp : ndarray or float
        Parallel and perpendicular beta values.
    alphas, Deltas : ndarray
        Loss-cone parameters (alpha_s, Delta_s).
    vA : float
        Alfven speed [m/s].
    c2 : float
        Speed of light squared [m^2/s^2].
    S, N, J : int
        Number of species, harmonics, and J-poles.
    iem : int
        Electromagnetic (1) or electrostatic (0) flag.
    par : ndarray
        Scan parameter array.
    ipbtmp : int
        Index (0-based) for the fixed parameter display.
    runtime : float
        Computation runtime in seconds.
    savepath : str
        Path to save figure files.
    figstr : str
        Figure filename identifier string.
    wpdat : ndarray, shape (npl, 3)
        Initial point data for each dispersion surface to select.
        Column 0: pa value at start point.
        Column 1: pb value at start point (arbitrary for 1D scan).
        Column 2: omega/omega_{c1} value at start point (complex).
                   If real part == 0, search uses imaginary part;
                   otherwise search uses real part.
    rex, rey, rez : float, optional
        Rescaling factors for plot axes. Default is 1.0.

    Returns
    -------
    wws : ndarray, shape (npa, npb, npl)
        Selected dispersion surface solutions (normalized by wcs1).
    """

    npl = wpdat.shape[0]
    wws = np.zeros((npa, npb, npl), dtype=complex)

    # ==================================================================
    # Search for dispersion surfaces matching each initial point
    # ==================================================================
    for jpl in range(npl):
        datstart = wpdat[jpl, :]

        if ipa == ipb:
            # ----------------------------------------------------------
            # 1D scan case
            # ----------------------------------------------------------

            # Find nearest pa index to start point
            indpa = np.argmin(np.abs(pa - datstart[0]))

            # Find nearest omega at start point
            if np.real(datstart[2]) == 0:
                # Search by imaginary part
                indww = np.argmin(np.abs(
                    np.imag(wwn[indpa, 0, :]) - np.imag(datstart[2])))
            else:
                # Search by real part
                indww = np.argmin(np.abs(
                    np.real(wwn[indpa, 0, :]) - np.real(datstart[2])))

            wwstart = wwn[indpa, 0, indww]

            # Trace dispersion surface across all pa values
            # MATLAB uses 1-based indices; here we use 0-based
            pas_arr = np.zeros(npa, dtype=float)
            ws = np.zeros(npa, dtype=complex)

            pas_arr[indpa] = pa[indpa]
            ws[indpa] = wwstart

            # jjpa: list of visited pa indices (0-based), mirrors MATLAB logic
            jjpa = [indpa]
            jjww = [indww]

            while len(jjpa) < npa:
                jjpa0 = list(jjpa)

                if max(jjpa) < npa - 1:
                    # Search right direction
                    indm = max(jjpa)
                    indpa_next = indm + 1
                    jjpa.append(indpa_next)
                else:
                    # Search left direction
                    indm = min(jjpa)
                    indpa_next = indm - 1
                    jjpa.insert(0, indpa_next)

                # Predict omega for next point
                if len(jjpa) <= 2:
                    wpred = ws[indm]  # nearest neighbor
                else:
                    # PCHIP interpolation with extrapolation
                    idx0 = np.array(jjpa0)
                    try:
                        wpred = pchip_extrap(
                            pas_arr[idx0], ws[idx0], pa[indpa_next])
                    except Exception:
                        wpred = ws[indm]

                # Find nearest solution to prediction
                indww = np.argmin(np.abs(wwn[indpa_next, 0, :] - wpred))
                jjww.append(indww)

                pas_arr[indpa_next] = pa[indpa_next]
                ws[indpa_next] = wwn[indpa_next, 0, indww]

            # Store selected surface (1D: npb=1, index 0)
            wws[:, 0, jpl] = ws[:]

        else:
            # ----------------------------------------------------------
            # 2D scan case
            # ----------------------------------------------------------

            # Find nearest (pa, pb) index to start point
            indpa = np.argmin(np.abs(pa - datstart[0]))
            indpb = np.argmin(np.abs(pb - datstart[1]))

            # Find nearest omega at start point
            if np.real(datstart[2]) == 0:
                indww = np.argmin(np.abs(
                    np.imag(wwn[indpa, indpb, :]) - np.imag(datstart[2])))
            else:
                indww = np.argmin(np.abs(
                    np.real(wwn[indpa, indpb, :]) - np.real(datstart[2])))

            wwstart = wwn[indpa, indpb, indww]

            # 2D tracing
            ws = np.zeros((npa, npb), dtype=complex)
            ws[indpa, indpb] = wwstart
            indpa0 = indpa
            indpb0 = indpb
            wpredb = wwstart

            jjpb = [indpb]

            while len(jjpb) <= npb:

                # --- Trace along parameter a for current indpb ---
                indpa = indpa0
                jjpa = [indpa]
                jjww = [indww]
                jstart = True

                while len(jjpa) < npa:
                    jjpa0 = list(jjpa)

                    if jstart:
                        wpred = wpredb
                        jstart = False
                    else:
                        if max(jjpa) < npa - 1:
                            # Search right
                            indm = max(jjpa)
                            indpa_next = indm + 1
                            jjpa.append(indpa_next)
                        else:
                            # Search left
                            indm = min(jjpa)
                            indpa_next = indm - 1
                            jjpa.insert(0, indpa_next)

                        if len(jjpa) <= 2:
                            wpred = ws[indm, indpb]  # nearest neighbor
                        else:
                            # PCHIP interpolation
                            idx0 = np.array(jjpa0)
                            try:
                                wpred = pchip_extrap(
                                    pa[idx0],
                                    ws[idx0, indpb],
                                    pa[indpa_next])
                            except Exception:
                                wpred = ws[indm, indpb]

                    # Find nearest solution to prediction
                    indww = np.argmin(
                        np.abs(wwn[indpa, indpb, :] - wpred))
                    jjww.append(indww)
                    ws[indpa, indpb] = wwn[indpa, indpb, indww]

                    # Note: indpa is updated at the beginning of next
                    # iteration via jjpa logic; on first iteration (jstart),
                    # indpa == indpa0 and we set ws[indpa0, indpb]

                # After tracing along pa, advance along pb
                jjpb0 = list(jjpb)

                if max(jjpb) < npb - 1:
                    # Search right in pb
                    indbm = max(jjpb)
                    indpb_next = indbm + 1
                    jjpb.append(indpb_next)
                else:
                    # Search left in pb
                    indbm = min(jjpb)
                    indpb_next = indbm - 1
                    jjpb.insert(0, indpb_next)

                indpb = indpb_next

                # Predict for next pb
                if len(jjpb) < npb:
                    if len(jjpb) <= 2:
                        wpredb = ws[indpa0, indbm]  # nearest neighbor
                    else:
                        idx_pb0 = np.array(jjpb0)
                        try:
                            wpredb = pchip_extrap(
                                pb[idx_pb0],
                                ws[indpa0, idx_pb0],
                                pb[indpb])
                        except Exception:
                            wpredb = ws[indpa0, indbm]

            # Store selected 2D surface
            wws[:, :, jpl] = ws[:, :]

    # ==================================================================
    # Plotting
    # ==================================================================

    # Color table for different dispersion surfaces
    pltc = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.2, 0.2, 1.0],
        [0.8, 0.8, 0.0],
        [1.0, 0.6, 0.0],
        [0.9, 0.0, 0.9],
        [0.0, 0.8, 0.8],
        [0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        [0.4, 0.7, 0.4],
        [0.0, 0.0, 0.5],
        [0.6, 0.0, 0.6],
        [0.0, 0.5, 1.0],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Meshgrid for 2D plots (equivalent to MATLAB ndgrid via indexing='ij')
    ppa, ppb = np.meshgrid(pa, pb, indexing='ij')

    for jpl in range(npl):
        color = pltc[jpl % len(pltc)]

        if ipa == ipb:
            # ----------------------------------------------------------
            # 1D plot
            # ----------------------------------------------------------

            # Left panel: real part of omega
            ax1 = axes[0]
            if iloga == 0:
                ax1.plot(rex * pa, rez * np.real(wws[:, 0, jpl]),
                         '--', color=color, linewidth=2)
                ax1.set_xlim([rex * np.min(pa), rex * np.max(pa)])
            else:
                ax1.semilogx(rex * 10.0**pa, rez * np.real(wws[:, 0, jpl]),
                             '--', color=color, linewidth=2)

            ax1.set_xlabel(f'{rex}*{strpa}, npa={npa}')
            ax1.set_ylabel(
                f'{rez}*' + r'$\omega_r/\omega_{c1}$'
                + f', ' + r'$\alpha$' + f'={alphas}')
            ax1.set_title(
                r'(a) $\beta_{||}$' + f'={betasz:.3g}'
                + r', $\beta_\perp$' + f'={betasp:.3g}')
            ax1.grid(True)

            # Right panel: imaginary part of omega
            ax2 = axes[1]
            if iloga == 0:
                ax2.plot(rex * pa, rez * np.imag(wws[:, 0, jpl]),
                         '--', color=color, linewidth=2)
                ax2.set_xlim([rex * np.min(pa), rex * np.max(pa)])
            else:
                ax2.semilogx(rex * 10.0**pa, rez * np.imag(wws[:, 0, jpl]),
                             '--', color=color, linewidth=2)

            ax2.set_xlabel(f'{rex}*{strpa}, iem={iem}')
            ax2.set_ylabel(
                f'{rez}*' + r'$\omega_i/\omega_{c1}$'
                + f', ' + r'$\Delta$' + f'={Deltas}')
            ax2.set_title(
                f'(b) $v_A/c$={vA / np.sqrt(c2):.2g}, {strpb}='
                + f'{par[ipbtmp]}, (S={S},N={N},J={J})')
            ax2.grid(True)

        else:
            # ----------------------------------------------------------
            # 2D surface plot
            # ----------------------------------------------------------
            wwjp = wws[:, :, jpl]

            # Left panel: real part
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.plot_surface(rex * ppa, rey * ppb, rez * np.real(wwjp),
                             alpha=0.7)
            ax1.set_xlabel(f'{rex}*{strpa}, ilogx={iloga}')
            ax1.set_ylabel(f'{rey}*{strpb}, ilogy={ilogb}')
            ax1.set_zlabel(
                f'{rez}*' + r'$\omega_r/\omega_{c1}$'
                + f', npa={npa}, npb={npb}')
            ax1.set_title(
                r'(a) $\beta_{||}$' + f'={betasz:.3g}'
                + r', $\beta_\perp$' + f'={betasp:.3g}')

            # Right panel: imaginary part
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.plot_surface(rex * ppa, rey * ppb, rez * np.imag(wwjp),
                             alpha=0.7)
            ax2.set_xlabel(f'{rex}*{strpa}')
            ax2.set_ylabel(f'{rey}*{strpb}')
            ax2.set_zlabel(
                f'{rez}*' + r'$\omega_i/\omega_{c1}$'
                + f', S={S}, N={N}, J={J}')
            ax2.set_title(f'(b) runtime={runtime}s')

    plt.tight_layout()

    # Save figures
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(os.path.join(savepath, f'fig_pdrk_{figstr}_select.png'),
                dpi=150)
    # Note: MATLAB .fig format not available in Python; save as PDF instead
    fig.savefig(os.path.join(savepath, f'fig_pdrk_{figstr}_select.pdf'))

    plt.show()

    return wws

