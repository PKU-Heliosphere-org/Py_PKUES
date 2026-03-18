"""
pdrk_plot_all.py
Interactive dispersion relation visualization and wave mode selector.

Enhanced with Hungarian-algorithm-based robust mode tracking that prevents
mode misidentification at crossings/avoided crossings.

Workflow (mirrors MATLAB pdrk_plot_all.m + pdrk_plot_select.m):
  1. Plot ALL eigenvalue solutions as scatter plots
  2. User clicks on the plot to select starting points for specific modes
  3. The selected points form 'wpdat'
  4. Automatically trace the selected dispersion surfaces using robust tracking
  5. Plot the traced results overlaid on the scatter

Interactive controls:
  - Left click:   Select/snap to the nearest solution point
  - Right click:  Remove the last selected point
  - Key 'Enter':  Confirm selection -> trace and plot selected modes
  - Key 'r':      Remove the last selected point (same as right click)
  - Key 'c':      Clear all selected points
  - Key 'q':      Quit without tracing
  - Key 'a':      Auto-select the most unstable mode and trace it
  - Key 't':      Toggle tracking method (simple / Hungarian)

Author: Converted from MATLAB pdrk suite by Hua-sheng XIE
        Hungarian tracking based on optimal assignment algorithm
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import linear_sum_assignment
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
from pkues_output import pkues_output
from pdrk_kernel import pdrk_kernel

PLTC = np.array([
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

def get_cycle_color(idx):
    """
    循环取色，保证不同模态颜色区分，并且实部/虚部对应一致。
    """
    return PLTC[idx % len(PLTC)]

# warnings.filterwarnings('ignore', category=np.ComplexWarning)


# ######################################################################
#  SECTION 1: PREDICTION AND MATCHING UTILITIES
# ######################################################################

def predict_next_omega(pa_hist, omega_hist, pa_next):
    """
    Predict the frequency at the next scan point using historical data.

    Strategy (adaptive polynomial / cubic spline):
      - 1 point:   constant value (nearest neighbor)
      - 2 points:  linear extrapolation
      - 3 points:  quadratic extrapolation
      - >= 4 pts:  local cubic spline using nearest 4 points

    Parameters
    ----------
    pa_hist : array_like, shape (nh,)
        Historical scan-parameter values (already visited, in order).
    omega_hist : array_like, shape (nh,), complex
        Historical omega values at those scan points.
    pa_next : float
        The next scan-parameter value to predict omega at.

    Returns
    -------
    omega_pred : complex
        Predicted complex frequency.
    """
    pa_hist = np.asarray(pa_hist, dtype=float)
    omega_hist = np.asarray(omega_hist, dtype=complex)

    nh = len(pa_hist)
    if nh == 0:
        raise ValueError("Empty history for mode prediction.")
    if nh == 1:
        return omega_hist[-1]

    # Use at most the nearest 4 points for local extrapolation
    use = min(4, nh)
    kh = pa_hist[-use:]
    wh = omega_hist[-use:]

    try:
        if use == 2:
            coef_r = np.polyfit(kh, np.real(wh), 1)
            coef_i = np.polyfit(kh, np.imag(wh), 1)
            return np.polyval(coef_r, pa_next) + 1j * np.polyval(coef_i, pa_next)

        if use == 3:
            coef_r = np.polyfit(kh, np.real(wh), 2)
            coef_i = np.polyfit(kh, np.imag(wh), 2)
            return np.polyval(coef_r, pa_next) + 1j * np.polyval(coef_i, pa_next)

        # use >= 4: cubic spline
        cs_r = CubicSpline(kh, np.real(wh), bc_type='not-a-knot',
                           extrapolate=True)
        cs_i = CubicSpline(kh, np.imag(wh), bc_type='not-a-knot',
                           extrapolate=True)
        return cs_r(pa_next) + 1j * cs_i(pa_next)
    except Exception:
        # Fallback: nearest neighbor
        return omega_hist[-1]


def pchip_extrap(x_known, y_known, x_query):
    """
    PCHIP interpolation with extrapolation for complex data.
    Equivalent to MATLAB: interp1(x, y, xq, 'pchip', 'extrap').
    """
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=complex)
    sort_idx = np.argsort(x_known)
    x_s = x_known[sort_idx]
    y_s = y_known[sort_idx]
    _, uid = np.unique(x_s, return_index=True)
    x_s, y_s = x_s[uid], y_s[uid]
    if len(x_s) < 2:
        return y_s[0] if len(y_s) > 0 else 0.0 + 0j
    f_re = PchipInterpolator(x_s, np.real(y_s), extrapolate=True)
    f_im = PchipInterpolator(x_s, np.imag(y_s), extrapolate=True)
    return f_re(x_query) + 1j * f_im(x_query)


# ######################################################################
#  SECTION 2: MODE TRACING ALGORITHMS
# ######################################################################

# ----------------------------------------------------------------------
# 2a. Original simple tracing (single mode, backward compatible)
# ----------------------------------------------------------------------
def _trace_surface_1d_simple(wwn, pa, npa, indpa_start, indww_start):
    """
    Original PCHIP + nearest-neighbor tracing for a single mode (1D).
    Faithful to MATLAB pdrk_plot_select.m logic.
    """
    ws = np.zeros(npa, dtype=complex)
    pas_arr = np.zeros(npa, dtype=float)

    ws[indpa_start] = wwn[indpa_start, 0, indww_start]
    pas_arr[indpa_start] = pa[indpa_start]
    jjpa = [indpa_start]

    while len(jjpa) < npa:
        jjpa0 = list(jjpa)
        if max(jjpa) < npa - 1:
            indm = max(jjpa)
            indpa_next = indm + 1
            jjpa.append(indpa_next)
        else:
            indm = min(jjpa)
            indpa_next = indm - 1
            jjpa.insert(0, indpa_next)

        if len(jjpa) <= 2:
            wpred = ws[indm]
        else:
            idx0 = np.array(jjpa0)
            try:
                wpred = pchip_extrap(pas_arr[idx0], ws[idx0], pa[indpa_next])
            except Exception:
                wpred = ws[indm]

        indww = np.argmin(np.abs(wwn[indpa_next, 0, :] - wpred))
        pas_arr[indpa_next] = pa[indpa_next]
        ws[indpa_next] = wwn[indpa_next, 0, indww]

    return ws


# ----------------------------------------------------------------------
# 2b. Hungarian-algorithm-based robust multi-mode tracing (1D)
# ----------------------------------------------------------------------
def _trace_surfaces_1d_hungarian(wwn, pa, npa, start_list,
                                 pred_weight=0.70):
    """
    Simultaneously trace multiple dispersion surfaces using the Hungarian
    algorithm for optimal assignment at each scan step.

    This prevents mode misidentification at crossings by:
      1. Predicting each tracked mode's next omega from its history
         (adaptive poly/spline extrapolation)
      2. Building a cost matrix: rows = tracked modes, cols = candidate
         eigenvalues, cost = weighted sum of prediction distance and
         previous-step distance
      3. Using the Hungarian algorithm (linear_sum_assignment) for
         globally optimal one-to-one matching

    Parameters
    ----------
    wwn : ndarray, shape (npa, npb, nw)
        All normalized eigenvalue solutions.
    pa : ndarray, shape (npa,)
        Scan parameter array.
    npa : int
        Number of scan points.
    start_list : list of (indpa, indww)
        Starting points for each mode to trace.
    pred_weight : float
        Weight for the prediction-based cost term (0 to 1).
        The remaining (1 - pred_weight) goes to the previous-step
        distance term.  Default 0.70.

    Returns
    -------
    wws : ndarray, shape (npa, npl), complex
        Traced dispersion surfaces. wws[:, m] is the m-th mode.
    """
    npl = len(start_list)
    nw = wwn.shape[2]
    wws = np.zeros((npa, npl), dtype=complex)

    # Initialize at starting points
    for m, (indpa_s, indww_s) in enumerate(start_list):
        wws[indpa_s, m] = wwn[indpa_s, 0, indww_s]

    # Determine the starting pa index (must be the same for all modes
    # in simultaneous tracking — use the most common one)
    start_indices = [s[0] for s in start_list]
    indpa_start = max(set(start_indices), key=start_indices.count)

    # If modes start at different pa, fill them at the common start
    for m, (indpa_s, indww_s) in enumerate(start_list):
        if indpa_s != indpa_start:
            # Find closest eigenvalue to the original start
            target = wwn[indpa_s, 0, indww_s]
            indww_new = np.argmin(np.abs(wwn[indpa_start, 0, :] - target))
            wws[indpa_start, m] = wwn[indpa_start, 0, indww_new]

    # Build visit order: from start, go right first, then left
    visit_order = []
    # Right direction
    for i in range(indpa_start + 1, npa):
        visit_order.append(i)
    # Left direction
    for i in range(indpa_start - 1, -1, -1):
        visit_order.append(i)

    # History arrays for each mode (pa values and omega values, in order)
    pa_hists = [[] for _ in range(npl)]
    w_hists = [[] for _ in range(npl)]

    # Seed history with the starting point
    for m in range(npl):
        pa_hists[m].append(pa[indpa_start])
        w_hists[m].append(wws[indpa_start, m])

    # ---------- Trace rightward from start ----------
    right_steps = [i for i in visit_order if i > indpa_start]
    left_steps = [i for i in visit_order if i < indpa_start]

    for direction_steps in [right_steps, left_steps]:
        # Reset histories for this direction
        if direction_steps is left_steps:
            # Re-seed: use starting point only (will build leftward)
            pa_hists = [[] for _ in range(npl)]
            w_hists = [[] for _ in range(npl)]
            for m in range(npl):
                pa_hists[m].append(pa[indpa_start])
                w_hists[m].append(wws[indpa_start, m])

        for ipa in direction_steps:
            candidates = wwn[ipa, 0, :]  # all eigenvalues at this pa

            # Predict each mode's omega at this pa
            pred_w = np.zeros(npl, dtype=complex)
            prev_w = np.zeros(npl, dtype=complex)
            for m in range(npl):
                pred_w[m] = predict_next_omega(
                    pa_hists[m], w_hists[m], pa[ipa])
                prev_w[m] = w_hists[m][-1]

            # Build cost matrix: (npl modes) x (nw candidates)
            cost = np.zeros((npl, nw), dtype=float)
            for m in range(npl):
                dw_pred = np.abs(candidates - pred_w[m])
                dw_prev = np.abs(candidates - prev_w[m])
                cost[m, :] = (pred_weight * dw_pred +
                              (1.0 - pred_weight) * dw_prev)

            # Hungarian algorithm: find optimal assignment
            # rows = modes (npl), cols = candidates (nw)
            # If npl < nw, only npl candidates will be chosen
            row_ind, col_ind = linear_sum_assignment(cost)

            for row, col in zip(row_ind, col_ind):
                wws[ipa, row] = candidates[col]

            # Update histories
            for m in range(npl):
                pa_hists[m].append(pa[ipa])
                w_hists[m].append(wws[ipa, m])

    return wws


# ----------------------------------------------------------------------
# 2c. Simple tracing for 2D case
# ----------------------------------------------------------------------
def _trace_surface_2d_simple(wwn, pa, pb, npa, npb,
                             indpa_start, indpb_start, indww_start):
    """
    Original PCHIP + nearest-neighbor tracing for a single mode (2D).
    """
    ws = np.zeros((npa, npb), dtype=complex)
    wwstart = wwn[indpa_start, indpb_start, indww_start]
    ws[indpa_start, indpb_start] = wwstart
    indpa0 = indpa_start
    wpredb = wwstart
    indpb = indpb_start
    indww = indww_start

    jjpb = [indpb_start]

    while len(jjpb) <= npb:
        indpa = indpa0
        jjpa = [indpa]
        jstart = True

        while len(jjpa) < npa:
            jjpa0 = list(jjpa)
            if jstart:
                wpred = wpredb
                jstart = False
            else:
                if max(jjpa) < npa - 1:
                    indm = max(jjpa)
                    indpa_next = indm + 1
                    jjpa.append(indpa_next)
                else:
                    indm = min(jjpa)
                    indpa_next = indm - 1
                    jjpa.insert(0, indpa_next)

                if len(jjpa) <= 2:
                    wpred = ws[indm, indpb]
                else:
                    idx0 = np.array(jjpa0)
                    try:
                        wpred = pchip_extrap(
                            pa[idx0], ws[idx0, indpb], pa[indpa_next])
                    except Exception:
                        wpred = ws[indm, indpb]

                indpa = indpa_next

            indww = np.argmin(np.abs(wwn[indpa, indpb, :] - wpred))
            ws[indpa, indpb] = wwn[indpa, indpb, indww]

        jjpb0 = list(jjpb)
        if max(jjpb) < npb - 1:
            indbm = max(jjpb)
            indpb_next = indbm + 1
            jjpb.append(indpb_next)
        else:
            indbm = min(jjpb)
            indpb_next = indbm - 1
            jjpb.insert(0, indpb_next)
        indpb = indpb_next

        if len(jjpb) < npb:
            if len(jjpb) <= 2:
                wpredb = ws[indpa0, indbm]
            else:
                idx_pb0 = np.array(jjpb0)
                try:
                    wpredb = pchip_extrap(
                        pb[idx_pb0], ws[indpa0, idx_pb0], pb[indpb])
                except Exception:
                    wpredb = ws[indpa0, indbm]

    return ws


# ----------------------------------------------------------------------
# 2d. Hungarian-algorithm-based robust multi-mode tracing (2D)
# ----------------------------------------------------------------------
def _trace_surfaces_2d_hungarian(wwn, pa, pb, npa, npb, start_list,
                                 pred_weight=0.70):
    """
    Simultaneously trace multiple modes over the 2D (pa, pb) grid
    using Hungarian assignment.

    For each pb slice, traces all modes along pa simultaneously using
    prediction + Hungarian matching.

    Parameters
    ----------
    wwn : ndarray, shape (npa, npb, nw)
    pa, pb : ndarray
    npa, npb : int
    start_list : list of (indpa, indpb, indww)
    pred_weight : float

    Returns
    -------
    wws : ndarray, shape (npa, npb, npl), complex
    """
    npl = len(start_list)
    nw = wwn.shape[2]
    wws = np.zeros((npa, npb, npl), dtype=complex)

    # Find common starting pb
    indpb_start = start_list[0][1]
    indpa_start = start_list[0][0]

    # Initialize at starting points
    for m, (ipa_s, ipb_s, iww_s) in enumerate(start_list):
        wws[ipa_s, ipb_s, m] = wwn[ipa_s, ipb_s, iww_s]

    # Build pb visit order from indpb_start
    pb_right = list(range(indpb_start + 1, npb))
    pb_left = list(range(indpb_start - 1, -1, -1))
    pb_order = [indpb_start] + pb_right + pb_left

    for ipb in pb_order:
        # For this pb slice, do Hungarian tracking along pa
        # Determine seeds: if ipb == indpb_start, use original starts;
        # otherwise, use prediction from the previous pb slice
        if ipb == indpb_start:
            # Seed at starting points
            for m, (ipa_s, ipb_s, iww_s) in enumerate(start_list):
                if ipb_s == ipb:
                    wws[ipa_s, ipb, m] = wwn[ipa_s, ipb, iww_s]
                else:
                    target = wws[ipa_s, ipb_s, m]
                    iww_new = np.argmin(np.abs(wwn[ipa_s, ipb, :] - target))
                    wws[ipa_s, ipb, m] = wwn[ipa_s, ipb, iww_new]
        else:
            # Use prediction from neighbor pb
            if ipb > indpb_start:
                ipb_prev = ipb - 1
            else:
                ipb_prev = ipb + 1
            for m in range(npl):
                target = wws[indpa_start, ipb_prev, m]
                iww_new = np.argmin(np.abs(wwn[indpa_start, ipb, :] - target))
                wws[indpa_start, ipb, m] = wwn[indpa_start, ipb, iww_new]

        # Now trace along pa at this ipb
        pa_right = list(range(indpa_start + 1, npa))
        pa_left = list(range(indpa_start - 1, -1, -1))

        for direction_steps in [pa_right, pa_left]:
            pa_hists = [[pa[indpa_start]] for _ in range(npl)]
            w_hists = [[wws[indpa_start, ipb, m]] for m in range(npl)]

            for ipa in direction_steps:
                candidates = wwn[ipa, ipb, :]
                pred_w = np.zeros(npl, dtype=complex)
                prev_w = np.zeros(npl, dtype=complex)
                for m in range(npl):
                    pred_w[m] = predict_next_omega(
                        pa_hists[m], w_hists[m], pa[ipa])
                    prev_w[m] = w_hists[m][-1]

                cost = np.zeros((npl, nw), dtype=float)
                for m in range(npl):
                    dw_pred = np.abs(candidates - pred_w[m])
                    dw_prev = np.abs(candidates - prev_w[m])
                    cost[m, :] = (pred_weight * dw_pred +
                                  (1.0 - pred_weight) * dw_prev)

                row_ind, col_ind = linear_sum_assignment(cost)
                for row, col in zip(row_ind, col_ind):
                    wws[ipa, ipb, row] = candidates[col]

                for m in range(npl):
                    pa_hists[m].append(pa[ipa])
                    w_hists[m].append(wws[ipa, ipb, m])

    return wws

def sort_all_modes_1d_hungarian(wwn, pa, pred_weight=0.70):
    """
    Reorder all eigenmodes along pa using Hungarian assignment so that
    the same mode index corresponds to the same physical branch as much
    as possible.

    Parameters
    ----------
    wwn : ndarray, shape (npa, 1, nw)
        Unsorted eigenvalue solutions (normalized).
    pa : ndarray, shape (npa,)
        Scan parameter.
    pred_weight : float
        Weight for prediction term in the Hungarian cost.

    Returns
    -------
    wwn_sorted : ndarray, shape (npa, 1, nw)
        Reordered eigenvalue array.
    """
    npa, npb, nw = wwn.shape
    if npb != 1:
        raise ValueError("sort_all_modes_1d_hungarian requires npb == 1.")

    wwn_sorted = np.zeros_like(wwn, dtype=complex)

    # First scan point: keep original order
    wwn_sorted[0, 0, :] = wwn[0, 0, :]

    # History for each mode
    pa_hists = [[pa[0]] for _ in range(nw)]
    w_hists = [[wwn_sorted[0, 0, i]] for i in range(nw)]

    for ipa in range(1, npa):
        candidates = wwn[ipa, 0, :]

        pred_w = np.zeros(nw, dtype=complex)
        prev_w = np.zeros(nw, dtype=complex)

        for m in range(nw):
            pred_w[m] = predict_next_omega(pa_hists[m], w_hists[m], pa[ipa])
            prev_w[m] = w_hists[m][-1]

        cost = np.zeros((nw, nw), dtype=float)
        for m in range(nw):
            dw_pred = np.abs(candidates - pred_w[m])
            dw_prev = np.abs(candidates - prev_w[m])
            cost[m, :] = pred_weight * dw_pred + (1.0 - pred_weight) * dw_prev

        row_ind, col_ind = linear_sum_assignment(cost)

        # row = tracked mode index, col = candidate index
        new_row = np.zeros(nw, dtype=complex)
        for row, col in zip(row_ind, col_ind):
            new_row[row] = candidates[col]

        wwn_sorted[ipa, 0, :] = new_row

        for m in range(nw):
            pa_hists[m].append(pa[ipa])
            w_hists[m].append(wwn_sorted[ipa, 0, m])

    return wwn_sorted


# ######################################################################
#  SECTION 3: INTERACTIVE MODE SELECTORS
# ######################################################################

class _ModeSelector1D:
    """Interactive matplotlib handler for 1D dispersion plots."""

    def __init__(self, fig, ax_re, ax_im, wwn, pa, npa, nw,
                 iloga, rex, rez):
        self.fig = fig
        self.ax_re = ax_re
        self.ax_im = ax_im
        self.wwn = wwn
        self.pa = pa
        self.npa = npa
        self.nw = nw
        self.iloga = iloga
        self.rex = rex
        self.rez = rez

        if iloga == 0:
            self.x_plot = rex * pa
        else:
            self.x_plot = rex * 10.0**pa

        self.wr_all = rez * np.real(wwn[:, 0, :])
        self.wi_all = rez * np.imag(wwn[:, 0, :])

        self.selected_points = []
        self.markers_re = []
        self.markers_im = []
        self.texts_re = []
        self.texts_im = []

        self.done = False
        self.cancelled = False
        self.use_hungarian = True  # default: use robust tracking

        self.cid_click = fig.canvas.mpl_connect('button_press_event',
                                                 self._on_click)
        self.cid_key = fig.canvas.mpl_connect('key_press_event',
                                               self._on_key)

        method_str = 'Hungarian (robust)' if self.use_hungarian else 'Simple (basic)'
        self.status_text = fig.text(
            0.5, 0.01,
            f'Left-click: select | Right-click/r: undo | c: clear | a: auto most unstable '
            f'| t: toggle tracking({method_str}) | Enter: confirm | q: quit',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _find_nearest(self, ax, xdata, ydata):
        if ax == self.ax_re:
            y_arr = self.wr_all
        else:
            y_arr = self.wi_all

        best_dist = np.inf
        best_ipa, best_iww = 0, 0
        xrange = max(self.x_plot[-1] - self.x_plot[0], 1e-30)
        yrange = max(np.nanmax(y_arr) - np.nanmin(y_arr), 1e-30)

        for ipa in range(self.npa):
            for iww in range(self.nw):
                dx = (self.x_plot[ipa] - xdata) / xrange
                dy = (y_arr[ipa, iww] - ydata) / yrange
                dist = dx**2 + dy**2
                if dist < best_dist:
                    best_dist = dist
                    best_ipa = ipa
                    best_iww = iww
        return best_ipa, best_iww

    def _on_click(self, event):
        if self.done or event.inaxes not in [self.ax_re, self.ax_im]:
            return
        if event.button == 1:
            indpa, indww = self._find_nearest(event.inaxes,
                                               event.xdata, event.ydata)
            wn_val = self.wwn[indpa, 0, indww]
            pa_val = self.pa[indpa]
            self.selected_points.append((indpa, indww, pa_val, wn_val))
            self._draw_marker(indpa, indww, len(self.selected_points))
        elif event.button == 3:
            self._remove_last()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if self.done:
            return
        if event.key == 'enter':
            self.done = True
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.fig.canvas.mpl_disconnect(self.cid_key)
            plt.close(self.fig)
        elif event.key in ('r', 'R'):
            self._remove_last()
            self.fig.canvas.draw_idle()
        elif event.key in ('c', 'C'):
            self._clear_all()
            self.fig.canvas.draw_idle()
        elif event.key in ('a', 'A'):
            self._auto_select_most_unstable()
            self.fig.canvas.draw_idle()
        elif event.key in ('t', 'T'):
            self.use_hungarian = not self.use_hungarian
            method_str = 'Hungarian (robust)' if self.use_hungarian else 'Simple (basic)'
            self.status_text.set_text(
                f'Left-click: select | Right-click/r: undo | c: clear | a: auto most unstable '
                f'| t: toggle tracking({method_str}) | Enter: confirm | q: quit')
            self.fig.canvas.draw_idle()
        elif event.key in ('q', 'Q'):
            self.done = True
            self.cancelled = True
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.fig.canvas.mpl_disconnect(self.cid_key)
            plt.close(self.fig)

    def _draw_marker(self, indpa, indww, idx):
        x = self.x_plot[indpa]
        yr = self.wr_all[indpa, indww]
        yi = self.wi_all[indpa, indww]
        m_re, = self.ax_re.plot(x, yr, 'o', color='red', markersize=10,
                                markeredgewidth=2, markerfacecolor='none',
                                zorder=10)
        m_im, = self.ax_im.plot(x, yi, 'o', color='red', markersize=10,
                                markeredgewidth=2, markerfacecolor='none',
                                zorder=10)
        t_re = self.ax_re.annotate(f'{idx}', (x, yr), fontsize=10,
                                    fontweight='bold', color='red',
                                    xytext=(5, 5),
                                    textcoords='offset points', zorder=10)
        t_im = self.ax_im.annotate(f'{idx}', (x, yi), fontsize=10,
                                    fontweight='bold', color='red',
                                    xytext=(5, 5),
                                    textcoords='offset points', zorder=10)
        self.markers_re.append(m_re)
        self.markers_im.append(m_im)
        self.texts_re.append(t_re)
        self.texts_im.append(t_im)

    def _remove_last(self):
        if not self.selected_points:
            return
        self.selected_points.pop()
        for lst in [self.markers_re, self.markers_im,
                    self.texts_re, self.texts_im]:
            artist = lst.pop()
            artist.remove()

    def _clear_all(self):
        while self.selected_points:
            self._remove_last()

    def _auto_select_most_unstable(self):
        wi = np.imag(self.wwn[:, 0, :])
        idx = np.unravel_index(np.argmax(wi), wi.shape)
        indpa, indww = idx[0], idx[1]
        wn_val = self.wwn[indpa, 0, indww]
        pa_val = self.pa[indpa]
        self.selected_points.append((indpa, indww, pa_val, wn_val))
        self._draw_marker(indpa, indww, len(self.selected_points))


class _ModeSelector2D:
    """Interactive matplotlib handler for 2D dispersion surface plots."""

    def __init__(self, fig, ax_re, ax_im, wwn, pa, pb, npa, npb, nw,
                 rex, rey, rez):
        self.fig = fig
        self.ax_re = ax_re
        self.ax_im = ax_im
        self.wwn = wwn
        self.pa = pa
        self.pb = pb
        self.npa = npa
        self.npb = npb
        self.nw = nw
        self.rex = rex
        self.rey = rey
        self.rez = rez

        self.selected_points = []
        self.markers = []
        self.done = False
        self.cancelled = False
        self.use_hungarian = True

        self.cid_click = fig.canvas.mpl_connect('button_press_event',
                                                 self._on_click)
        self.cid_key = fig.canvas.mpl_connect('key_press_event',
                                               self._on_key)

        self.status_text = fig.text(
            0.5, 0.01,
            'Left-click: select | Right-click/r: undo | c: clear | a: auto '
            '| t: toggle tracking | Enter: confirm | q: quit',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _find_nearest_2d(self, xdata, ydata, use_imag=False):
        pa_range = max(self.pa[-1] - self.pa[0], 1e-30)
        pb_range = max(self.pb[-1] - self.pb[0], 1e-30) if self.npb > 1 else 1.0
        best_dist = np.inf
        best_ipa, best_ipb, best_iww = 0, 0, 0
        for ipa in range(self.npa):
            for ipb in range(self.npb):
                dx = (self.rex * self.pa[ipa] - xdata) / pa_range
                dy = (self.rey * self.pb[ipb] - ydata) / pb_range
                dist = dx**2 + dy**2
                if dist < best_dist:
                    best_dist = dist
                    best_ipa = ipa
                    best_ipb = ipb
        best_iww = np.argmax(np.imag(self.wwn[best_ipa, best_ipb, :]))
        return best_ipa, best_ipb, best_iww

    def _on_click(self, event):
        if self.done or event.inaxes not in [self.ax_re, self.ax_im]:
            return
        if event.button == 1:
            use_im = (event.inaxes == self.ax_im)
            ipa, ipb, iww = self._find_nearest_2d(
                event.xdata, event.ydata, use_imag=use_im)
            wn_val = self.wwn[ipa, ipb, iww]
            self.selected_points.append(
                (ipa, ipb, iww, self.pa[ipa], self.pb[ipb], wn_val))
            for ax in [self.ax_re, self.ax_im]:
                m, = ax.plot(self.rex * self.pa[ipa],
                             self.rey * self.pb[ipb],
                             'ro', markersize=12, markeredgewidth=2,
                             markerfacecolor='none', zorder=10)
                self.markers.append(m)
            self.fig.canvas.draw_idle()
        elif event.button == 3:
            self._remove_last()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if self.done:
            return
        if event.key == 'enter':
            self.done = True
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.fig.canvas.mpl_disconnect(self.cid_key)
            plt.close(self.fig)
        elif event.key in ('r', 'R'):
            self._remove_last()
            self.fig.canvas.draw_idle()
        elif event.key in ('c', 'C'):
            while self.selected_points:
                self._remove_last()
            self.fig.canvas.draw_idle()
        elif event.key in ('a', 'A'):
            wi = np.imag(self.wwn)
            idx = np.unravel_index(np.argmax(wi), wi.shape)
            wn_val = self.wwn[idx]
            self.selected_points.append(
                (idx[0], idx[1], idx[2],
                 self.pa[idx[0]], self.pb[idx[1]], wn_val))
            for ax in [self.ax_re, self.ax_im]:
                m, = ax.plot(self.rex * self.pa[idx[0]],
                             self.rey * self.pb[idx[1]],
                             'ro', markersize=12, markeredgewidth=2,
                             markerfacecolor='none', zorder=10)
                self.markers.append(m)
            self.fig.canvas.draw_idle()
        elif event.key in ('t', 'T'):
            self.use_hungarian = not self.use_hungarian
            method_str = 'Hungarian' if self.use_hungarian else 'Simple'
            self.status_text.set_text(
                f'Left-click: select | Right-click/r: undo | c: clear | a: auto '
                f'| t: toggle tracking({method_str}) | Enter: confirm | q: quit')
            self.fig.canvas.draw_idle()
        elif event.key in ('q', 'Q'):
            self.done = True
            self.cancelled = True
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.fig.canvas.mpl_disconnect(self.cid_key)
            plt.close(self.fig)

    def _remove_last(self):
        if not self.selected_points:
            return
        self.selected_points.pop()
        for _ in range(min(2, len(self.markers))):
            m = self.markers.pop()
            m.remove()


# ######################################################################
#  SECTION 4: MAIN ENTRY POINTS
# ######################################################################


def _run_pkues_output_from_selected_modes(
        init, wws, pa, pb, npa, npb, ipa, ipb, iloga, ilogb,
        strpa, strpb, savepath, figstr, betasz, betasp, start_list,
        idf=0, jpa_df=0, jpb_df=0, jpl_df=0, s_df=0, vdf_config=None):
    """
    对已追踪得到的选中波模 wws，复用 pdrk_kernel(icalp=1) 计算
    wws2 / Pola / Pola_norm / Pola_SI / dV / dVnorm / JE / Zp_norm / Zm_norm，
    然后直接调用 pkues_output 出图。
    """
    init2 = init.copy()
    nw0 = init2['nw']

    init2['iout'] = 2   # 必须让 pdrk_kernel 走 polarization run 分支
    init2['sp'] = 1
    init2['nw'] = 1
    init2['icalp'] = 1
    if 'sp0' in init2:
        init2['sp0'] = 1

    S = init2['S']
    N = init2['N']
    J = init2['J']
    nw0 = init2['nw']
    ns0 = init2['ns0']
    B0 = init2['B0']
    theta = init2['par'][1]

    npl = wws.shape[2]
    ppa, ppb = np.meshgrid(pa, pb, indexing='ij')

    # 颜色沿用你在选模图上的分支颜色
    pltc = np.array([get_cycle_color(item[-1]) for item in start_list])

    # 下面这些数组的形状，直接按 pdrk_kernel / pkues_velocity 的现有写法来
    wws2 = np.zeros((npa, npb, npl), dtype=complex)
    Pola = np.zeros((npa, npb, npl, 8), dtype=complex)
    Pola_norm = np.zeros((npa, npb, npl, 11), dtype=complex)
    Pola_SI = np.zeros((npa, npb, npl, 11), dtype=complex)

    Js = np.zeros((npa, 3, S, npl), dtype=complex)
    dV = np.zeros((npa, 3, S, npl), dtype=complex)
    dVnorm = np.zeros((npa, 3, S, npl), dtype=complex)
    xinorm = np.zeros((npa, S, npl), dtype=complex)
    JE = np.zeros((npa, 3, S, npl), dtype=complex)

    Zp_norm = np.zeros((npa, 3, S, npl), dtype=complex)
    Zm_norm = np.zeros((npa, 3, S, npl), dtype=complex)
    scaling = np.zeros((npa, npb, npl), dtype=complex)

    # 对每一支已选波模，复用 pdrk_kernel 的第二遍计算
    for jpl in range(npl):
        pdrk_kernel(
            init2,
            icalp=1,
            wws=wws,
            wws2=wws2,
            Pola=Pola,
            Pola_norm=Pola_norm,
            Pola_SI=Pola_SI,
            jpl=jpl,
            Js=Js,
            dV=dV,
            dVnorm=dVnorm,
            xinorm=xinorm,
            JE=JE,
            Zp_norm=Zp_norm,
            Zm_norm=Zm_norm,
            scaling=scaling,
            idf=idf,
            jpa_df=jpa_df,
            jpb_df=jpb_df,
            jpl_df=jpl_df,
            s_df=s_df,
            vdf_config=vdf_config,
            savepath=savepath,
            figstr=figstr
        )

    # pkues_output 里是直接拼 savepath + filename，最好保证末尾有分隔符
    if not savepath.endswith(os.sep):
        savepath = savepath + os.sep

    pkues_output(
        wws=wws,
        wws2=wws2,
        Pola=Pola,
        Pola_norm=Pola_norm,
        Pola_SI=Pola_SI,
        dV=dV,
        dVnorm=dVnorm,
        JE=JE,
        Zp_norm=Zp_norm,
        Zm_norm=Zm_norm,
        ns0=ns0,
        npa=npa,
        npb=npb,
        npl=npl,
        nw0=nw0,
        ipa=ipa,
        ipb=ipb,
        iloga=iloga,
        ilogb=ilogb,
        pa=pa,
        pas=pa,
        ppa=ppa,
        ppb=ppb,
        strpa=strpa,
        strpb=strpb,
        S=S,
        N=N,
        J=J,
        B0=B0,
        theta=theta,
        pltc=pltc,
        savepath=savepath,
        figstr=figstr,
        betasz=betasz,
        betasp=betasp,
        jpb=0
    )

def pdrk_plot_all_interactive(ww, pa, pb, npa, npb, ipa, ipb,
                               iloga=0, ilogb=0,
                               wcs1=1.0, strpa='pa', strpb='pb',
                               betasz=0.0, betasp=0.0,
                               alphas=1.0, Deltas=1.0,
                               vA=1.0, c2=(2.9979e8)**2,
                               S=1, N=1, J=8, iem=1,
                               par=None, ipbtmp=0,
                               runtime=0.0,
                               savepath='./', figstr='test',
                               rex=1.0, rey=1.0, rez=1.0,
                               wpdat=None, pred_weight=0.70,
                               init=None, run_pkues_output=True,
                               idf=0, jpa_df=0, jpb_df=0,
                               jpl_df=0, s_df=0, vdf_config=None):
    """
    Interactive plot of ALL dispersion solutions with click-to-select
    and robust Hungarian-algorithm-based mode tracing.

    Parameters
    ----------
    ww : ndarray, shape (npa, npb, nw)
        All eigenvalue solutions [rad/s].
    pa, pb : ndarray
        Scan parameter arrays.
    npa, npb : int
        Number of scan points.
    ipa, ipb : int
        Scan parameter type indices.
    iloga, ilogb : int
        Log scale flags.
    wcs1 : float
        |omega_{c1}| [rad/s].
    strpa, strpb : str
        Axis label strings.
    betasz, betasp : float or ndarray
        Parallel and perpendicular beta.
    alphas, Deltas : float or ndarray
        Loss-cone parameters.
    vA : float
        Alfven speed [m/s].
    c2 : float
        Speed of light squared.
    S, N, J, iem : int
        Species count, harmonics, poles, EM flag.
    par : ndarray or None
        Parameter array (for display).
    ipbtmp : int
        Index (0-based) for fixed parameter display.
    runtime : float
        Computation runtime [s].
    savepath : str
        Output figure directory.
    figstr : str
        Figure filename identifier.
    rex, rey, rez : float
        Rescaling factors.
    wpdat : ndarray or None, shape (npl, 3)
        Pre-set starting points. If None, interactive selection is used.
    pred_weight : float
        Weight for prediction term in Hungarian cost (0 to 1).

    Returns
    -------
    wws : ndarray, shape (npa, npb, npl)
        Traced dispersion surfaces (normalized by wcs1).
    wpdat_out : ndarray, shape (npl, 3)
        The wpdat used.
    """
    wwn = ww / wcs1
    nw = wwn.shape[2]

    # Pre-sort all modes before plotting, so that the same index corresponds
    # to the same physical branch as much as possible.
    if ipa == ipb:
        wwn_plot = sort_all_modes_1d_hungarian(wwn, pa, pred_weight=pred_weight)
    else:
        wwn_plot = wwn
    if par is None:
        par = np.zeros(5)

    use_hungarian = True  # default
    start_list = None

    # ==================================================================
    # Phase 1: Scatter plot & interactive selection (if wpdat not given)
    # ==================================================================
    if wpdat is None:

        if ipa == ipb:
            # ---- 1D scatter plot ----
            fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(15, 5.5))
            fig.subplots_adjust(bottom=0.12)

            x_vals = rex * pa if iloga == 0 else rex * 10.0**pa

            for iww in range(nw):
                color = get_cycle_color(iww)
                wr = rez * np.real(wwn_plot[:, 0, iww])
                wi = rez * np.imag(wwn_plot[:, 0, iww])

                ax_re.plot(
                    x_vals, wr,
                    '.-', color=color, markersize=3, linewidth=0.8, alpha=0.9
                )
                ax_im.plot(
                    x_vals, wi,
                    '.-', color=color, markersize=3, linewidth=0.8, alpha=0.9
                )

            ax_re.set_xlabel(f'{rex}*{strpa}, npa={npa}')
            ax_re.set_ylabel(f'{rez}*' + r'$\omega_r / \omega_{c1}$')
            ax_re.set_title(r'(a) $\omega_r$: real part')
            ax_re.grid(True, alpha=0.3)

            ax_im.set_xlabel(f'{rex}*{strpa}, iem={iem}')
            ax_im.set_ylabel(f'{rez}*' + r'$\omega_i / \omega_{c1}$')
            ax_im.set_title(r'(b) $\omega_i$: growth rate')
            ax_im.grid(True, alpha=0.3)

            if iloga == 1:
                ax_re.set_xscale('log')
                ax_im.set_xscale('log')

            selector = _ModeSelector1D(fig, ax_re, ax_im, wwn_plot, pa, npa, nw,
                                        iloga, rex, rez)
            print("=" * 65)
            print("  Interactive Mode Selector")
            print("=" * 65)
            print("  Left-click:  select the starting mode (auto-snaps to nearest solution)")
            print("  Right-click/r: undo last selection")
            print("  c:           clear all selections")
            print("  a:           auto-select the most unstable mode")
            print("  t:           toggle tracking method (Hungarian (robust) / Simple (basic))")
            print("  Enter:       confirm selection and start tracking")
            print("  q:           quit (do not track)")
            print("=" * 65)

            plt.show(block=True)

            if selector.cancelled or len(selector.selected_points) == 0:
                print("No modes selected, exiting.")
                return None, None

            use_hungarian = selector.use_hungarian
            start_list = [(indpa, indww)
                             for indpa, indww, pa_val, wn_val in selector.selected_points]

            npl = len(selector.selected_points)
            wpdat = np.zeros((npl, 3), dtype=complex)
            for i, (indpa, indww, pa_val, wn_val) in enumerate(
                    selector.selected_points):
                wpdat[i, 0] = pa_val
                wpdat[i, 1] = 0.0
                wpdat[i, 2] = wn_val

            print(f"\nSelected {npl} mode starting points:")
            for i in range(npl):
                print(f"  [{i+1}] pa={np.real(wpdat[i,0]):.4g}, "
                      f"omega/wc1 = {wpdat[i,2]:.6g}")

        else:
            # ---- 2D contour ----
            max_gamma = np.max(np.imag(wwn), axis=2)
            max_wr = np.zeros((npa, npb))
            for ipa_i in range(npa):
                for ipb_i in range(npb):
                    idx_max = np.argmax(np.imag(wwn[ipa_i, ipb_i, :]))
                    max_wr[ipa_i, ipb_i] = np.real(wwn[ipa_i, ipb_i, idx_max])

            ppa, ppb = np.meshgrid(pa, pb, indexing='ij')
            fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(14, 5))
            fig.subplots_adjust(bottom=0.12)

            c1 = ax_re.pcolormesh(rex * ppa, rey * ppb, rez * max_wr,
                                   shading='auto', cmap='RdBu_r')
            fig.colorbar(c1, ax=ax_re, label=f'{rez}*wr/wc1')
            ax_re.set_xlabel(f'{rex}*{strpa}')
            ax_re.set_ylabel(f'{rey}*{strpb}')
            ax_re.set_title(r'(a) $\omega_r$ of most unstable mode')

            c2p = ax_im.pcolormesh(rex * ppa, rey * ppb, rez * max_gamma,
                                    shading='auto', cmap='hot_r')
            fig.colorbar(c2p, ax=ax_im, label=f'{rez}*wi/wc1')
            ax_im.set_xlabel(f'{rex}*{strpa}')
            ax_im.set_ylabel(f'{rey}*{strpb}')
            ax_im.set_title(r'(b) max $\gamma / \omega_{c1}$')

            selector = _ModeSelector2D(fig, ax_re, ax_im, wwn,
                                        pa, pb, npa, npb, nw,
                                        rex, rey, rez)

            print("=" * 65)
            print("  2D Interactive Mode Selector")
            print("  Left-click: select | Right-click/r: undo | c: clear | a: auto "
                  "| t: toggle tracking | Enter: confirm | q: quit")
            print("=" * 65)

            plt.show(block=True)

            if selector.cancelled or len(selector.selected_points) == 0:
                print("No modes selected, exiting.")
                return None, None

            use_hungarian = selector.use_hungarian
            start_list = [(ipa_s, ipb_s, iww_s)
                            for ipa_s, ipb_s, iww_s, pa_s, pb_s, wn_s
                            in selector.selected_points]

            npl = len(selector.selected_points)
            wpdat = np.zeros((npl, 3), dtype=complex)
            for i, (ipa_s, ipb_s, iww_s, pa_s, pb_s, wn_s) in enumerate(
                    selector.selected_points):
                wpdat[i, 0] = pa_s
                wpdat[i, 1] = pb_s
                wpdat[i, 2] = wn_s

            print(f"\nSelected {npl} mode starting points:")
            for i in range(npl):
                print(f"  [{i+1}] pa={np.real(wpdat[i,0]):.4g}, "
                      f"pb={np.real(wpdat[i,1]):.4g}, "
                      f"omega/wc1 = {wpdat[i,2]:.6g}")

    # ==================================================================
    # Phase 2: Trace selected dispersion surfaces
    # ==================================================================
    wpdat = np.asarray(wpdat)
    npl = wpdat.shape[0]
    wws = np.zeros((npa, npb, npl), dtype=complex)

    method_name = "Hungarian (robust)" if use_hungarian else "Simple (basic)"
    print(f"\nTracing {npl} dispersion surfaces... Method: {method_name}")

    if ipa == ipb:
        # ---- 1D tracing ----
        # Build start_list: [(indpa, indww), ...]
        if start_list is None:
            start_list = []
            for jpl in range(npl):
                datstart = wpdat[jpl, :]
                indpa = np.argmin(np.abs(pa - np.real(datstart[0])))
                indww = np.argmin(np.abs(wwn_plot[indpa, 0, :] - datstart[2]))
                start_list.append((indpa, indww))

        if use_hungarian and npl > 1:
            # Simultaneous multi-mode Hungarian tracking
            ws_all = _trace_surfaces_1d_hungarian(
                        wwn_plot, pa, npa, start_list, pred_weight=pred_weight)
            for m in range(npl):
                wws[:, 0, m] = ws_all[:, m]
        else:
            # Single-mode tracing (simple or only 1 mode selected)
            for jpl in range(npl):
                indpa, indww = start_list[jpl]
                if use_hungarian:
                    # Even for 1 mode, Hungarian gives better prediction
                    ws_1 = _trace_surfaces_1d_hungarian(
                        wwn_plot, pa, npa, [(indpa, indww)],
                        pred_weight=pred_weight)
                    wws[:, 0, jpl] = ws_1[:, 0]
                else:
                    ws = _trace_surface_1d_simple(wwn_plot, pa, npa, indpa, indww)
                    wws[:, 0, jpl] = ws[:]

    else:
        # ---- 2D tracing ----
        start_list = []
        for jpl in range(npl):
            datstart = wpdat[jpl, :]
            indpa = np.argmin(np.abs(pa - np.real(datstart[0])))
            indpb = np.argmin(np.abs(pb - np.real(datstart[1])))
            indww = np.argmin(np.abs(wwn_plot[indpa, 0, :] - datstart[2]))
            start_list.append((indpa, indpb, indww))

        if use_hungarian and npl > 1:
            wws = _trace_surfaces_2d_hungarian(
                wwn_plot, pa, pb, npa, npb, start_list,
                pred_weight=pred_weight)
        else:
            for jpl in range(npl):
                indpa, indpb, indww = start_list[jpl]
                ws = _trace_surface_2d_simple(
                    wwn_plot, pa, pb, npa, npb, indpa, indpb, indww)
                wws[:, :, jpl] = ws[:, :]

    print("  Trace complete.")

    # ==================================================================
    # Phase 3: Plot results
    # ==================================================================
    ppa, ppb = np.meshgrid(pa, pb, indexing='ij')
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
    fig2.subplots_adjust(bottom=0.08)

    if ipa == ipb:
        ax_re2, ax_im2 = axes2
        x_vals = rex * pa if iloga == 0 else rex * 10.0**pa

        # Background scatter (gray)
        for iww in range(nw):
            color = get_cycle_color(iww)
            wr = rez * np.real(wwn_plot[:, 0, iww])
            wi = rez * np.imag(wwn_plot[:, 0, iww])

            ax_re2.plot(
                x_vals, wr,
                '.-', color=color, markersize=2.5, linewidth=0.7, alpha=0.45
            )
            ax_im2.plot(
                x_vals, wi,
                '.-', color=color, markersize=2.5, linewidth=0.7, alpha=0.45
            )

        # Overlay traced curves
        for jpl in range(npl):
            _, indww0 = start_list[jpl]
            color = get_cycle_color(indww0)
            if iloga == 0:
                ax_re2.plot(x_vals, rez * np.real(wws[:, 0, jpl]),
                            '--', color=color, linewidth=2.5,
                            label=f'Mode {jpl+1}')
                ax_im2.plot(x_vals, rez * np.imag(wws[:, 0, jpl]),
                            '--', color=color, linewidth=2.5,
                            label=f'Mode {jpl+1}')
            else:
                ax_re2.semilogx(x_vals, rez * np.real(wws[:, 0, jpl]),
                                '--', color=color, linewidth=2.5,
                                label=f'Mode {jpl+1}')
                ax_im2.semilogx(x_vals, rez * np.imag(wws[:, 0, jpl]),
                                '--', color=color, linewidth=2.5,
                                label=f'Mode {jpl+1}')

        # Mark starting points
        for jpl in range(npl):
            indpa_s = np.argmin(np.abs(pa - np.real(wpdat[jpl, 0])))
            x_s = x_vals[indpa_s]
            ax_re2.plot(x_s, rez * np.real(wws[indpa_s, 0, jpl]),
                        'o', color='red', markersize=8, markeredgewidth=2,
                        markerfacecolor='none', zorder=10)
            ax_im2.plot(x_s, rez * np.imag(wws[indpa_s, 0, jpl]),
                        'o', color='red', markersize=8, markeredgewidth=2,
                        markerfacecolor='none', zorder=10)

        ax_re2.set_xlabel(f'{rex}*{strpa}, npa={npa}')
        ax_re2.set_ylabel(
            f'{rez}*' + r'$\omega_r/\omega_{c1}$'
            + f', ' + r'$\alpha$' + f'={alphas[0]}')
        ax_re2.set_title(
            r'(a) $\beta_{||}$' + f'={betasz[0]:.3f}'
            + r', $\beta_\perp$' + f'={betasp[0]:.3f}')
        ax_re2.grid(True, alpha=0.3)
        ax_re2.legend(fontsize=8, loc='best')

        ax_im2.set_xlabel(f'{rex}*{strpa}, iem={iem}')
        ax_im2.set_ylabel(
            f'{rez}*' + r'$\omega_i/\omega_{c1}$'
            + f', ' + r'$\Delta$' + f'={Deltas}')
        ax_im2.set_title(
            f'(b) $v_A/c$={vA / np.sqrt(c2):.2f}, {strpb}='
            f'{par[ipbtmp]}, (S={S},N={N},J={J})')
        ax_im2.grid(True, alpha=0.3)
        ax_im2.legend(fontsize=8, loc='best')

        if iloga == 1:
            ax_re2.set_xscale('log')
            ax_im2.set_xscale('log')

    else:
        fig2.clf()
        for jpl in range(npl):
            wwjp = wws[:, :, jpl]
            ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
            ax1.plot_surface(rex * ppa, rey * ppb,
                             rez * np.real(wwjp), alpha=0.6, cmap='viridis')
            ax1.set_xlabel(f'{rex}*{strpa}')
            ax1.set_ylabel(f'{rey}*{strpb}')
            ax1.set_zlabel(f'{rez}*' + r'$\omega_r/\omega_{c1}$')
            ax1.set_title(
                r'(a) $\beta_{||}$' + f'={betasz[0]:.3f}'
                + r', $\beta_\perp$' + f'={betasp[0]:.3f}')

            ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
            ax2.plot_surface(rex * ppa, rey * ppb,
                             rez * np.imag(wwjp), alpha=0.6, cmap='hot_r')
            ax2.set_xlabel(f'{rex}*{strpa}')
            ax2.set_ylabel(f'{rey}*{strpb}')
            ax2.set_zlabel(f'{rez}*' + r'$\omega_i/\omega_{c1}$')
            ax2.set_title(f'(b) runtime={runtime:.1f}s')

    plt.tight_layout()

    os.makedirs(savepath, exist_ok=True)
    fig2.savefig(os.path.join(savepath, f'fig_pdrk_{figstr}_select.png'),
                 dpi=150)
    fig2.savefig(os.path.join(savepath, f'fig_pdrk_{figstr}_select.pdf'))
    print(f"\nImage saved to: {savepath}")

    plt.show()

    if run_pkues_output:
        if init is None:
            raise ValueError("run_pkues_output=True 时必须传入 init。")
        _run_pkues_output_from_selected_modes(
            init=init,
            wws=wws,
            pa=pa, pb=pb,
            npa=npa, npb=npb,
            ipa=ipa, ipb=ipb,
            iloga=iloga, ilogb=ilogb,
            strpa=strpa, strpb=strpb,
            savepath=savepath,
            figstr=figstr,
            betasz=betasz,
            betasp=betasp,
            start_list=start_list,
            idf=idf, jpa_df=jpa_df, jpb_df=jpb_df,
            jpl_df=jpl_df, s_df=s_df, vdf_config=vdf_config
        )

    return wws, wpdat


# ======================================================================
# Non-interactive batch mode
# ======================================================================
def pdrk_plot_select(wwn_plot, pa, pb, npa, npb, ipa, ipb, wpdat,
                     iloga=0, ilogb=0,
                     wcs1=1.0, strpa='pa', strpb='pb',
                     betasz=0.0, betasp=0.0,
                     alphas=1.0, Deltas=1.0,
                     vA=1.0, c2=(2.9979e8)**2,
                     S=1, N=1, J=8, iem=1,
                     par=None, ipbtmp=0,
                     runtime=0.0,
                     savepath='./', figstr='test',
                     rex=1.0, rey=1.0, rez=1.0,
                     use_hungarian=True, pred_weight=0.70,
                     init=None, run_pkues_output=False,
                     idf=0, jpa_df=0, jpb_df=0,
                     jpl_df=0, s_df=0, vdf_config=None):
    """
    Non-interactive batch mode: trace and plot given wpdat directly.

    Parameters
    ----------
    wwn : ndarray, shape (npa, npb, nw)
        Normalized eigenvalue solutions (omega / wcs1).
    wpdat : ndarray, shape (npl, 3)
        Starting points: [pa, pb, omega/wcs1].
    use_hungarian : bool
        True = use robust Hungarian tracking; False = simple PCHIP.
    pred_weight : float
        Prediction weight for Hungarian cost.

    Returns
    -------
    wws : ndarray, shape (npa, npb, npl)
    """
    npl = wpdat.shape[0]
    wws = np.zeros((npa, npb, npl), dtype=complex)

    if par is None:
        par = np.zeros(5)

    if ipa == ipb:
        # Build start list
        start_list = []
        for jpl in range(npl):
            datstart = wpdat[jpl, :]
            indpa = np.argmin(np.abs(pa - np.real(datstart[0])))
            indww = np.argmin(np.abs(wwn_plot[indpa, indpb, :] - datstart[2]))
            start_list.append((indpa, indww))

        if use_hungarian and npl > 1:
            ws_all = _trace_surfaces_1d_hungarian(
                wwn_plot, pa, npa, start_list, pred_weight=pred_weight)
            for m in range(npl):
                wws[:, 0, m] = ws_all[:, m]
        else:
            for jpl in range(npl):
                indpa, indww = start_list[jpl]
                if use_hungarian:
                    ws_1 = _trace_surfaces_1d_hungarian(
                        wwn_plot, pa, npa, [(indpa, indww)],
                        pred_weight=pred_weight)
                    wws[:, 0, jpl] = ws_1[:, 0]
                else:
                    ws = _trace_surface_1d_simple(wwn_plot, pa, npa, indpa, indww)
                    wws[:, 0, jpl] = ws[:]
    else:
        start_list = []
        for jpl in range(npl):
            datstart = wpdat[jpl, :]
            indpa = np.argmin(np.abs(pa - np.real(datstart[0])))
            indpb = np.argmin(np.abs(pb - np.real(datstart[1])))
            indww = np.argmin(np.abs(wwn_plot[indpa, 0, :] - datstart[2]))

        if use_hungarian and npl > 1:
            wws = _trace_surfaces_2d_hungarian(
                wwn_plot, pa, pb, npa, npb, start_list,
                pred_weight=pred_weight)
        else:
            for jpl in range(npl):
                indpa, indpb, indww = start_list[jpl]
                ws = _trace_surface_2d_simple(
                    wwn_plot, pa, pb, npa, npb, indpa, indpb, indww)
                wws[:, :, jpl] = ws[:, :]

    # ---- Plot ----
    ppa, ppb = np.meshgrid(pa, pb, indexing='ij')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for jpl in range(npl):
        _, indww0 = start_list[jpl]
        color = get_cycle_color(indww0)
        if ipa == ipb:
            ax1, ax2 = axes
            if iloga == 0:
                ax1.plot(rex * pa, rez * np.real(wws[:, 0, jpl]),
                         '--', color=color, linewidth=2)
                ax1.set_xlim([rex * np.min(pa), rex * np.max(pa)])
                ax2.plot(rex * pa, rez * np.imag(wws[:, 0, jpl]),
                         '--', color=color, linewidth=2)
                ax2.set_xlim([rex * np.min(pa), rex * np.max(pa)])
            else:
                ax1.semilogx(rex * 10.0**pa, rez * np.real(wws[:, 0, jpl]),
                             '--', color=color, linewidth=2)
                ax2.semilogx(rex * 10.0**pa, rez * np.imag(wws[:, 0, jpl]),
                             '--', color=color, linewidth=2)
            ax1.set_xlabel(f'{rex}*{strpa}, npa={npa}')
            ax1.set_ylabel(f'{rez}*' + r'$\omega_r/\omega_{c1}$'
                           + f', ' + r'$\alpha$' + f'={alphas}')
            ax1.set_title(r'(a) $\beta_{||}$' + f'={betasz[0]:.3g}'
                          + r', $\beta_\perp$' + f'={betasp[0]:.3g}')
            ax1.grid(True)
            ax2.set_xlabel(f'{rex}*{strpa}, iem={iem}')
            ax2.set_ylabel(f'{rez}*' + r'$\omega_i/\omega_{c1}$'
                           + f', ' + r'$\Delta$' + f'={Deltas}')
            ax2.set_title(
                f'(b) $v_A/c$={vA / np.sqrt(c2):.2g}, {strpb}='
                f'{par[ipbtmp]}, (S={S},N={N},J={J})')
            ax2.grid(True)
        else:
            fig.clf()
            wwjp = wws[:, :, jpl]
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.plot_surface(rex * ppa, rey * ppb,
                             rez * np.real(wwjp), alpha=0.7)
            ax1.set_xlabel(f'{rex}*{strpa}, ilogx={iloga}')
            ax1.set_ylabel(f'{rey}*{strpb}, ilogy={ilogb}')
            ax1.set_zlabel(f'{rez}*' + r'$\omega_r/\omega_{c1}$'
                           + f', npa={npa}, npb={npb}')
            ax1.set_title(r'(a) $\beta_{||}$' + f'={betasz[0]:.3g}'
                          + r', $\beta_\perp$' + f'={betasp[0]:.3g}')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.plot_surface(rex * ppa, rey * ppb,
                             rez * np.imag(wwjp), alpha=0.7)
            ax2.set_xlabel(f'{rex}*{strpa}')
            ax2.set_ylabel(f'{rey}*{strpb}')
            ax2.set_zlabel(f'{rez}*' + r'$\omega_i/\omega_{c1}$'
                           + f', S={S}, N={N}, J={J}')
            ax2.set_title(f'(b) runtime={runtime}s')

    plt.tight_layout()
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(os.path.join(savepath, f'fig_pdrk_{figstr}_select.png'),
                dpi=150)
    fig.savefig(os.path.join(savepath, f'fig_pdrk_{figstr}_select.pdf'))
    plt.show()

    if run_pkues_output:
        if init is None:
            raise ValueError("run_pkues_output=True 时必须传入 init。")
        _run_pkues_output_from_selected_modes(
            init=init,
            wws=wws,
            pa=pa, pb=pb,
            npa=npa, npb=npb,
            ipa=ipa, ipb=ipb,
            iloga=iloga, ilogb=ilogb,
            strpa=strpa, strpb=strpb,
            savepath=savepath,
            figstr=figstr,
            betasz=betasz,
            betasp=betasp,
            start_list=start_list
        )

    return wws


# ######################################################################
#  SECTION 5: DEMO
# ######################################################################
if __name__ == '__main__':
    print("""
==========================================================
  pdrk_plot_all.py — Interactive dispersion relation mode selector
  (Enhanced with Hungarian-algorithm robust tracking)
==========================================================

Usage:
------
  ## Interactive selection (recommended)
  from pdrk_plot_all import pdrk_plot_all_interactive
  wws, wpdat = pdrk_plot_all_interactive(ww, pa, pb, ...)

  ## Batch mode (wpdat known)
  from pdrk_plot_all import pdrk_plot_select
  wws = pdrk_plot_select(wwn, pa, pb, ..., wpdat=wpdat,
                          use_hungarian=True)

Tracking methods:
-----------------
  Simple:    Original PCHIP + nearest-neighbor (per mode, independent)
  Hungarian: Multi-point prediction + cost matrix + global optimal
             assignment via Hungarian algorithm (prevents mode
             misidentification at crossings)

==========================================================
""")

