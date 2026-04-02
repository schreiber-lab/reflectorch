import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from typing import Optional, Tuple, Sequence


def print_prediction_results(prediction_dict, param_names=None, width=10, precision=3, header=True, print_err=False):

    if param_names is None:
        param_names = prediction_dict.get("param_names", [])

    pred = np.asarray(prediction_dict.get("predicted_params_array", []), dtype=float)
    pol = prediction_dict.get("polished_params_array", None)
    pol = np.asarray(pol, dtype=float) if pol is not None else None
    pol_err = prediction_dict.get('polished_params_error_array', None) if print_err else None
    pol_err = np.asarray(pol_err, dtype=float) if pol_err is not None else None

    name_w = max(14, max((len(str(n)) for n in param_names), default=14))

    num_fmt = f"{{:>{width}.{precision}f}}"

    if header:
        hdr = f"{'Parameter'.ljust(name_w)}  {'Predicted'.rjust(width)}"
        if pol is not None:
            hdr += f"  {'Polished'.rjust(width)}"
        if pol_err is not None:
            hdr += f"   {'Polished err'.rjust(width)}"
        print(hdr)
        print("-" * len(hdr))

    for i, name in enumerate(param_names):
        pred_val = pred[i] if i < pred.size else float("nan")
        row = f"{str(name).ljust(name_w)}  {num_fmt.format(pred_val)}"
        if pol is not None:
            pol_val = pol[i] if i < pol.size else float("nan")
            row += f"  {num_fmt.format(pol_val)}"
        if pol_err is not None:
            pol_err_val = pol_err[i] if i < pol_err.size else float('nan')
            row += f'   {num_fmt.format(pol_err_val)}'
        print(row)


def plot_prediction_results(
    prediction_dict: dict,
    q_exp: np.ndarray,
    curve_exp: np.ndarray,
    sigmas_exp: np.ndarray = None,
    logx=False,
):
    q_pred = prediction_dict['q_plot_pred']
    r_pred = prediction_dict['predicted_curve']
    r_pol = prediction_dict.get('polished_curve', None)

    q_pol = None
    if r_pol is not None:
        if len(r_pol) == len(q_pred):
            q_pol = q_pred
        elif len(r_pol) == len(q_exp):
            q_pol = q_exp

    z_sld = prediction_dict.get('predicted_sld_xaxis', None)
    sld_pred_c = prediction_dict.get('predicted_sld_profile', None)
    sld_pol_c  = prediction_dict.get('sld_profile_polished', None)

    plot_sld = (z_sld is not None) and (sld_pred_c is not None or sld_pol_c is not None)

    sld_is_complex = np.iscomplexobj(sld_pred_c)

    sld_pred_label = 'pred. SLD (Re)' if sld_is_complex else 'pred. SLD'
    sld_pol_label  = 'polished SLD (Re)' if sld_is_complex else 'polished SLD'

    fig, axes = plot_reflectivity(
        q_exp=q_exp, r_exp=curve_exp, yerr=sigmas_exp,
        q_pred=q_pred, r_pred=r_pred,
        q_pol=q_pol,  r_pol=r_pol,
        z_sld=z_sld,
        sld_pred=sld_pred_c.real if sld_pred_c is not None else None,
        sld_pol=sld_pol_c.real  if sld_pol_c  is not None else None,
        sld_pred_label=sld_pred_label,
        sld_pol_label=sld_pol_label,
        plot_sld_profile=plot_sld,
        logx=logx,
    )

    if sld_is_complex and plot_sld:
        ax_r, ax_s = axes
        ax_s.plot(z_sld, sld_pred_c.imag, color='darkgreen', lw=2.0, ls='-', zorder=4, label='pred. SLD (Im)')
        if sld_pol_c is not None:
            ax_s.plot(z_sld, sld_pol_c.imag, color='cyan', lw=2.0, ls='--', zorder=5, label='polished SLD (Im)')
        ax_s.legend(fontsize=14, frameon=True)

    return fig, axes


def plot_reflectivity( ### for back-compatibility with the documentation page from version 1.5 
    *,
    q_exp=None, 
    r_exp=None, 
    yerr=None, 
    xerr=None,
    q_pred=None, 
    r_pred=None,
    q_pol=None,  
    r_pol=None,
    z_sld=None, 
    sld_pred=None, 
    sld_pol=None,
    plot_sld_profile=False,
    figsize=None,
    logx=False,
    logy=True,
    x_ticks_log=None,
    y_ticks_log=(10.0 ** -np.arange(0, 12, 2)),
    q_label=r'q [$\mathrm{\AA^{-1}}$]',
    r_label='R(q)',
    z_label=r'z [$\mathrm{\AA}$]',
    sld_label=r'SLD [$10^{-6}\ \mathrm{\AA^{-2}}$]',
    xlim=None,
    axis_label_size=20,
    tick_label_size=15,
    legend_fontsize=14,
    exp_style='auto',
    exp_color='blue',
    exp_facecolor='none',
    exp_marker='o',
    exp_ms=3,
    exp_alpha=1.0,
    exp_errcolor='purple',
    exp_elinewidth=1.0,
    exp_capsize=1.0,
    exp_capthick=1.0,
    exp_zorder=2,
    pred_color='red',
    pred_lw=2.0,
    pred_ls='-',
    pred_alpha=1.0,
    pred_zorder=3,
    pol_color='orange',
    pol_lw=2.0,
    pol_ls='--',
    pol_alpha=1.0,
    pol_zorder=4,
    sld_pred_color='red',   
    sld_pred_lw=2.0,   
    sld_pred_ls='-',
    sld_pol_color='orange', 
    sld_pol_lw=2.0,    
    sld_pol_ls='--',
    exp_label='exp. data',
    pred_label='prediction',
    pol_label='polished prediction',
    sld_pred_label='pred. SLD',
    sld_pol_label='polished SLD',
    legend=True,
    legend_kwargs=None
):

    def _np(a):
        return None if a is None else np.asarray(a)

    def _mask(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if logx: m &= (x > 0.0)
        if logy: m &= (y > 0.0)
        return m

    def _slice_sym_err(err, mask):
        if err is None:
            return None
        if np.isscalar(err):
            return err
        e = np.asarray(err)
        if e.ndim != 1:
            raise ValueError("Errors must be scalar or 1-D array.")
        return e[mask]

    q_exp, r_exp, yerr, xerr = _np(q_exp), _np(r_exp), _np(yerr), _np(xerr)
    q_pred, r_pred = _np(q_pred), _np(r_pred)
    q_pol,  r_pol  = _np(q_pol),  _np(r_pol)
    z_sld, sld_pred, sld_pol = _np(z_sld), _np(sld_pred), _np(sld_pol)

    # Figure & axes
    if figsize is None:
        figsize = (12, 6) if plot_sld_profile else (6, 6)
    if plot_sld_profile:
        fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax_r = plt.subplots(1, 1, figsize=figsize)
        ax_s = None

    # Apply x-limits (right-only or both)
    if xlim is not None:
        if np.isscalar(xlim):
            cur_left, _ = ax_r.get_xlim()
            if logx and cur_left <= 0:
                cur_left = 1e-12
            ax_r.set_xlim(left=cur_left, right=float(xlim))
        else:
            xmin, xmax = xlim
            if logx and xmin is not None and xmin <= 0:
                raise ValueError("For log-x, xmin must be > 0.")
            ax_r.set_xlim(left=xmin, right=xmax)

    # Axis scales / labels / ticks
    if logx: ax_r.set_xscale('log')
    if logy: ax_r.set_yscale('log')

    ax_r.set_xlabel(q_label, fontsize=axis_label_size)
    ax_r.set_ylabel(r_label, fontsize=axis_label_size)
    ax_r.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_r.tick_params(axis='both', which='minor', labelsize=tick_label_size)
    if logx and x_ticks_log is not None:
        ax_r.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_log))
    if logy and y_ticks_log is not None:
        ax_r.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_log))

    handles = []

    # Experimental plot
    exp_handle = None
    if q_exp is not None and r_exp is not None:
        m = _mask(q_exp, r_exp)
        style = exp_style if exp_style != 'auto' else ('errorbar' if yerr is not None else 'scatter')

        if style == 'errorbar' and (yerr is not None):
            yerr_m = _slice_sym_err(yerr, m)
            xerr_m = _slice_sym_err(xerr, m)
            ax_r.errorbar(
                q_exp[m], r_exp[m], yerr=yerr_m, xerr=xerr_m,
                color=exp_color, ecolor=exp_errcolor,
                elinewidth=exp_elinewidth, capsize=exp_capsize,
                capthick=(exp_elinewidth if exp_capthick is None else exp_capthick),
                marker=exp_marker, linestyle='none', markersize=exp_ms,
                markerfacecolor=exp_facecolor, markeredgecolor=exp_color,
                alpha=exp_alpha, zorder=exp_zorder, label=None
            )
            exp_handle = Line2D([], [], color=exp_color, marker=exp_marker,
                                linestyle='none', markersize=exp_ms,
                                markerfacecolor=exp_facecolor, markeredgecolor=exp_color,
                                alpha=exp_alpha, label=exp_label)
        elif style == 'scatter':
            ax_r.scatter(
                q_exp[m], r_exp[m],
                s=exp_ms**2, marker=exp_marker,
                facecolors=exp_facecolor, edgecolors=exp_color,
                alpha=exp_alpha, zorder=exp_zorder, label=None
            )
            exp_handle = Line2D([], [], color=exp_color, marker=exp_marker,
                                linestyle='none', markersize=exp_ms,
                                markerfacecolor=exp_facecolor, markeredgecolor=exp_color,
                                alpha=exp_alpha, label=exp_label)
        else:  # 'line'
            ln = ax_r.plot(
                q_exp[m], r_exp[m], color=exp_color, lw=1.0, ls='-',
                alpha=exp_alpha, zorder=exp_zorder, label=exp_label
            )[0]
            exp_handle = ln

        if exp_handle is not None:
            handles.append(exp_handle)

    # Predicted line
    pred_handle = None
    if q_pred is not None and r_pred is not None:
        m = _mask(q_pred, r_pred)
        pred_handle = ax_r.plot(
            q_pred[m], r_pred[m],
            color=pred_color, lw=pred_lw, ls=pred_ls,
            alpha=pred_alpha, zorder=pred_zorder, label=pred_label
        )[0]
        handles.append(pred_handle)

    # Polished line
    pol_handle = None
    if q_pol is not None and r_pol is not None:
        m = _mask(q_pol, r_pol)
        pol_handle = ax_r.plot(
            q_pol[m], r_pol[m],
            color=pol_color, lw=pol_lw, ls=pol_ls,
            alpha=pol_alpha, zorder=pol_zorder, label=pol_label
        )[0]
        handles.append(pol_handle)

    if legend and handles:
        lk = {} if legend_kwargs is None else dict(legend_kwargs)
        ax_r.legend(handles=handles,
                    labels=[h.get_label() for h in handles],
                    fontsize=legend_fontsize, loc='best', **lk)

    # SLD panel (optional)
    if ax_s is not None:
        ax_s.set_xlabel(z_label, fontsize=axis_label_size)
        ax_s.set_ylabel(sld_label, fontsize=axis_label_size)
        ax_s.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax_s.tick_params(axis='both', which='minor', labelsize=tick_label_size)

        if z_sld is not None and sld_pred is not None:
            ax_s.plot(z_sld, sld_pred,
                      color=sld_pred_color, lw=sld_pred_lw, ls=sld_pred_ls,
                      label=sld_pred_label)
        if z_sld is not None and sld_pol is not None:
            ax_s.plot(z_sld, sld_pol,
                      color=sld_pol_color, lw=sld_pol_lw, ls=sld_pol_ls,
                      label=sld_pol_label)

        if legend:
            ax_s.legend(fontsize=legend_fontsize, loc='best', **(legend_kwargs or {}))

    plt.tight_layout()
    return (fig, (ax_r, ax_s)) if ax_s is not None else (fig, ax_r)


def plot_reflectivity_v2( #used in the InferenceResult class
    *,
    q_exp=None,
    r_exp=None,
    yerr=None,
    xerr=None,
    q_pred=None,
    r_pred=None,
    q_pol=None,
    r_pol=None,
    figsize=None,
    ax=None,
    logx=False,
    logy=True,
    x_ticks_log=None,
    y_ticks_log=(10.0 ** -np.arange(0, 12, 2)),
    q_label=r'q [$\mathrm{\AA^{-1}}$]',
    r_label='R(q)',
    xlim=None,
    axis_label_size=20,
    tick_label_size=15,
    legend_fontsize=14,
    show_grid=False,
    grid_alpha=0.25,
    tight_layout=True,

    exp_style='auto',
    exp_color='blue',
    exp_facecolor='none',
    exp_marker='o',
    exp_ms=3.5,
    exp_alpha=1.0,
    exp_errcolor=None,
    exp_elinewidth=0.8,
    exp_capsize=0.0,
    exp_capthick=0.8,
    exp_zorder=2,

    pred_color='red',
    pred_lw=1.5,
    pred_ls='-',
    pred_alpha=1.0,
    pred_zorder=3,

    pol_color='orange',
    pol_lw=2.0,
    pol_ls='--',
    pol_alpha=1.0,
    pol_zorder=4,

    exp_label='exp. data',
    pred_label='prediction',
    pol_label='polished prediction',

    legend=True,
    legend_kwargs=None
):
    def _np(a):
        return None if a is None else np.asarray(a)

    def _mask(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if logx:
            m &= (x > 0.0)
        if logy:
            m &= (y > 0.0)
        return m

    def _slice_sym_err(err, mask):
        if err is None:
            return None
        if np.isscalar(err):
            return err
        e = np.asarray(err)
        if e.ndim != 1:
            raise ValueError("Errors must be scalar or 1-D array.")
        return e[mask]

    q_exp, r_exp, yerr, xerr = _np(q_exp), _np(r_exp), _np(yerr), _np(xerr)
    q_pred, r_pred = _np(q_pred), _np(r_pred)
    q_pol, r_pol = _np(q_pol), _np(r_pol)

    if exp_errcolor is None:
        exp_errcolor = exp_color

    if ax is None:
        if figsize is None:
            figsize = (6, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    if xlim is not None:
        if np.isscalar(xlim):
            cur_left, _ = ax.get_xlim()
            if logx and cur_left <= 0:
                cur_left = 1e-12
            ax.set_xlim(left=cur_left, right=float(xlim))
        else:
            xmin, xmax = xlim
            if logx and xmin is not None and xmin <= 0:
                raise ValueError("For log-x, xmin must be > 0.")
            ax.set_xlim(left=xmin, right=xmax)

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(q_label, fontsize=axis_label_size)
    ax.set_ylabel(r_label, fontsize=axis_label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_size)

    if logx and x_ticks_log is not None:
        ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_log))
    if logy and y_ticks_log is not None:
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_log))

    if show_grid:
        ax.grid(True, alpha=grid_alpha)

    handles = []

    if q_exp is not None and r_exp is not None:
        m = _mask(q_exp, r_exp)

        style = exp_style if exp_style != 'auto' else (
            'errorbar' if (yerr is not None or xerr is not None) else 'scatter'
        )

        if style == 'errorbar' and (yerr is not None or xerr is not None):
            yerr_m = _slice_sym_err(yerr, m)
            xerr_m = _slice_sym_err(xerr, m)

            ax.errorbar(
                q_exp[m], r_exp[m],
                yerr=yerr_m, xerr=xerr_m,
                color=exp_color,
                ecolor=exp_errcolor,
                elinewidth=exp_elinewidth,
                capsize=exp_capsize,
                capthick=(exp_elinewidth if exp_capthick is None else exp_capthick),
                marker=exp_marker,
                linestyle='none',
                markersize=exp_ms,
                markerfacecolor=exp_facecolor,
                markeredgecolor=exp_color,
                alpha=exp_alpha,
                zorder=exp_zorder,
                label=None
            )
            exp_handle = Line2D(
                [], [],
                color=exp_color,
                marker=exp_marker,
                linestyle='none',
                markersize=exp_ms,
                markerfacecolor=exp_facecolor,
                markeredgecolor=exp_color,
                alpha=exp_alpha,
                label=exp_label
            )
            handles.append(exp_handle)

        elif style == 'scatter':
            ax.scatter(
                q_exp[m], r_exp[m],
                s=exp_ms**2,
                marker=exp_marker,
                facecolors=exp_facecolor,
                edgecolors=exp_color,
                alpha=exp_alpha,
                zorder=exp_zorder,
                label=None
            )
            exp_handle = Line2D(
                [], [],
                color=exp_color,
                marker=exp_marker,
                linestyle='none',
                markersize=exp_ms,
                markerfacecolor=exp_facecolor,
                markeredgecolor=exp_color,
                alpha=exp_alpha,
                label=exp_label
            )
            handles.append(exp_handle)

        else:
            exp_handle = ax.plot(
                q_exp[m], r_exp[m],
                color=exp_color,
                lw=1.0,
                ls='-',
                alpha=exp_alpha,
                zorder=exp_zorder,
                label=exp_label
            )[0]
            handles.append(exp_handle)

    if q_pred is not None and r_pred is not None:
        m = _mask(q_pred, r_pred)
        h = ax.plot(
            q_pred[m], r_pred[m],
            color=pred_color,
            lw=pred_lw,
            ls=pred_ls,
            alpha=pred_alpha,
            zorder=pred_zorder,
            label=pred_label
        )[0]
        handles.append(h)

    if q_pol is not None and r_pol is not None:
        m = _mask(q_pol, r_pol)
        h = ax.plot(
            q_pol[m], r_pol[m],
            color=pol_color,
            lw=pol_lw,
            ls=pol_ls,
            alpha=pol_alpha,
            zorder=pol_zorder,
            label=pol_label
        )[0]
        handles.append(h)

    if legend and handles:
        lk = {} if legend_kwargs is None else dict(legend_kwargs)
        ax.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            fontsize=legend_fontsize,
            loc='best',
            **lk
        )

    if tight_layout and created_fig:
        plt.tight_layout()

    return fig, ax


def plot_sld_profile(
    *,
    z_sld=None,
    sld_pred=None,
    sld_pol=None,
    figsize=(6, 6),
    ax=None,
    z_label=r'z [$\mathrm{\AA}$]',
    sld_label=r'SLD [$10^{-6}\ \mathrm{\AA^{-2}}$]',
    xlim=None,
    ylim=None,
    axis_label_size=20,
    tick_label_size=15,
    legend_fontsize=14,
    show_grid=False,
    grid_alpha=0.25,
    tight_layout=True,

    pred_color='red',
    pred_lw=2.0,
    pred_ls='-',
    pred_alpha=1.0,
    pred_zorder=2,

    pol_color='orange',
    pol_lw=2.0,
    pol_ls='--',
    pol_alpha=1.0,
    pol_zorder=3,

    pred_label=None,
    pol_label='polished',

    legend=True,
    legend_kwargs=None,
):
    def _np(a):
        return None if a is None else np.asarray(a)

    def _mask(x, y):
        return np.isfinite(x) & np.isfinite(y)

    z_sld = _np(z_sld)
    sld_pred = _np(sld_pred)
    sld_pol = _np(sld_pol)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    ax.set_xlabel(z_label, fontsize=axis_label_size)
    ax.set_ylabel(sld_label, fontsize=axis_label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_size)

    if xlim is not None:
        if np.isscalar(xlim):
            cur_left, _ = ax.get_xlim()
            ax.set_xlim(left=cur_left, right=float(xlim))
        else:
            xmin, xmax = xlim
            ax.set_xlim(left=xmin, right=xmax)

    if ylim is not None:
        if np.isscalar(ylim):
            cur_bottom, _ = ax.get_ylim()
            ax.set_ylim(bottom=cur_bottom, top=float(ylim))
        else:
            ymin, ymax = ylim
            ax.set_ylim(bottom=ymin, top=ymax)

    if show_grid:
        ax.grid(True, alpha=grid_alpha)

    handles = []

    if z_sld is not None and sld_pred is not None:
        m = _mask(z_sld, sld_pred)
        h = ax.plot(
            z_sld[m], sld_pred[m],
            color=pred_color,
            lw=pred_lw,
            ls=pred_ls,
            alpha=pred_alpha,
            zorder=pred_zorder,
            label=pred_label,
        )[0]
        if pred_label is not None:
            handles.append(h)

    if z_sld is not None and sld_pol is not None:
        m = _mask(z_sld, sld_pol)
        h = ax.plot(
            z_sld[m], sld_pol[m],
            color=pol_color,
            lw=pol_lw,
            ls=pol_ls,
            alpha=pol_alpha,
            zorder=pol_zorder,
            label=pol_label,
        )[0]
        if pol_label is not None:
            handles.append(h)

    if legend and handles:
        lk = {} if legend_kwargs is None else dict(legend_kwargs)
        ax.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            fontsize=legend_fontsize,
            loc='best',
            **lk,
        )

    if tight_layout and created_fig:
        plt.tight_layout()

    return fig, ax


def plot_reflectivity_multi(
    *,
    rq_series,
    sld_series=None,
    plot_sld_profile=False,
    figsize=None,
    logx=False,
    logy=True,
    xlim=None,
    x_ticks_log=None,
    y_ticks_log=(10.0 ** -np.arange(0, 12, 2)),
    q_label=r'q [$\mathrm{\AA^{-1}}$]',
    r_label='R(q)',
    z_label=r'z [$\mathrm{\AA}$]',
    sld_label=r'SLD [$10^{-6}\ \mathrm{\AA^{-2}}$]',
    axis_label_size=20,
    tick_label_size=15,
    legend=True,
    legend_fontsize=12,
    legend_kwargs=None,
):
    """
    Plot multiple R(q) series (and optional SLD lines) with per-series styling.

    rq_series: list of dicts, each with:
        required:
            - x: 1D array
            - y: 1D array
        optional (per series):
            - kind: 'errorbar' | 'scatter' | 'line'  (default 'line')
            - label: str
            - color: str
            - alpha: float (0..1)
            - zorder: int
            # scatter / marker for errorbar:
            - marker: str (default 'o')
            - ms: float (marker size in pt; for scatter internally converted to s=ms**2)
            - facecolor: str (scatter marker face)
            # errorbar only:
            - yerr: scalar or 1D array
            - xerr: scalar or 1D array
            - ecolor: str
            - elinewidth: float
            - capsize: float
            - capthick: float
            # line only:
            - lw: float
            - ls: str (e.g. '-', '--', ':')

    sld_series: list of dicts (only lines), each with:
        - x: 1D array (z-axis)
        - y: 1D array (SLD)
        - label: str (optional)
        - color: str (optional)
        - lw: float (optional)
        - ls: str (optional)
        - alpha: float (optional)
        - zorder: int (optional)
    """

    def _np(a): return None if a is None else np.asarray(a)

    def _mask(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if logx: m &= (x > 0.0)
        if logy: m &= (y > 0.0)
        return m

    # Figure & axes
    if figsize is None:
        figsize = (12, 6) if plot_sld_profile else (6, 6)
    if plot_sld_profile:
        fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax_r = plt.subplots(1, 1, figsize=figsize)
        ax_s = None

    # Axis scales / labels / ticks
    if logx: ax_r.set_xscale('log')
    if logy: ax_r.set_yscale('log')

    ax_r.set_xlabel(q_label, fontsize=axis_label_size)
    ax_r.set_ylabel(r_label, fontsize=axis_label_size)
    ax_r.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_r.tick_params(axis='both', which='minor', labelsize=tick_label_size)
    if logx and x_ticks_log is not None:
        ax_r.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_log))
    if logy and y_ticks_log is not None:
        ax_r.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_log))

    # Apply x-limits (right-only or both)
    if xlim is not None:
        if np.isscalar(xlim):
            left, _ = ax_r.get_xlim()
            if logx and left <= 0:
                left = 1e-12
            ax_r.set_xlim(left=left, right=float(xlim))
        else:
            xmin, xmax = xlim
            if logx and xmin is not None and xmin <= 0:
                raise ValueError("For log-x, xmin must be > 0.")
            ax_r.set_xlim(left=xmin, right=xmax)

    # Plot all R(q) series in the given order (order = legend order)
    handles = []
    for s in rq_series:
        kind  = s.get('kind', 'line')
        x     = _np(s.get('x'))
        y     = _np(s.get('y'))
        if x is None or y is None:
            continue

        label = s.get('label', None)
        color = s.get('color', None)
        alpha = s.get('alpha', 1.0)
        zord  = s.get('zorder', None)
        ms    = s.get('ms', 5.0)
        marker = s.get('marker', 'o')

        m = _mask(x, y)

        if kind == 'errorbar':
            yerr = s.get('yerr', None)
            xerr = s.get('xerr', None)
            ecolor     = s.get('ecolor', color)
            elinewidth = s.get('elinewidth', 1.0)
            capsize    = s.get('capsize', 0.0)
            capthick   = s.get('capthick', elinewidth)

            # Symmetric error input: scalar or 1D
            def _slice_sym(err):
                if err is None: return None
                if np.isscalar(err): return err
                arr = np.asarray(err)
                if arr.ndim != 1:
                    raise ValueError("For symmetric error bars, provide scalar or 1-D array.")
                return arr[m]

            yerr_m = _slice_sym(yerr)
            xerr_m = _slice_sym(xerr)

            ax_r.errorbar(
                x[m], y[m], yerr=yerr_m, xerr=xerr_m,
                color=color, ecolor=ecolor,
                elinewidth=elinewidth, capsize=capsize, capthick=capthick,
                marker=marker, linestyle='none', markersize=ms,
                markerfacecolor=s.get('facecolor', 'none'),
                markeredgecolor=color,
                alpha=alpha, zorder=zord, label=None
            )

            h = Line2D([], [], color=color, marker=marker, linestyle='none',
                       markersize=ms, markerfacecolor=s.get('facecolor','none'),
                       markeredgecolor=color, alpha=alpha, label=label)
            if label is not None:
                handles.append(h)

        elif kind == 'scatter':
            facecolor = s.get('facecolor', 'none')
            sc = ax_r.scatter(
                x[m], y[m], s=ms**2, marker=marker,
                facecolors=facecolor, edgecolors=color,
                alpha=alpha, zorder=zord, label=None
            )
            h = Line2D([], [], color=color, marker=marker, linestyle='none',
                       markersize=ms, markerfacecolor=facecolor,
                       markeredgecolor=color, alpha=alpha, label=label)
            if label is not None:
                handles.append(h)

        else:  # 'line'
            lw = s.get('lw', 2.0)
            ls = s.get('ls', '-')
            line = ax_r.plot(
                x[m], y[m],
                color=color, lw=lw, ls=ls,
                alpha=alpha, zorder=zord, label=label
            )[0]
            if label is not None:
                handles.append(line)

    if legend and handles:
        lk = {} if legend_kwargs is None else dict(legend_kwargs)
        ax_r.legend(handles=handles,
                    labels=[h.get_label() for h in handles],
                    fontsize=legend_fontsize, loc='best', **lk)

    # Optional SLD panel
    if plot_sld_profile:
        ax_s.set_xlabel(z_label, fontsize=axis_label_size)
        ax_s.set_ylabel(sld_label, fontsize=axis_label_size)
        ax_s.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax_s.tick_params(axis='both', which='minor', labelsize=tick_label_size)

        if sld_series:
            for s in sld_series:
                zx = _np(s.get('x')); zy = _np(s.get('y'))
                if zx is None or zy is None:
                    continue
                label = s.get('label', None)
                color = s.get('color', None)
                lw    = s.get('lw', 2.0)
                ls    = s.get('ls', '-')
                alpha = s.get('alpha', 1.0)
                zord  = s.get('zorder', None)
                ax_s.plot(zx, zy, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zord, label=label)

            if legend:
                ax_s.legend(fontsize=legend_fontsize, loc='best', **(legend_kwargs or {}))

    plt.tight_layout()
    return (fig, (ax_r, ax_s)) if plot_sld_profile else (fig, ax_r)

def plot_sampled_reflectivity_curves(
    q_exp,
    curve_exp,
    q_sampled,
    sampled_curves,
    yerr=None,
    xerr=None,
    highlight_indices=None,
    highlight_colors=None,
    highlight_labels=None,
    highlight_lw=1.0,
    highlight_alpha=1.0,

    exp_style='auto',
    exp_label='exp. data',
    exp_color='blue',
    exp_marker='o',
    exp_facecolor='none',
    exp_ms=5.0,
    exp_alpha=1.0,
    exp_errcolor=None,
    exp_elinewidth=0.8,
    exp_capsize=0.0,
    exp_capthick=0.8,
    exp_zorder=3,

    sample_color='lightgreen',
    sample_lw=1.0,
    sample_alpha_scale=120,
    sample_zorder=1,
    proxy_label='sampled curves',

    q_label=r'q [$\mathrm{\AA^{-1}}$]',
    r_label='R(q)',
    logy=True,
    logx=False,
    xlim=None,
    ylim=None,
    legend=True,
    legend_fontsize=14,
    legend_loc='best',
    axis_label_size=18,
    tick_label_size=14,
    figsize=(8, 6),
    tight_layout=True,
):
    fig, ax = plt.subplots(figsize=figsize)

    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    ax.set_xlabel(q_label, fontsize=axis_label_size)
    ax.set_ylabel(r_label, fontsize=axis_label_size)
    ax.tick_params(axis='both', which='both', labelsize=tick_label_size)

    if exp_errcolor is None:
        exp_errcolor = exp_color

    style = exp_style if exp_style != 'auto' else (
        'errorbar' if (yerr is not None or xerr is not None) else 'scatter'
    )

    if style == 'errorbar' and (yerr is not None or xerr is not None):
        exp_handle = ax.errorbar(
            q_exp,
            curve_exp + 1e-10,
            yerr=yerr,
            xerr=xerr,
            fmt=exp_marker,
            label=exp_label,
            color=exp_color,
            ecolor=exp_errcolor,
            elinewidth=exp_elinewidth,
            capsize=exp_capsize,
            capthick=exp_capthick,
            markersize=exp_ms,
            markerfacecolor=exp_facecolor,
            alpha=exp_alpha,
            zorder=exp_zorder,
        )
    elif style == 'scatter':
        ax.scatter(
            q_exp,
            curve_exp + 1e-10,
            s=exp_ms**2,
            marker=exp_marker,
            facecolors=exp_facecolor,
            edgecolors=exp_color,
            alpha=exp_alpha,
            zorder=exp_zorder,
            label=None,
        )
        exp_handle = Line2D(
            [0], [0],
            color=exp_color,
            marker=exp_marker,
            linestyle='none',
            markersize=exp_ms,
            markerfacecolor=exp_facecolor,
            markeredgecolor=exp_color,
            alpha=exp_alpha,
            label=exp_label,
        )
    else:
        exp_handle = ax.plot(
            q_exp,
            curve_exp + 1e-10,
            color=exp_color,
            lw=1.0,
            ls='-',
            alpha=exp_alpha,
            zorder=exp_zorder,
            label=exp_label,
        )[0]

    q_np = q_sampled.cpu().numpy() if hasattr(q_sampled, 'cpu') else q_sampled
    sampled_np = sampled_curves.cpu().numpy() if hasattr(sampled_curves, 'cpu') else sampled_curves

    segments = [np.column_stack((q_np, y)) for y in sampled_np]
    alpha = min(sample_alpha_scale / max(len(sampled_np), 1), 1.0)
    line_collection = LineCollection(
        segments,
        colors=sample_color,
        linewidths=sample_lw,
        alpha=alpha,
        zorder=sample_zorder
    )
    ax.add_collection(line_collection)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.autoscale(enable=True, axis='x')

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.autoscale(enable=True, axis='y')

    highlight_handles = []
    highlight_labels_used = []
    if highlight_indices:
        for i, idx in enumerate(highlight_indices):
            color = (highlight_colors or ['red'] * len(highlight_indices))[i]
            label = (highlight_labels or [None] * len(highlight_indices))[i]
            line, = ax.plot(
                q_np, sampled_np[idx],
                color=color,
                lw=highlight_lw,
                alpha=highlight_alpha,
                zorder=exp_zorder + 1,
                label=label
            )
            if label:
                highlight_handles.append(line)
                highlight_labels_used.append(label)

    if legend:
        handles = [exp_handle]
        labels = [exp_label]

        proxy_line = Line2D(
            [0], [0],
            color=sample_color,
            lw=max(sample_lw, 2.0),
            alpha=1.0,
            label=proxy_label
        )
        handles.append(proxy_line)
        labels.append(proxy_label)

        if highlight_handles:
            handles.extend(highlight_handles)
            labels.extend(highlight_labels_used)

        ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=legend_fontsize)

    if tight_layout:
        plt.tight_layout()

    return fig, ax


def plot_sampled_sld_profiles(
    z_sld,
    sampled_slds,
    highlight_indices=None,
    highlight_colors=None,
    highlight_lw=1.0,
    highlight_alpha=1.0,
    highlight_labels=None,
    z_label=r'z [$\mathrm{\AA}$]',
    sld_label=r'SLD [$10^{-6}\ \mathrm{\AA^{-2}}$]',
    sample_color='lightgreen',
    sample_lw=1.0,
    sample_alpha_scale=50,
    sample_zorder=1,
    proxy_label='sampled SLDs',
    legend=True,
    legend_fontsize=14,
    legend_loc='best',
    axis_label_size=18,
    tick_label_size=14,
    figsize=(8, 6),
    tight_layout=True,
    color_plot=False,
    cmap='jet',
    color_plot_alpha_scale=200,
    colorbar=False,
    colorbar_label=None,
    sort_mode='pca',
):
    """
    Plot sampled SLD profiles.

    If color_plot=False, all profiles are drawn in the same color with alpha scaling.

    If color_plot=True, profiles are colored according to their position in an ordering.
    By default this ordering is based on a whole-profile PCA score, so similar profiles
    tend to have similar colors and different profiles tend to have different colors.

    sort_mode options:
        - "pca": sort whole profiles by first principal-component score
        - "mean": sort whole profiles by mean SLD value
        - "none": keep original sample order
        - "pointwise": sort independently at each z point
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel(z_label, fontsize=axis_label_size)
    ax.set_ylabel(sld_label, fontsize=axis_label_size)
    ax.tick_params(axis='both', which='both', labelsize=tick_label_size)

    z_np = z_sld.cpu().numpy() if hasattr(z_sld, 'cpu') else np.asarray(z_sld)
    sld_np = sampled_slds.cpu().numpy() if hasattr(sampled_slds, 'cpu') else np.asarray(sampled_slds)

    if sld_np.ndim != 2:
        raise ValueError(f"`sampled_slds` must have shape (N, n_z), got shape {sld_np.shape}")
    if z_np.ndim != 1:
        raise ValueError(f"`z_sld` must be 1-D, got shape {z_np.shape}")
    if sld_np.shape[1] != z_np.shape[0]:
        raise ValueError(
            f"Incompatible shapes: sampled_slds has shape {sld_np.shape} but z_sld has shape {z_np.shape}"
        )

    num_profiles = sld_np.shape[0]

    sort_mode = str(sort_mode).lower()

    if sort_mode == 'pca':
        X = sld_np - sld_np.mean(axis=0, keepdims=True)
        if num_profiles <= 1 or np.allclose(X, 0.0):
            order = np.arange(num_profiles)
        else:
            try:
                _, _, vt = np.linalg.svd(X, full_matrices=False)
                scores = X @ vt[0]
                order = np.argsort(scores)
            except np.linalg.LinAlgError:
                scores = X.mean(axis=1)
                order = np.argsort(scores)

        ordered_profiles = sld_np[order]

    elif sort_mode == 'mean':
        scores = sld_np.mean(axis=1)
        order = np.argsort(scores)
        ordered_profiles = sld_np[order]

    elif sort_mode == 'none':
        order = np.arange(num_profiles)
        ordered_profiles = sld_np

    elif sort_mode == 'pointwise':
        order = None
        ordered_profiles = np.sort(sld_np, axis=0)

    else:
        raise ValueError("sort_mode must be one of: 'pca', 'mean', 'none', 'pointwise'")

    segments = [np.column_stack((z_np, profile)) for profile in ordered_profiles]

    if not color_plot:
        alpha = min(sample_alpha_scale / max(num_profiles, 1), 1.0)
        collection = LineCollection(
            segments,
            colors=sample_color,
            linewidths=sample_lw,
            alpha=alpha,
            zorder=sample_zorder
        )
        ax.add_collection(collection)

    else:
        cmap_obj = plt.get_cmap(cmap)
        base_colors = cmap_obj(np.linspace(0, 1, num_profiles))
        per_line_alpha = min(color_plot_alpha_scale / max(num_profiles, 1), 1.0)
        base_colors[:, -1] = per_line_alpha

        collection = LineCollection(
            segments,
            colors=base_colors,
            linewidths=sample_lw,
            zorder=sample_zorder
        )
        ax.add_collection(collection)

        if colorbar:
            norm = Normalize(vmin=0, vmax=max(num_profiles - 1, 1))
            sm = ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

    ax.autoscale()

    if highlight_indices:
        if sort_mode == 'pointwise':
            for i, idx in enumerate(highlight_indices):
                label = highlight_labels[i] if highlight_labels and i < len(highlight_labels) else None
                color = highlight_colors[i] if highlight_colors and i < len(highlight_colors) else None
                ax.plot(
                    z_np,
                    sld_np[idx],
                    color=color,
                    lw=highlight_lw,
                    ls='--',
                    alpha=highlight_alpha,
                    zorder=sample_zorder + 1,
                    label=label
                )
        else:
            for i, idx in enumerate(highlight_indices):
                label = highlight_labels[i] if highlight_labels and i < len(highlight_labels) else None
                color = highlight_colors[i] if highlight_colors and i < len(highlight_colors) else None
                ax.plot(
                    z_np,
                    sld_np[idx],
                    color=color,
                    lw=highlight_lw,
                    ls='--',
                    alpha=highlight_alpha,
                    zorder=sample_zorder + 1,
                    label=label
                )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        proxy_line = plt.Line2D(
            [0], [0],
            color=sample_color if not color_plot else 'gray',
            lw=sample_lw,
            alpha=1.0,
            label=proxy_label
        )
        handles.insert(0, proxy_line)
        labels.insert(0, proxy_label)
        ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=legend_fontsize)

    if tight_layout:
        plt.tight_layout()

    return fig, ax


def plot_sampled_profiles_multi_type(
    z_axis: np.ndarray,
    sampled_profiles_list: Sequence[np.ndarray],
    profile_types: Sequence[str],
    *,
    profile_labels: Optional[Sequence[str]] = None,
    profile_colors: Optional[Sequence[Optional[str]]] = None,
    use_twin_axis: bool = False,
    twin_axis_index: int = 1,
    sample_lw: float = 1.0,
    sample_alpha_scale: float = 50.0,
    sample_zorder: int = 1,
    z_label: str = r"z [$\mathrm{\AA}$]",
    ylabel_left: Optional[str] = None,
    ylabel_right: Optional[str] = None,
    axis_label_size: int = 18,
    tick_label_size: int = 14,
    legend: bool = True,
    legend_fontsize: int = 14,
    legend_loc: str = "best",
    figsize: Tuple[float, float] = (8, 6),
    tight_layout: bool = True,
):
    """
    Plot several sampled profile ensembles on the same figure.

    Args:
        z_axis:
            Shared depth axis of shape ``(n_z,)``.
        sampled_profiles_list:
            Sequence of sampled profile arrays. Each entry must have shape
            ``(n_samples, n_z)``.
        profile_types:
            Names of the plotted profile types.
        profile_labels:
            Optional display labels for the legend. If omitted, ``profile_types``
            are used.
        profile_colors:
            Optional colors, one per profile type.
        use_twin_axis:
            If ``True`` and exactly two profile types are plotted, draw the
            second profile type on a twin y-axis on the right.
        twin_axis_index:
            Index of the profile type that should use the right-hand axis when
            ``use_twin_axis=True``.
        sample_lw:
            Line width used for sampled profiles.
        sample_alpha_scale:
            Global alpha scaling factor. Effective alpha for each ensemble is
            ``min(sample_alpha_scale / n_samples, 1.0)``.
        sample_zorder:
            Z-order of the line collections.
        z_label:
            Label of the horizontal axis.
        ylabel_left:
            Optional custom label for the left y-axis.
        ylabel_right:
            Optional custom label for the right y-axis when using a twin axis.
        axis_label_size:
            Axis-label font size.
        tick_label_size:
            Tick-label font size.
        legend:
            Whether to draw a legend.
        legend_fontsize:
            Legend font size.
        legend_loc:
            Legend location.
        figsize:
            Matplotlib figure size.
        tight_layout:
            Whether to call ``plt.tight_layout()``.

    Returns:
        If ``use_twin_axis=False``, returns ``(fig, ax)``.
        If ``use_twin_axis=True``, returns ``(fig, ax_left, ax_right)``.
    """
    z_axis = np.asarray(z_axis)
    sampled_profiles_list = [np.asarray(p) for p in sampled_profiles_list]
    profile_types = list(profile_types)

    n_types = len(profile_types)
    if n_types == 0:
        raise ValueError("Need at least one profile type.")
    if len(sampled_profiles_list) != n_types:
        raise ValueError("sampled_profiles_list and profile_types must have the same length.")

    for i, prof in enumerate(sampled_profiles_list):
        if prof.ndim != 2:
            raise ValueError(
                f"Profile array at index {i} must have shape (n_samples, n_z), got {prof.shape}."
            )
        if prof.shape[1] != z_axis.shape[0]:
            raise ValueError(
                f"Profile array at index {i} has incompatible z dimension: "
                f"{prof.shape[1]} vs {z_axis.shape[0]}."
            )

    if profile_labels is None:
        profile_labels = list(profile_types)
    else:
        profile_labels = list(profile_labels)
        if len(profile_labels) != n_types:
            raise ValueError(f"Expected {n_types} profile_labels, got {len(profile_labels)}.")

    if profile_colors is None:
        profile_colors = [None] * n_types
    else:
        profile_colors = list(profile_colors)
        if len(profile_colors) != n_types:
            raise ValueError(f"Expected {n_types} profile_colors, got {len(profile_colors)}.")

    if use_twin_axis:
        if n_types != 2:
            raise ValueError("use_twin_axis=True is only supported when exactly two profile types are plotted.")
        if twin_axis_index not in (0, 1):
            raise ValueError("twin_axis_index must be 0 or 1 when plotting two profile types.")

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx() if use_twin_axis else None

    legend_handles = []

    for i, (ptype, prof, label, color) in enumerate(
        zip(profile_types, sampled_profiles_list, profile_labels, profile_colors)
    ):
        ax = ax_right if (use_twin_axis and i == twin_axis_index) else ax_left

        alpha = min(sample_alpha_scale / max(prof.shape[0], 1), 1.0)
        segments = [np.column_stack((z_axis, row)) for row in prof]

        lc = LineCollection(
            segments,
            colors=color,
            linewidths=sample_lw,
            alpha=alpha,
            zorder=sample_zorder,
        )
        ax.add_collection(lc)

        proxy = Line2D(
            [0], [0],
            color=color,
            lw=max(sample_lw, 2.0),
            label=label,
        )
        legend_handles.append(proxy)

    ax_left.set_xlabel(z_label, fontsize=axis_label_size)
    ax_left.tick_params(axis="both", which="both", labelsize=tick_label_size)
    ax_left.autoscale()

    if use_twin_axis:
        ax_right.tick_params(axis="y", which="both", labelsize=tick_label_size)
        ax_right.autoscale()
        if ylabel_left is not None:
            ax_left.set_ylabel(ylabel_left, fontsize=axis_label_size)
        if ylabel_right is not None:
            ax_right.set_ylabel(ylabel_right, fontsize=axis_label_size)
    else:
        if ylabel_left is not None:
            ax_left.set_ylabel(ylabel_left, fontsize=axis_label_size)

    if legend:
        ax_left.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc=legend_loc,
            fontsize=legend_fontsize,
        )

    if tight_layout:
        plt.tight_layout()

    if use_twin_axis:
        return fig, ax_left, ax_right
    return fig, ax_left