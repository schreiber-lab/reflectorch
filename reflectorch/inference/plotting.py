import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def print_prediction_results(prediction_dict, param_names=None, width=10, precision=3, header=True):

    if param_names is None:
        param_names = prediction_dict.get("param_names", [])

    pred = np.asarray(prediction_dict.get("predicted_params_array", []), dtype=float)
    pol = prediction_dict.get("polished_params_array", None)
    pol = np.asarray(pol, dtype=float) if pol is not None else None
    pol_err = prediction_dict.get('polished_params_error_array', None)
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


def plot_reflectivity(
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