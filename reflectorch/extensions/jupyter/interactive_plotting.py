"""
Interactive Plotting Support for Reflectorch Jupyter Widgets

This module provides enhanced plotting capabilities with interactive features:
- Figure persistence and in-place updates
- Cursor coordinate display
- Zoom and pan functionality
- Dynamic plot updates without figure recreation

The module is designed to be a drop-in replacement for static matplotlib plotting
while adding interactive capabilities for Jupyter environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.widgets import Cursor
from IPython.display import display, HTML
import ipywidgets as widgets
from typing import Optional, Tuple, Dict, Any, Union
import warnings
from contextlib import contextmanager

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None


@contextmanager
def suppress_log_scale_warnings():
    """Context manager to suppress matplotlib log scale warnings when no positive data"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                              message='Data has no positive values, and therefore cannot be log-scaled.',
                              category=UserWarning)
        yield


class InteractivePlotManager:
    """
    Manager for interactive matplotlib figures in Jupyter widgets
    
    This class handles the creation and management of persistent matplotlib figures
    that can be updated in-place without creating new figure windows.
    
    Features:
    - Persistent figure management
    - Interactive cursor with coordinate display
    - Zoom and pan capabilities
    - In-place plot updates
    - Automatic layout management
    """
    
    def __init__(self):
        self.figures = {}  # Store persistent figures
        self.axes = {}     # Store axes references
        self.cursors = {}  # Store cursor objects
        self.coord_displays = {}  # Store coordinate display widgets
        self._setup_interactive_backend()
    
    def _setup_interactive_backend(self):
        """Setup interactive matplotlib backend for Jupyter"""
        try:
            # Enable interactive matplotlib widget backend
            if get_ipython is not None:
                get_ipython().run_line_magic('matplotlib', 'widget')
            else:
                plt.ion()  # Turn on interactive mode
        except (NameError, AttributeError):
            # Not in IPython environment, use standard backend
            plt.ion()  # Turn on interactive mode
    
    def create_figure(self, 
                     figure_id: str, 
                     figsize: Tuple[float, float] = (12, 6),
                     plot_sld_profile: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create or retrieve a persistent figure with interactive features
        
        Args:
            figure_id: Unique identifier for the figure
            figsize: Figure size as (width, height)
            plot_sld_profile: Whether to create dual-axis layout for SLD profile
            
        Returns:
            Tuple of (figure, axes) where axes is single axis or tuple of axes
        """
        # Close existing figure if it exists
        if figure_id in self.figures:
            plt.close(self.figures[figure_id])
        
        # Create new figure
        if plot_sld_profile:
            fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=figsize)
            axes = (ax_r, ax_s)
        else:
            fig, ax_r = plt.subplots(1, 1, figsize=figsize)
            ax_s = None
            axes = ax_r
        
        # Store references
        self.figures[figure_id] = fig
        self.axes[figure_id] = axes
        
        # Setup interactive features
        self._setup_cursor(figure_id, ax_r)
        
        return fig, axes
    
    def _setup_cursor(self, figure_id: str, ax: plt.Axes):
        """Setup interactive cursor with coordinate display"""
        # Create coordinate display widget
        coord_display = widgets.HTML(
            value="<i>Move cursor over plot to see coordinates</i>",
            style={'description_width': 'initial'},
            layout={'margin': '5px 0px'}
        )
        
        # Store the display widget
        self.coord_displays[figure_id] = coord_display
        
        # Create cursor object
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1, alpha=0.7)
        self.cursors[figure_id] = cursor
        
        # Connect motion event for coordinate display
        def on_mouse_move(event):
            if event.inaxes == ax:
                if hasattr(ax, 'get_xscale') and ax.get_xscale() == 'log' and event.xdata and event.xdata > 0:
                    x_str = f"{event.xdata:.3e}"
                else:
                    x_str = f"{event.xdata:.4f}" if event.xdata else "N/A"
                
                if hasattr(ax, 'get_yscale') and ax.get_yscale() == 'log' and event.ydata and event.ydata > 0:
                    y_str = f"{event.ydata:.3e}"
                else:
                    y_str = f"{event.ydata:.4f}" if event.ydata else "N/A"
                
                coord_display.value = f"<b>Coordinates:</b> x = {x_str}, y = {y_str}"
            else:
                coord_display.value = "<i>Cursor outside plot area</i>"
        
        self.figures[figure_id].canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    def get_coordinate_display(self, figure_id: str) -> widgets.HTML:
        """Get the coordinate display widget for a figure"""
        return self.coord_displays.get(figure_id)
    
    def clear_and_update(self, figure_id: str, clear_sld: bool = True):
        """Clear axes content while preserving the figure structure and scales"""
        if figure_id not in self.axes:
            return
        
        axes = self.axes[figure_id]
        
        if isinstance(axes, tuple):
            ax_r, ax_s = axes
            
            # Store current scale settings before clearing
            r_xscale = ax_r.get_xscale()
            r_yscale = ax_r.get_yscale()
            
            ax_r.clear()
            
            # Restore scale settings (suppress warnings when no data)
            with suppress_log_scale_warnings():
                if r_xscale == 'log':
                    ax_r.set_xscale('log')
                if r_yscale == 'log':
                    ax_r.set_yscale('log')
            
            if clear_sld and ax_s is not None:
                s_xscale = ax_s.get_xscale()
                s_yscale = ax_s.get_yscale()
                
                ax_s.clear()
                
                # Restore SLD axis scales (suppress warnings when no data)
                with suppress_log_scale_warnings():
                    if s_xscale == 'log':
                        ax_s.set_xscale('log')
                    if s_yscale == 'log':
                        ax_s.set_yscale('log')
        else:
            # Store current scale settings
            xscale = axes.get_xscale()
            yscale = axes.get_yscale()
            
            axes.clear()
            
            # Restore scale settings (suppress warnings when no data)
            with suppress_log_scale_warnings():
                if xscale == 'log':
                    axes.set_xscale('log')
                if yscale == 'log':
                    axes.set_yscale('log')
    
    def get_figure_and_axes(self, figure_id: str) -> Tuple[plt.Figure, plt.Axes]:
        """Get existing figure and axes"""
        if figure_id not in self.figures:
            raise ValueError(f"Figure '{figure_id}' not found. Create it first with create_figure().")
        
        return self.figures[figure_id], self.axes[figure_id]
    
    def close_figure(self, figure_id: str):
        """Close and cleanup a figure"""
        if figure_id in self.figures:
            plt.close(self.figures[figure_id])
            del self.figures[figure_id]
            del self.axes[figure_id]
            
        if figure_id in self.cursors:
            del self.cursors[figure_id]
            
        if figure_id in self.coord_displays:
            del self.coord_displays[figure_id]


def plot_reflectivity_interactive(
    plot_manager: InteractivePlotManager,
    figure_id: str,
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
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Interactive version of plot_reflectivity that updates existing figures in-place
    
    This function is designed to be a drop-in replacement for the original plot_reflectivity
    function but with enhanced interactive capabilities and figure persistence.
    """
    
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
            return e
        return e[mask]

    # Convert inputs to numpy arrays
    q_exp, r_exp, yerr, xerr = _np(q_exp), _np(r_exp), _np(yerr), _np(xerr)
    q_pred, r_pred = _np(q_pred), _np(r_pred)
    q_pol, r_pol = _np(q_pol), _np(r_pol)
    z_sld, sld_pred, sld_pol = _np(z_sld), _np(sld_pred), _np(sld_pol)

    # Set default figure size
    if figsize is None:
        figsize = (12, 6) if plot_sld_profile else (6, 6)

    # Create or get existing figure
    try:
        fig, axes = plot_manager.get_figure_and_axes(figure_id)
        # Clear existing content
        plot_manager.clear_and_update(figure_id, clear_sld=plot_sld_profile)
    except ValueError:
        # Figure doesn't exist, create new one
        fig, axes = plot_manager.create_figure(figure_id, figsize, plot_sld_profile)

    # Extract axes
    if isinstance(axes, tuple):
        ax_r, ax_s = axes
    else:
        ax_r = axes
        ax_s = None

    # Apply x-limits
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

    # Set axis scales and labels (suppress warnings for log scale with no positive data)
    with suppress_log_scale_warnings():
        if logx: 
            ax_r.set_xscale('log')
        if logy:
            ax_r.set_yscale('log')

    ax_r.set_xlabel(q_label, fontsize=axis_label_size)
    ax_r.set_ylabel(r_label, fontsize=axis_label_size)
    ax_r.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_r.tick_params(axis='both', which='minor', labelsize=tick_label_size)
    
    if logx and x_ticks_log is not None:
        ax_r.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_log))
    if logy and y_ticks_log is not None:
        ax_r.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_log))

    handles = []

    # Plot experimental data
    exp_handle = None
    if q_exp is not None and r_exp is not None:
        mask_exp = _mask(q_exp, r_exp)
        q_exp_m, r_exp_m = q_exp[mask_exp], r_exp[mask_exp]
        yerr_m = _slice_sym_err(yerr, mask_exp)
        xerr_m = _slice_sym_err(xerr, mask_exp)

        has_yerr = yerr_m is not None
        has_xerr = xerr_m is not None
        
        if exp_style == 'auto':
            exp_style = 'errorbar' if (has_yerr or has_xerr) else 'scatter'

        if exp_style == 'errorbar':
            exp_handle = ax_r.errorbar(
                q_exp_m, r_exp_m, yerr=yerr_m, xerr=xerr_m,
                fmt=exp_marker, color=exp_color, markerfacecolor=exp_facecolor,
                markersize=exp_ms, alpha=exp_alpha, ecolor=exp_errcolor,
                elinewidth=exp_elinewidth, capsize=exp_capsize, capthick=exp_capthick,
                zorder=exp_zorder, label=exp_label
            )
        else:  # scatter
            exp_handle = ax_r.scatter(
                q_exp_m, r_exp_m, c=exp_color, marker=exp_marker, 
                s=exp_ms**2, alpha=exp_alpha, facecolors=exp_facecolor,
                zorder=exp_zorder, label=exp_label
            )
        handles.append(exp_handle)

    # Plot predicted curve
    if q_pred is not None and r_pred is not None:
        mask_pred = _mask(q_pred, r_pred)
        q_pred_m, r_pred_m = q_pred[mask_pred], r_pred[mask_pred]
        pred_handle = ax_r.plot(
            q_pred_m, r_pred_m, color=pred_color, lw=pred_lw, ls=pred_ls,
            alpha=pred_alpha, zorder=pred_zorder, label=pred_label
        )[0]
        handles.append(pred_handle)

    # Plot polished curve
    if q_pol is not None and r_pol is not None:
        mask_pol = _mask(q_pol, r_pol)
        q_pol_m, r_pol_m = q_pol[mask_pol], r_pol[mask_pol]
        pol_handle = ax_r.plot(
            q_pol_m, r_pol_m, color=pol_color, lw=pol_lw, ls=pol_ls,
            alpha=pol_alpha, zorder=pol_zorder, label=pol_label
        )[0]
        handles.append(pol_handle)

    # Plot SLD profile if requested
    if ax_s is not None and z_sld is not None:
        ax_s.set_xlabel(z_label, fontsize=axis_label_size)
        ax_s.set_ylabel(sld_label, fontsize=axis_label_size)
        ax_s.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax_s.tick_params(axis='both', which='minor', labelsize=tick_label_size)

        if sld_pred is not None:
            sld_pred_handle = ax_s.plot(
                z_sld, sld_pred, color=sld_pred_color, lw=sld_pred_lw, 
                ls=sld_pred_ls, label=sld_pred_label
            )[0]
            handles.append(sld_pred_handle)

        if sld_pol is not None:
            sld_pol_handle = ax_s.plot(
                z_sld, sld_pol, color=sld_pol_color, lw=sld_pol_lw, 
                ls=sld_pol_ls, label=sld_pol_label
            )[0]
            handles.append(sld_pol_handle)

        if legend:
            legend_kw = legend_kwargs or {}
            ax_s.legend(fontsize=legend_fontsize, **legend_kw)

    # Add legend to main plot
    if legend and handles:
        legend_kw = legend_kwargs or {}
        ax_r.legend(handles=handles, fontsize=legend_fontsize, **legend_kw)

    # Apply tight layout
    plt.tight_layout()
    
    # Force canvas draw for immediate update
    fig.canvas.draw()
    
    return (fig, (ax_r, ax_s)) if ax_s is not None else (fig, ax_r)


def plot_prediction_results_interactive(
    plot_manager: InteractivePlotManager,
    figure_id: str,
    prediction_dict: dict,
    q_exp: np.ndarray,
    curve_exp: np.ndarray,
    sigmas_exp: np.ndarray = None,
    logx=False,
):
    """
    Interactive version of plot_prediction_results that updates existing figures
    
    This is a convenience wrapper around plot_reflectivity_interactive that
    extracts data from the prediction dictionary.
    """
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
    sld_pol_c = prediction_dict.get('sld_profile_polished', None)

    plot_sld = (z_sld is not None) and (sld_pred_c is not None or sld_pol_c is not None)

    sld_is_complex = np.iscomplexobj(sld_pred_c)
    sld_pred_label = 'pred. SLD (Re)' if sld_is_complex else 'pred. SLD'
    sld_pol_label = 'polished SLD (Re)' if sld_is_complex else 'polished SLD'

    fig, axes = plot_reflectivity_interactive(
        plot_manager=plot_manager,
        figure_id=figure_id,
        q_exp=q_exp, r_exp=curve_exp, yerr=sigmas_exp,
        q_pred=q_pred, r_pred=r_pred,
        q_pol=q_pol, r_pol=r_pol,
        z_sld=z_sld,
        sld_pred=sld_pred_c.real if sld_pred_c is not None else None,
        sld_pol=sld_pol_c.real if sld_pol_c is not None else None,
        sld_pred_label=sld_pred_label,
        sld_pol_label=sld_pol_label,
        plot_sld_profile=plot_sld,
        logx=logx,
    )

    # Handle complex SLD plotting
    if sld_is_complex and plot_sld and isinstance(axes, tuple):
        ax_r, ax_s = axes
        ax_s.plot(z_sld, sld_pred_c.imag, color='darkgreen', lw=2.0, ls='-', zorder=4, label='pred. SLD (Im)')
        if sld_pol_c is not None:
            ax_s.plot(z_sld, sld_pol_c.imag, color='cyan', lw=2.0, ls='--', zorder=5, label='polished SLD (Im)')
        
        # Update legend
        ax_s.legend(fontsize=14)

    return fig, axes
