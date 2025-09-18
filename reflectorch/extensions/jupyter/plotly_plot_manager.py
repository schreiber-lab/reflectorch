import numpy as np

import plotly.graph_objects as go


class PlotlyPlotManager:
    """
    Manager for Plotly figures in Jupyter widgets
    """
    
    def __init__(self, verbose: bool = False):
        self.figures = {}  # Store persistent figures
        self.widgets = {}  # Store plotly widgets
        self.verbose = verbose
            
    def create_reflectivity_figure(self, 
                                  figure_id: str,
                                  width: int = 600,
                                  height: int = 300):
        """Create a reflectivity-only figure widget"""
        fig = go.Figure()
        
        fig.update_layout(
            width=width,
            height=height,
            showlegend=True,
            hovermode='closest',
            template='plotly_white',
            margin=dict(l=60, r=20, t=60, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )
        
        # Create Plotly widget
        plotly_widget = go.FigureWidget(fig)
        
        # Store references
        self.figures[figure_id] = fig
        self.widgets[figure_id] = plotly_widget
        
        return plotly_widget
    
    def create_sld_figure(self, 
                         figure_id: str,
                         width: int = 600,
                         height: int = 250):
        """Create an SLD-only figure widget"""
        fig = go.Figure()
        
        fig.update_layout(
            width=width,
            height=height,
            showlegend=True,
            hovermode='closest',
            template='plotly_white',
            margin=dict(l=60, r=20, t=60, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )
        
        # Create Plotly widget
        plotly_widget = go.FigureWidget(fig)
        
        # Store references
        self.figures[figure_id] = fig
        self.widgets[figure_id] = plotly_widget
        
        return plotly_widget
    
    def _setup_figure_hover(self, figure_id: str, plotly_widget):
        """Setup hover functionality for the figure (no coordinate display)"""
        # Just ensure data is mutable for future trace additions
        plotly_widget.data = list(plotly_widget.data)
    
    def get_figure(self, figure_id: str):
        """Get existing figure widget"""
        if figure_id not in self.widgets:
            raise ValueError(f"Figure '{figure_id}' not found. Create it first with create_figure().")
        return self.widgets[figure_id]
    
    def get_widget(self, figure_id: str):
        """Get the plotly widget for display"""
        if figure_id not in self.widgets:
            raise ValueError(f"Widget for figure '{figure_id}' not found.")
        return self.widgets[figure_id]
    
    
    def clear_figure(self, figure_id: str):
        """Clear all traces from the figure"""
        if figure_id in self.widgets:
            widget = self.widgets[figure_id]
            # Use Plotly's proper method to clear traces
            with widget.batch_update():
                widget.data = []
    
    def close_figure(self, figure_id: str):
        """Close and cleanup a figure"""
        if figure_id in self.figures:
            del self.figures[figure_id]
            
        if figure_id in self.widgets:
            del self.widgets[figure_id]
            


def plot_reflectivity_only(
    plot_manager: PlotlyPlotManager,
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
    logx=False,
    logy=True,
    exp_color='blue',
    exp_errcolor='purple',
    pred_color='red',
    pol_color='orange',
    exp_label='experimental data',
    pred_label='prediction',
    pol_label='polished prediction',
    width=600,
    height=300
):
    """
    Plot reflectivity data only using Plotly
    
    This function creates or updates a Plotly figure widget with reflectivity data only.
    """
    
    def _np(a): 
        return None if a is None else np.asarray(a)

    def _mask_data(x, y):
        """Create mask for finite values and positive values if log scale"""
        if x is None or y is None:
            return None, None
        
        x, y = np.asarray(x), np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        
        if logx: 
            mask &= (x > 0.0)
        if logy: 
            mask &= (y > 0.0)
            
        return x[mask], y[mask]

    # Convert inputs to numpy arrays
    q_exp, r_exp, yerr, xerr = _np(q_exp), _np(r_exp), _np(yerr), _np(xerr)
    q_pred, r_pred = _np(q_pred), _np(r_pred)
    q_pol, r_pol = _np(q_pol), _np(r_pol)

    # Create or get existing figure widget
    try:
        fig = plot_manager.get_figure(figure_id)
        # Clear existing traces
        plot_manager.clear_figure(figure_id)
    except ValueError:
        # Figure doesn't exist, create new one
        fig = plot_manager.create_reflectivity_figure(figure_id, width, height)

    # Plot experimental data
    if q_exp is not None and r_exp is not None:
        q_exp_clean, r_exp_clean = _mask_data(q_exp, r_exp)
        
        if q_exp_clean is not None and len(q_exp_clean) > 0:
            # Handle error bars
            error_y = None
            error_x = None
            
            if yerr is not None:
                yerr_clean = yerr[np.isfinite(q_exp) & np.isfinite(r_exp)]
                if logx:
                    yerr_clean = yerr_clean[q_exp > 0.0]
                if logy:
                    yerr_clean = yerr_clean[r_exp > 0.0]
                error_y = dict(type='data', array=yerr_clean, visible=True, color=exp_errcolor)
            
            if xerr is not None:
                xerr_clean = xerr[np.isfinite(q_exp) & np.isfinite(r_exp)]
                if logx:
                    xerr_clean = xerr_clean[q_exp > 0.0]
                if logy:
                    xerr_clean = xerr_clean[r_exp > 0.0]
                error_x = dict(type='data', array=xerr_clean, visible=True, color=exp_errcolor)
            
            # Add experimental data trace
            fig.add_trace(
                go.Scatter(
                    x=q_exp_clean,
                    y=r_exp_clean,
                    mode='markers',
                    marker=dict(color=exp_color, size=6),
                    error_y=error_y,
                    error_x=error_x,
                    name=exp_label,
                    hovertemplate='<b>%{fullData.name}</b><br>q: %{x}<br>R: %{y}<extra></extra>'
                )
            )

    # Plot predicted curve
    if q_pred is not None and r_pred is not None:
        q_pred_clean, r_pred_clean = _mask_data(q_pred, r_pred)
        
        if q_pred_clean is not None and len(q_pred_clean) > 0:
            fig.add_trace(
                go.Scatter(
                    x=q_pred_clean,
                    y=r_pred_clean,
                    mode='lines',
                    line=dict(color=pred_color, width=2),
                    name=pred_label,
                    hovertemplate='<b>%{fullData.name}</b><br>q: %{x}<br>R: %{y}<extra></extra>'
                )
            )

    # Plot polished curve
    if q_pol is not None and r_pol is not None:
        q_pol_clean, r_pol_clean = _mask_data(q_pol, r_pol)
        
        if q_pol_clean is not None and len(q_pol_clean) > 0:
            fig.add_trace(
                go.Scatter(
                    x=q_pol_clean,
                    y=r_pol_clean,
                    mode='lines',
                    line=dict(color=pol_color, width=2, dash='dash'),
                    name=pol_label,
                    hovertemplate='<b>%{fullData.name}</b><br>q: %{x}<br>R: %{y}<extra></extra>'
                )
            )

    # Update axis settings for reflectivity plot
    fig.update_xaxes(
        title_text="q [Å⁻¹]",
        type='log' if logx else 'linear'
    )
    fig.update_yaxes(
        title_text="R(q)",
        type='log' if logy else 'linear'
    )

    # The fig is already a FigureWidget, so changes are automatically reflected
    return fig


def plot_sld_only(
    plot_manager: PlotlyPlotManager,
    figure_id: str,
    *,
    z_sld=None, 
    sld_pred=None, 
    sld_pol=None,
    sld_pred_color='red',
    sld_pol_color='orange',
    sld_pred_label='pred. SLD',
    sld_pol_label='polished SLD',
    width=600,
    height=250
):
    """
    Plot SLD profile data only using Plotly
    
    This function creates or updates a Plotly figure widget with SLD profile data only.
    """
    
    def _np(a): 
        return None if a is None else np.asarray(a)

    # Convert inputs to numpy arrays
    z_sld, sld_pred, sld_pol = _np(z_sld), _np(sld_pred), _np(sld_pol)

    # Create or get existing figure widget
    try:
        fig = plot_manager.get_figure(figure_id)
        # Clear existing traces
        plot_manager.clear_figure(figure_id)
    except ValueError:
        # Figure doesn't exist, create new one
        fig = plot_manager.create_sld_figure(figure_id, width, height)

    # Plot SLD profiles
    if z_sld is not None:
        if sld_pred is not None:
            fig.add_trace(
                go.Scatter(
                    x=z_sld,
                    y=sld_pred,
                    mode='lines',
                    line=dict(color=sld_pred_color, width=2),
                    name=sld_pred_label,
                    hovertemplate='<b>%{fullData.name}</b><br>z: %{x}<br>SLD: %{y}<extra></extra>'
                )
            )

        if sld_pol is not None:
            fig.add_trace(
                go.Scatter(
                    x=z_sld,
                    y=sld_pol,
                    mode='lines',
                    line=dict(color=sld_pol_color, width=2, dash='dash'),
                    name=sld_pol_label,
                    hovertemplate='<b>%{fullData.name}</b><br>z: %{x}<br>SLD: %{y}<extra></extra>'
                )
            )

    # Update axis settings for SLD plot
    fig.update_xaxes(title_text="z [Å]")
    fig.update_yaxes(title_text="SLD [10⁻⁶ Å⁻²]")

    # The fig is already a FigureWidget, so changes are automatically reflected
    return fig
