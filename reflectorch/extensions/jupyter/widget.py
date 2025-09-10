"""
Reflectorch Jupyter Widget

This module provides a clean, simplified widget interface for reflectometry analysis.
The widget features a tabbed interface with organized controls and real-time results.

Key Features:
- Professional tabbed interface
- Interactive parameter table with live results
- Comprehensive preprocessing and plotting controls
- Interactive plotting with coordinate display
- Clean, modular architecture
"""

import numpy as np
from typing import Optional, Dict, Any, Union
import ipywidgets as widgets
from IPython.display import display

from .components import (
    ParameterTable, 
    PreprocessingControls, 
    PredictionControls, 
    PlottingControls, 
    WidgetSettingsExtractor
)
from .interactive_plotting import InteractivePlotManager, plot_reflectivity_interactive
from ...inference.plotting import plot_reflectivity


class ReflectorchWidget:
    """
    Interactive Jupyter Widget for Reflectometry Analysis
    
    A modern, tabbed interface for reflectometry data analysis with Reflectorch models.
    Features real-time parameter updates, interactive plotting, and comprehensive controls.
    
    Attributes:
        model: The EasyInferenceModel instance
        prediction_result: Latest prediction results
        interactive_plots: Whether interactive plotting is enabled
        
    Example:
        ```python
        from reflectorch.inference import EasyInferenceModel
        from reflectorch.extensions.jupyter import ReflectorchWidget
        
        model = EasyInferenceModel('config.yaml')
        widget = ReflectorchWidget(model)
        
        widget.display(
            reflectivity_curve=data,
            q_values=q_values,
            sigmas=sigmas
        )
        
        # Access results
        results = widget.prediction_result
        ```
    """
    
    def __init__(self, model, interactive_plots: bool = True):
        """
        Initialize the Reflectorch widget
        
        Args:
            model: EasyInferenceModel instance for making predictions
            interactive_plots: Enable interactive plotting features (default: True)
        """
        self.model = model
        self.interactive_plots = interactive_plots
        self.prediction_result = None
        self.plot_manager = InteractivePlotManager() if interactive_plots else None
        
        # Widget components (initialized when display is called)
        self.parameter_table = None
        self.preprocessing_controls = None
        self.prediction_controls = None
        self.plotting_controls = None
        
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model has required attributes"""
        required_attrs = ['trainer', 'preprocess_and_predict']
        for attr in required_attrs:
            if not hasattr(self.model, attr):
                raise ValueError(f"Model must have '{attr}' attribute")
    
    def display(self, 
                reflectivity_curve: np.ndarray,
                q_values: np.ndarray,
                sigmas: Optional[np.ndarray] = None,
                q_resolution: Optional[Union[float, np.ndarray]] = None,
                initial_prior_bounds: Optional[np.ndarray] = None,
                ambient_sld: Optional[float] = None):
        """
        Display the widget interface
        
        Args:
            reflectivity_curve: Experimental reflectivity data
            q_values: Momentum transfer values (required)
            sigmas: Experimental uncertainties (optional)
            q_resolution: Q-resolution, float or array (optional)
            initial_prior_bounds: Initial bounds for priors, shape (n_params, 2)
            ambient_sld: Ambient SLD value (optional)
        """
        if q_values is None:
            raise ValueError("q_values must be provided")
        
        # Store data for prediction
        self._data = {
            'reflectivity_curve': reflectivity_curve,
            'q_values': q_values,
            'sigmas': sigmas,
            'q_resolution': q_resolution,
            'ambient_sld': ambient_sld
        }
        
        # Get model parameters info
        param_labels = self.model.trainer.loader.prior_sampler.param_model.get_param_labels()
        min_bounds = self.model.trainer.loader.prior_sampler.min_bounds.cpu().numpy().flatten()
        max_bounds = self.model.trainer.loader.prior_sampler.max_bounds.cpu().numpy().flatten()
        max_deltas = self.model.trainer.loader.prior_sampler.max_delta.cpu().numpy().flatten()
        
        # Create widget components
        self.parameter_table = ParameterTable(
            param_labels, min_bounds, max_bounds, max_deltas, initial_prior_bounds
        )
        self.preprocessing_controls = PreprocessingControls(len(reflectivity_curve))
        self.prediction_controls = PredictionControls()
        self.plotting_controls = PlottingControls()
        
        # Create tabbed interface
        tabs = widgets.Tab()
        tabs.children = [
            self.parameter_table.widget,
            self.preprocessing_controls.widget,
            self.prediction_controls.widget,
            self.plotting_controls.widget
        ]
        tabs.titles = ['Parameters', 'Preprocessing', 'Prediction', 'Plotting']
        
        # Control buttons
        predict_button = widgets.Button(
            description="Predict",
            button_style='primary',
            tooltip='Run prediction with current settings',
            layout=widgets.Layout(width='120px')
        )
        close_button = widgets.Button(
            description="Close",
            button_style='danger',
            tooltip='Close this widget',
            layout=widgets.Layout(width='120px')
        )
        
        # Output areas
        output = widgets.Output()
        
        # Coordinate display for interactive plotting
        coord_display = None
        if self.interactive_plots:
            coord_display = widgets.HTML(
                value="<i>Interactive plotting enabled - move cursor over plot for coordinates</i>",
                style={'description_width': 'initial'},
                layout={'margin': '5px 0px'}
            )
        
        # Main layout
        header = widgets.HTML("<h2>🔬 Reflectorch Analysis Widget</h2>")
        
        main_controls = widgets.VBox([
            header,
            tabs,
            widgets.HBox([predict_button, close_button], layout=widgets.Layout(justify_content='center')),
        ])
        
        layout_components = [main_controls]
        if coord_display:
            layout_components.append(coord_display)
        layout_components.append(output)
        
        container = widgets.VBox(layout_components)
        display(container)
        
        # Setup event handlers
        self._setup_event_handlers(predict_button, close_button, output, coord_display, container)
        
        # Setup truncation synchronization
        self._setup_truncation_sync()
        
        # Create initial plot with experimental data
        self._create_initial_plot(coord_display)
        
        # Setup reactive plot updates for plotting controls
        self._setup_reactive_plot_updates(coord_display)
    
    def _create_initial_plot(self, coord_display):
        """Create initial plot showing experimental data"""
        try:
            # Use default settings for initial plot since controls are just being created
            settings = {
                'show_error_bars': True,
                'show_q_resolution': True,
                'exp_color': 'blue',
                'exp_errcolor': 'purple',
                'log_x_axis': False,
                'plot_sld_profile': True  # Always create with SLD support for later updates
            }
            
            # Plot initial experimental data
            self._plot_initial_data(settings, coord_display)
            
        except Exception as e:
            print(f"⚠️  Could not create initial plot: {str(e)}")
    
    def _plot_initial_data(self, settings, coord_display):
        """Plot only experimental data before any prediction"""
        # Prepare experimental data for plotting
        q_exp_plot = self._data['q_values']
        r_exp_plot = self._data['reflectivity_curve']
        yerr_plot = self._data['sigmas'] if settings['show_error_bars'] and self._data['sigmas'] is not None else None
        xerr_plot = self._data['q_resolution'] if settings['show_q_resolution'] and self._data['q_resolution'] is not None else None
        
        # Plot with appropriate backend
        if self.interactive_plots:
            self._plot_initial_interactive(
                q_exp_plot, r_exp_plot, yerr_plot, xerr_plot,
                settings, coord_display
            )
        else:
            self._plot_initial_static(
                q_exp_plot, r_exp_plot, yerr_plot, xerr_plot,
                settings
            )
    
    def _plot_initial_interactive(self, q_exp, r_exp, yerr, xerr, settings, coord_display):
        """Plot initial experimental data using interactive backend"""
        figure_id = "reflectorch_widget"
        
        # Always create with SLD profile support to match PlottingControls default (True)
        # and prevent layout mismatches during prediction updates
        fig, axes = plot_reflectivity_interactive(
            plot_manager=self.plot_manager,
            figure_id=figure_id,
            q_exp=q_exp, r_exp=r_exp, yerr=yerr, xerr=xerr,
            exp_style=('errorbar' if (yerr is not None or xerr is not None) else 'scatter'),
            exp_color=settings['exp_color'],
            exp_errcolor=settings['exp_errcolor'],
            exp_label='experimental data',
            # No prediction data yet
            q_pred=None, r_pred=None, 
            q_pol=None, r_pol=None,
            z_sld=None, sld_pred=None, sld_pol=None,
            plot_sld_profile=True,  # Always True to match default and prevent layout changes
            logx=settings['log_x_axis'], logy=True,
            figsize=(12, 6),
            legend=True
        )
        
        # Sync coordinate display
        if coord_display:
            plot_coord_display = self.plot_manager.get_coordinate_display(figure_id)
            if plot_coord_display:
                coord_display.value = plot_coord_display.value
                def sync_coords(change):
                    coord_display.value = change['new']
                plot_coord_display.observe(sync_coords, names='value')
    
    def _plot_initial_static(self, q_exp, r_exp, yerr, xerr, settings):
        """Plot initial experimental data using static matplotlib"""
        plot_reflectivity(
            q_exp=q_exp, r_exp=r_exp, yerr=yerr, xerr=xerr,
            exp_style=('errorbar' if (yerr is not None or xerr is not None) else 'scatter'),
            exp_color=settings['exp_color'],
            exp_errcolor=settings['exp_errcolor'],
            exp_label='experimental data',
            # No prediction data yet
            q_pred=None, r_pred=None,
            q_pol=None, r_pol=None,
            z_sld=None, sld_pred=None, sld_pol=None,
            plot_sld_profile=True,  # Always True to match default and prevent layout changes
            logx=settings['log_x_axis'], logy=True,
            figsize=(12, 6),
            legend=True
        )
    
    def _setup_event_handlers(self, predict_button, close_button, output, coord_display, container):
        """Setup button event handlers"""
        
        @output.capture(clear_output=True)
        def on_predict(_):
            """Handle predict button click"""
            output.clear_output(wait=True)
            
            try:
                # Extract settings from all components
                settings = WidgetSettingsExtractor.extract_settings(
                    self.parameter_table,
                    self.preprocessing_controls,
                    self.prediction_controls,
                    self.plotting_controls
                )
                
                # Run prediction
                prediction_result = self.model.preprocess_and_predict(
                    reflectivity_curve=self._data['reflectivity_curve'],
                    q_values=self._data['q_values'],
                    prior_bounds=settings['prior_bounds'],
                    sigmas=self._data['sigmas'],
                    q_resolution=self._data['q_resolution'],
                    ambient_sld=self._data['ambient_sld'],
                    
                    # Prediction settings
                    clip_prediction=True,
                    polish_prediction=settings['polish_prediction'],
                    use_sigmas_for_polishing=settings['use_sigmas_for_polishing'],
                    calc_pred_curve=settings['calc_pred_curve'],
                    calc_pred_sld_profile=(settings['calc_pred_sld'] or settings['plot_sld_profile']),
                    calc_polished_sld_profile=(settings['calc_pol_sld'] or settings['plot_sld_profile']),
                    sld_profile_padding_left=settings['sld_pad_left'],
                    sld_profile_padding_right=settings['sld_pad_right'],
                    
                    # Preprocessing settings
                    truncate_index_left=settings['truncate_left'],
                    truncate_index_right=settings['truncate_right'],
                    enable_error_bars_filtering=settings['enable_filtering'],
                    filter_threshold=settings['filter_threshold'],
                    filter_remove_singles=settings['filter_remove_singles'],
                    filter_remove_consecutives=settings['filter_remove_consecutives'],
                    filter_consecutive=settings['filter_consecutive'],
                    filter_q_start_trunc=settings['filter_q_start_trunc'],
                )
                
                # Update parameter table with results
                self.parameter_table.update_results(prediction_result)
                
                # Plot results
                self._plot_results(prediction_result, settings, coord_display)
                
                # Store results
                self.prediction_result = prediction_result
                
            except Exception as e:
                print(f"❌ Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        def on_close(_):
            """Handle close button click"""
            if self.interactive_plots and self.plot_manager:
                self.plot_manager.close_figure("reflectorch_widget")
            container.close()
            print("✅ Reflectorch widget closed")
        
        # Connect event handlers
        predict_button.on_click(on_predict)
        close_button.on_click(on_close)
    
    def _plot_results(self, prediction_result, settings, coord_display):
        """Plot prediction results with current settings"""
        # Prepare plotting data
        q_exp_plot = self._data['q_values']
        r_exp_plot = self._data['reflectivity_curve']
        yerr_plot = self._data['sigmas'] if settings['show_error_bars'] else None
        xerr_plot = self._data['q_resolution'] if settings['show_q_resolution'] else None
        
        q_pred = prediction_result.get('q_plot_pred', None)
        r_pred = prediction_result.get('predicted_curve', None)
        q_pol = self._data['q_values'] if 'polished_curve' in prediction_result else None
        r_pol = prediction_result.get('polished_curve', None)
        
        z_sld = prediction_result.get('predicted_sld_xaxis', None)
        sld_pred = prediction_result.get('predicted_sld_profile', None)
        sld_pol = prediction_result.get('sld_profile_polished', None)
        
        # Plot with appropriate backend
        if self.interactive_plots:
            self._plot_interactive(
                q_exp_plot, r_exp_plot, yerr_plot, xerr_plot,
                q_pred, r_pred, q_pol, r_pol,
                z_sld, sld_pred, sld_pol,
                settings, coord_display
            )
        else:
            self._plot_static(
                q_exp_plot, r_exp_plot, yerr_plot, xerr_plot,
                q_pred, r_pred, q_pol, r_pol,
                z_sld, sld_pred, sld_pol,
                settings
            )
    
    def _plot_interactive(self, q_exp, r_exp, yerr, xerr, q_pred, r_pred, q_pol, r_pol, 
                         z_sld, sld_pred, sld_pol, settings, coord_display):
        """Plot using interactive backend"""
        figure_id = "reflectorch_widget"
        fig, axes = plot_reflectivity_interactive(
            plot_manager=self.plot_manager,
            figure_id=figure_id,
            q_exp=q_exp, r_exp=r_exp, yerr=yerr, xerr=xerr,
            exp_style=('errorbar' if (settings['show_error_bars'] or settings['show_q_resolution']) else 'scatter'),
            exp_color=settings['exp_color'],
            exp_errcolor=settings['exp_errcolor'],
            q_pred=q_pred, r_pred=r_pred, pred_color=settings['pred_color'],
            q_pol=q_pol, r_pol=r_pol, pol_color=settings['pol_color'],
            z_sld=z_sld, sld_pred=sld_pred, sld_pol=sld_pol,
            sld_pred_color=settings['sld_pred_color'],
            sld_pol_color=settings['sld_pol_color'],
            plot_sld_profile=settings['plot_sld_profile'],
            logx=settings['log_x_axis'], logy=True,
            figsize=(12, 6),
            legend=True
        )
        
        # Sync coordinate display
        if coord_display:
            plot_coord_display = self.plot_manager.get_coordinate_display(figure_id)
            if plot_coord_display:
                coord_display.value = plot_coord_display.value
                def sync_coords(change):
                    coord_display.value = change['new']
                plot_coord_display.observe(sync_coords, names='value')
    
    def _plot_static(self, q_exp, r_exp, yerr, xerr, q_pred, r_pred, q_pol, r_pol,
                    z_sld, sld_pred, sld_pol, settings):
        """Plot using static matplotlib"""
        plot_reflectivity(
            q_exp=q_exp, r_exp=r_exp, yerr=yerr, xerr=xerr,
            exp_style=('errorbar' if (settings['show_error_bars'] or settings['show_q_resolution']) else 'scatter'),
            exp_color=settings['exp_color'],
            exp_errcolor=settings['exp_errcolor'],
            q_pred=q_pred, r_pred=r_pred, pred_color=settings['pred_color'],
            q_pol=q_pol, r_pol=r_pol, pol_color=settings['pol_color'],
            z_sld=z_sld, sld_pred=sld_pred, sld_pol=sld_pol,
            sld_pred_color=settings['sld_pred_color'],
            sld_pol_color=settings['sld_pol_color'],
            plot_sld_profile=settings['plot_sld_profile'],
            logx=settings['log_x_axis'], logy=True,
            figsize=(12, 6),
            legend=True
        )
    
    def _setup_truncation_sync(self):
        """Setup synchronization between truncation sliders"""
        # Find truncation widgets
        trunc_widgets = WidgetSettingsExtractor._find_widgets_by_description(
            self.preprocessing_controls.widget, ['Left index:', 'Right index:']
        )
        
        if len(trunc_widgets) == 2:
            trunc_left, trunc_right = trunc_widgets
            
            def sync_truncation(_):
                if trunc_left.value >= trunc_right.value:
                    trunc_left.value = max(0, trunc_right.value - 1)
            
            trunc_left.observe(sync_truncation, names='value')
            trunc_right.observe(sync_truncation, names='value')
    
    def _setup_reactive_plot_updates(self, coord_display):
        """Setup observers for plotting controls that should trigger immediate plot updates"""
        if not self.plotting_controls:
            return
            
        # Find plotting controls that should trigger plot updates
        reactive_controls = WidgetSettingsExtractor._find_widgets_by_description(
            self.plotting_controls.widget, 
            [
                'Show error bars', 'Show q-resolution', 'Log x-axis', 'Plot SLD profile',
                'Data color:', 'Error bars:', 'Prediction:', 'Polished:', 
                'SLD pred:', 'SLD polish:'
            ]
        )
        
        def update_plot_on_change(change):
            """Update plot when plotting controls change"""
            # Only update if we have prediction results to show
            if self.prediction_result is not None:
                try:
                    # Extract current settings
                    settings = WidgetSettingsExtractor.extract_settings(
                        self.parameter_table,
                        self.preprocessing_controls,
                        self.prediction_controls,
                        self.plotting_controls
                    )
                    
                    # Update plot with new settings
                    self._plot_results(self.prediction_result, settings, coord_display)
                    
                except Exception as e:
                    print(f"⚠️  Error updating plot: {str(e)}")
            else:
                # If no prediction results yet, just update the initial plot
                try:
                    self._update_initial_plot_style(coord_display)
                except Exception as e:
                    print(f"⚠️  Error updating initial plot: {str(e)}")
        
        # Setup observers for all reactive controls
        for control in reactive_controls:
            if hasattr(control, 'observe'):
                control.observe(update_plot_on_change, names='value')
    
    def _update_initial_plot_style(self, coord_display):
        """Update initial plot styling based on current control settings"""
        if not self.plotting_controls:
            return
            
        try:
            # Extract current plotting settings
            settings = WidgetSettingsExtractor.extract_settings(
                self.parameter_table,
                self.preprocessing_controls,
                self.prediction_controls,
                self.plotting_controls
            )
            
            # Update initial plot with new styling
            self._plot_initial_data(settings, coord_display)
            
        except Exception as e:
            print(f"⚠️  Error updating initial plot style: {str(e)}")
