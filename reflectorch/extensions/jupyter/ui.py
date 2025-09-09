

"""
Jupyter Widget Interface for Reflectorch

This module provides interactive Jupyter widgets for reflectometry analysis using Reflectorch models.
The widgets offer a user-friendly interface for data preprocessing, prediction, and visualization.

Key Features:
    - Interactive parameter bounds adjustment with sliders
    - Real-time prediction and visualization with interactive plots
    - Persistent figures that update in-place (no figure accumulation)
    - Cursor coordinate display and zoom/pan functionality
    - Comprehensive data preprocessing controls
    - Customizable plotting and color options
    - Easy integration with existing Reflectorch workflows

Interactive Plotting:
    - Uses ipympl backend for enhanced interactivity in JupyterLab
    - Figures update in-place rather than creating new windows
    - Real-time coordinate display as you move cursor over plots
    - Built-in zoom, pan, and navigation controls
    - Graceful fallback to standard matplotlib if interactive mode disabled

Quick Start:
    ```python
    # Basic widget usage with interactive plotting
    from reflectorch.inference import EasyInferenceModel
    from reflectorch.extensions.jupyter import create_basic_widget
    
    model = EasyInferenceModel('your_config.yaml')
    widget = create_basic_widget(model, reflectivity_curve, q_values=q_values)
    
    # Advanced widget usage with full interactive features
    from reflectorch.extensions.jupyter import create_advanced_widget
    
    widget = create_advanced_widget(
        model, 
        reflectivity_curve, 
        q_values,
        sigmas=sigmas,
        initial_prior_bounds=my_bounds,
        interactive_plots=True  # Default, enables interactive plotting
    )
    
    # Disable interactive plotting if needed
    widget = create_basic_widget(model, data, interactive_plots=False)
    
    # Access results
    results = widget.prediction_result
    ```

Widget Types:
    1. **Basic Widget**: Simple interface for adjusting prior bounds and making predictions
    2. **Advanced Widget**: Full-featured interface with preprocessing and visualization controls

Note: For best interactive experience, ensure ipympl is installed and use '%matplotlib widget' in Jupyter.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, List, Tuple
import ipywidgets as widgets
from IPython.display import display

from ...inference.plotting import plot_reflectivity, plot_prediction_results, print_prediction_results
from .interactive_plotting import InteractivePlotManager, plot_reflectivity_interactive, plot_prediction_results_interactive


class ReflectJupyterWidget:
    """
    Interactive Jupyter Widget for Reflectometry Analysis with Reflectorch
    
    This class provides two main widget interfaces:
    1. Basic prediction widget with prior bounds adjustment
    2. Advanced preprocessing and prediction widget with full control
    
    The widgets allow interactive adjustment of parameters and real-time visualization
    of prediction results.
    
    Attributes:
        model: The EasyInferenceModel instance
        prediction_result (dict): Latest prediction results from widget interaction
        
    Example:
        ```python
        from reflectorch.inference import EasyInferenceModel
        from reflectorch.extensions.jupyter import ReflectJupyterWidget
        
        # Load model
        model = EasyInferenceModel('your_config')
        
        # Create widget
        widget = ReflectJupyterWidget(model)
        
        # Use basic prediction widget
        widget.basic_predict_widget(reflectivity_curve, q_values=q_values)
        
        # Or use advanced widget with preprocessing
        widget.advanced_predict_widget(
            reflectivity_curve, 
            q_values=q_values,
            sigmas=sigmas
        )
        ```
    """
    
    def __init__(self, model, interactive_plots=True):
        """
        Initialize the widget with a Reflectorch model
        
        Args:
            model: EasyInferenceModel instance for making predictions
            interactive_plots: Enable interactive plotting features (default: True)
        """
        self.model = model
        self.prediction_result = None
        self.interactive_plots = interactive_plots
        self.plot_manager = InteractivePlotManager() if interactive_plots else None
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model has required attributes"""
        required_attrs = ['trainer', 'predict', 'preprocess_and_predict']
        for attr in required_attrs:
            if not hasattr(self.model, attr):
                raise ValueError(f"Model must have '{attr}' attribute")
    
    def basic_predict_widget(
        self, 
        reflectivity_curve: np.ndarray,
        q_values: Optional[np.ndarray] = None,
        initial_prior_bounds: Optional[np.ndarray] = None,
        **predict_kwargs
    ):
        """
        Display a basic prediction widget with prior bounds sliders
        
        This widget provides a simplified interface focused on adjusting prior bounds
        and making predictions. Results are stored in self.prediction_result.
        
        Args:
            reflectivity_curve: Experimental reflectivity data
            q_values: Momentum transfer values (optional)
            initial_prior_bounds: Initial bounds for priors, shape (n_params, 2)
            **predict_kwargs: Additional keyword arguments passed to model.predict()
            
        Features:
            - Interactive sliders for each parameter's prior bounds
            - Automatic validation of slider ranges
            - Real-time prediction and plotting
            - Parameter results display
            
        Example:
            ```python
            widget.basic_predict_widget(
                reflectivity_curve,
                q_values=q_values,
                initial_prior_bounds=my_bounds,
                calc_pred_sld_profile=True
            )
            ```
        """
        # Get model parameters info
        param_labels = self.model.trainer.loader.prior_sampler.param_model.get_param_labels()
        min_bounds = self.model.trainer.loader.prior_sampler.min_bounds.cpu().numpy().flatten()
        max_bounds = self.model.trainer.loader.prior_sampler.max_bounds.cpu().numpy().flatten()
        max_deltas = self.model.trainer.loader.prior_sampler.max_delta.cpu().numpy().flatten()
        
        print(f'Adjust the sliders for each parameter and press "Predict". Repeat as desired. Press "Close Widget" to finish.')
        
        # Create parameter sliders (matching original basic widget logic exactly)
        interval_widgets = []
        for i in range(len(param_labels)):
            label = widgets.Label(value=f'{param_labels[i]}')
            initial_max = min(max_bounds[i], min_bounds[i] + max_deltas[i])
            slider = widgets.FloatRangeSlider(
                value=[min_bounds[i], initial_max],
                min=min_bounds[i],
                max=max_bounds[i],
                step=0.01,
                layout=widgets.Layout(width='400px'),
                style={'description_width': '60px'}
            )

            def validate_range(change, slider=slider, max_width=max_deltas[i]):
                min_val, max_val = change['new']
                if max_val - min_val > max_width:
                    old_min_val, old_max_val = change['old']
                    if abs(old_min_val - min_val) > abs(old_max_val - max_val):
                        max_val = min_val + max_width
                    else:
                        min_val = max_val - max_width
                    slider.value = [min_val, max_val]

            slider.observe(validate_range, names='value')
            interval_widgets.append((slider, widgets.HBox([label, slider])))
        
        # Create control buttons
        predict_button = widgets.Button(
            description="Predict",
            button_style='primary',
            tooltip='Run prediction with current prior bounds'
        )
        close_button = widgets.Button(
            description="Close Widget",
            button_style='danger',
            tooltip='Close this widget'
        )
        
        # Output area for results
        output = widgets.Output()
        
        # Create coordinate display if interactive plotting is enabled
        coord_display = None
        if self.interactive_plots:
            coord_display = widgets.HTML(
                value="<i>Interactive plotting enabled - move cursor over plot for coordinates</i>",
                style={'description_width': 'initial'},
                layout={'margin': '5px 0px'}
            )
        
        # Layout
        sliders_box = widgets.VBox([iw[1] for iw in interval_widgets])
        buttons_box = widgets.HBox([predict_button, close_button])
        
        layout_components = [
            widgets.HTML("<h3>Reflectorch Basic Prediction Widget</h3>"),
            sliders_box, 
            buttons_box
        ]
        
        if coord_display:
            layout_components.append(coord_display)
            
        layout_components.append(output)
        container = widgets.VBox(layout_components)
        
        display(container)
        
        # Event handlers
        @output.capture(clear_output=True)
        def on_predict_click(_):
            """Handle predict button click"""
            try:
                # Get current slider values
                prior_bounds = np.array([[slider.value[0], slider.value[1]] 
                                       for slider, _ in interval_widgets])
                
                # Make prediction
                prediction_result = self.model.predict(
                    reflectivity_curve=reflectivity_curve,
                    q_values=q_values,
                    prior_bounds=prior_bounds,
                    **predict_kwargs
                )
                
                # Display results
                print_prediction_results(prediction_result)
                
                # Plot results
                q_exp_plot = q_values if q_values is not None else self.model.trainer.loader.q_generator.q.cpu().numpy()
                
                if self.interactive_plots:
                    # Use interactive plotting
                    figure_id = "basic_prediction"
                    fig, axes = plot_prediction_results_interactive(
                        plot_manager=self.plot_manager,
                        figure_id=figure_id,
                        prediction_dict=prediction_result,
                        q_exp=q_exp_plot,
                        curve_exp=reflectivity_curve,
                    )
                    
                    # Update coordinate display with the plot manager's version
                    plot_coord_display = self.plot_manager.get_coordinate_display(figure_id)
                    if plot_coord_display and coord_display:
                        coord_display.value = plot_coord_display.value
                        # Sync the displays
                        def sync_coords(change):
                            coord_display.value = change['new']
                        plot_coord_display.observe(sync_coords, names='value')
                else:
                    # Use standard plotting
                    plot_prediction_results(
                        prediction_result,
                        q_exp=q_exp_plot,
                        curve_exp=reflectivity_curve,
                    )
                
                # Store results
                self.prediction_result = prediction_result
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        def on_close_click(_):
            """Handle close button click"""
            # Clean up interactive plots
            if self.interactive_plots and self.plot_manager:
                self.plot_manager.close_figure("basic_prediction")
            container.close()
            print("Basic prediction widget closed.")
        
        # Connect event handlers
        predict_button.on_click(on_predict_click)
        close_button.on_click(on_close_click)
    
    def _create_parameter_sliders(self, param_labels, min_bounds, max_bounds, max_deltas, initial_bounds=None):
        """Create parameter slider widgets with validation"""
        sliders = []
        init_bounds = np.array(initial_bounds) if initial_bounds is not None else None
        
        for i, label in enumerate(param_labels):
            # Determine initial values
            if init_bounds is not None and i < len(init_bounds):
                init_min, init_max = float(init_bounds[i, 0]), float(init_bounds[i, 1])
            else:
                init_min = float(min_bounds[i])
                init_max = float(min(min_bounds[i] + max_deltas[i], max_bounds[i]))
            
            # Create widgets
            label_widget = widgets.Label(value=f'{label}')
            slider = widgets.FloatRangeSlider(
                value=[init_min, init_max],
                min=float(min_bounds[i]),
                max=float(max_bounds[i]),
                step=0.01,
                layout=widgets.Layout(width='420px'),
                readout_format='.3f'
            )
            
            # Add validation for maximum range
            def create_validator(slider_widget, max_width=float(max_deltas[i])):
                def validate_range(change):
                    a, b = change['new']
                    if b - a > max_width:
                        oa, ob = change['old']
                        if abs(oa - a) > abs(ob - b):
                            b = a + max_width
                        else:
                            a = b - max_width
                        slider_widget.value = (a, b)
                return validate_range
            
            slider.observe(create_validator(slider), names='value')
            slider_row = widgets.HBox([label_widget, slider])
            sliders.append((slider, slider_row))
        
        return sliders
    
    def advanced_predict_widget(
        self,
        reflectivity_curve: np.ndarray,
        q_values: np.ndarray,
        sigmas: Optional[np.ndarray] = None,
        q_resolution: Optional[Union[float, np.ndarray]] = None,
        initial_prior_bounds: Optional[np.ndarray] = None,
        ambient_sld: Optional[float] = None,
    ):
        """
        Display an advanced prediction widget with full preprocessing and visualization controls
        
        This widget provides comprehensive control over data preprocessing, prediction parameters,
        and visualization options. Results are stored in self.prediction_result.
        
        Args:
            reflectivity_curve: Experimental reflectivity data
            q_values: Momentum transfer values (required)
            sigmas: Experimental uncertainties (optional)
            q_resolution: Q-resolution, float or array (optional)
            initial_prior_bounds: Initial bounds for priors, shape (n_params, 2)
            ambient_sld: Ambient SLD value (optional)
            
        Features:
            - Prior bounds adjustment with sliders
            - Data preprocessing controls (truncation, filtering)
            - Prediction polishing options
            - Comprehensive plotting controls
            - Color customization
            - Real-time computation toggles
            
        Example:
            ```python
            widget.advanced_predict_widget(
                reflectivity_curve,
                q_values=q_values,
                sigmas=sigmas,
                q_resolution=0.05,
                initial_prior_bounds=my_bounds,
                ambient_sld=0.0
            )
            ```
        """
        if q_values is None:
            raise ValueError("q_values must be provided for the advanced widget.")
        
        n_datapoints = len(reflectivity_curve)
        
        # Get model parameters info
        param_labels = self.model.trainer.loader.prior_sampler.param_model.get_param_labels()
        min_bounds = self.model.trainer.loader.prior_sampler.min_bounds.cpu().numpy().flatten()
        max_bounds = self.model.trainer.loader.prior_sampler.max_bounds.cpu().numpy().flatten()
        max_deltas = self.model.trainer.loader.prior_sampler.max_delta.cpu().numpy().flatten()
        
        # Create widget sections
        priors_section = self._create_priors_section(
            param_labels, min_bounds, max_bounds, max_deltas, initial_prior_bounds
        )
        controls_section = self._create_controls_section(n_datapoints)
        
        # Control buttons
        predict_button = widgets.Button(
            description="Predict",
            button_style='primary',
            tooltip='Run prediction with current settings'
        )
        close_button = widgets.Button(
            description="Close Widget", 
            button_style='danger',
            tooltip='Close this widget'
        )
        
        # Output area
        output = widgets.Output()
        
        # Create coordinate display if interactive plotting is enabled
        coord_display = None
        if self.interactive_plots:
            coord_display = widgets.HTML(
                value="<i>Interactive plotting enabled - move cursor over plot for coordinates</i>",
                style={'description_width': 'initial'},
                layout={'margin': '5px 0px'}
            )
        
        # Main layout
        main_controls = widgets.VBox([
            widgets.HTML("<h3>Reflectorch Advanced Prediction Widget</h3>"),
            priors_section,
            controls_section,
            widgets.HBox([predict_button, close_button]),
        ])
        
        layout_components = [main_controls]
        if coord_display:
            layout_components.append(coord_display)
        layout_components.append(output)
        
        container = widgets.VBox(layout_components)
        display(container)
        
        # Event handlers
        @output.capture(clear_output=True)
        def _on_predict(_):
            """Handle predict button click with all current settings (exact replication of original)"""
            output.clear_output(wait=True)
            
            try:
                # Get current parameter values
                settings = self._extract_widget_settings(priors_section, controls_section)
                
                # Run prediction
                prediction_result = self.model.preprocess_and_predict(
                    reflectivity_curve=reflectivity_curve,
                    q_values=q_values,
                    prior_bounds=settings['prior_bounds'],
                    sigmas=sigmas,
                    q_resolution=q_resolution,
                    ambient_sld=ambient_sld,
                    
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
                
                # Display results
                print_prediction_results(prediction_result)
                
                # Prepare plotting data
                q_exp_plot = q_values
                r_exp_plot = reflectivity_curve
                yerr_plot = sigmas if settings['show_error_bars'] else None
                xerr_plot = q_resolution if settings['show_q_resolution'] else None
                
                q_pred = prediction_result.get('q_plot_pred', None)
                r_pred = prediction_result.get('predicted_curve', None)
                q_pol = q_values if 'polished_curve' in prediction_result else None
                r_pol = prediction_result.get('polished_curve', None)
                
                z_sld = prediction_result.get('predicted_sld_xaxis', None)
                sld_pred = prediction_result.get('predicted_sld_profile', None)
                sld_pol = prediction_result.get('sld_profile_polished', None)
                
                # Plot results with full customization
                if self.interactive_plots:
                    # Use interactive plotting
                    figure_id = "advanced_prediction"
                    fig, axes = plot_reflectivity_interactive(
                        plot_manager=self.plot_manager,
                        figure_id=figure_id,
                        q_exp=q_exp_plot, r_exp=r_exp_plot,
                        yerr=yerr_plot, xerr=xerr_plot,
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
                    
                    # Update coordinate display with the plot manager's version
                    plot_coord_display = self.plot_manager.get_coordinate_display(figure_id)
                    if plot_coord_display and coord_display:
                        coord_display.value = plot_coord_display.value
                        # Sync the displays
                        def sync_coords(change):
                            coord_display.value = change['new']
                        plot_coord_display.observe(sync_coords, names='value')
                else:
                    # Use standard plotting
                    plot_reflectivity(
                        q_exp=q_exp_plot, r_exp=r_exp_plot,
                        yerr=yerr_plot, xerr=xerr_plot,
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
                
                # Store results
                self.prediction_result = prediction_result
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        def _on_close(_):
            """Handle close button click"""
            # Clean up interactive plots
            if self.interactive_plots and self.plot_manager:
                self.plot_manager.close_figure("advanced_prediction")
            container.close()
            print("Advanced prediction widget closed.")
        
        # Connect event handlers
        predict_button.on_click(_on_predict)
        close_button.on_click(_on_close)
        
        # Setup truncation synchronization
        trunc_widgets = self._find_widgets_by_description(controls_section, ['truncate left', 'truncate right'])
        if len(trunc_widgets) == 2:
            trunc_left, trunc_right = trunc_widgets
            
            def sync_truncation(_):
                if trunc_left.value >= trunc_right.value:
                    trunc_left.value = max(0, trunc_right.value - 1)
            
            trunc_left.observe(sync_truncation, names='value')
            trunc_right.observe(sync_truncation, names='value')
    
    def _create_priors_section(self, param_labels, min_bounds, max_bounds, max_deltas, initial_bounds=None):
        """Create the priors section with parameter sliders (exact replication of original advanced widget)"""
        sliders, rows = [], []
        init_pb = np.array(initial_bounds) if initial_bounds is not None else None
        
        for i in range(len(param_labels)):
            init_min = float(init_pb[i, 0]) if init_pb is not None else float(min_bounds[i])
            init_max = float(init_pb[i, 1]) if init_pb is not None else float(min(min_bounds[i] + max_deltas[i], max_bounds[i]))
            lab = widgets.Label(value=param_labels[i])
            s = widgets.FloatRangeSlider(
                value=[init_min, init_max],
                min=float(min_bounds[i]),
                max=float(max_bounds[i]),
                step=0.01,
                layout=widgets.Layout(width='420px'),
                readout_format='.3f'
            )
            # Constrain slider widths (exact replication of original)
            def _mk_validator(slider, max_width=float(max_deltas[i])):
                def _validate(change):
                    a, b = change['new']
                    if b - a > max_width:
                        oa, ob = change['old']
                        if abs(oa - a) > abs(ob - b):
                            b = a + max_width
                        else:
                            a = b - max_width
                        slider.value = (a, b)
                return _validate
            s.observe(_mk_validator(s), names='value')
            sliders.append(s)
            rows.append(widgets.HBox([lab, s]))
            
        # Store sliders for later access
        self._priors_sliders = sliders
        
        return widgets.VBox([widgets.HTML("<h4>Prior Bounds</h4>")] + rows)
    
    def _create_controls_section(self, n_datapoints):
        """Create comprehensive controls section with preprocessing, prediction, and plotting options"""
        # Preprocessing section
        preprocessing_section = widgets.VBox([
            widgets.HTML("<h4>Data Preprocessing</h4>"),
            widgets.HTML("<i>Truncation</i>"),
            widgets.HBox([
                widgets.IntSlider(
                    description='truncate left', min=0, max=max(0, n_datapoints-1), 
                    step=1, value=0, style={'description_width': '120px'}
                ),
                widgets.IntSlider(
                    description='truncate right', min=1, max=n_datapoints, 
                    step=1, value=n_datapoints, style={'description_width': '120px'}
                )
            ]),
            widgets.HTML("<i>Error Bar Filtering</i>"),
            widgets.HBox([
                widgets.Checkbox(description='enable filtering', value=True),
                widgets.Checkbox(description='remove singles', value=True),
                widgets.Checkbox(description='remove consecutives', value=True)
            ]),
            widgets.HBox([
                widgets.FloatSlider(
                    description='threshold', min=0.0, max=1.0, step=0.01, value=0.3,
                    style={'description_width': '120px'}
                ),
                widgets.IntSlider(
                    description='consecutive', min=1, max=10, step=1, value=3,
                    style={'description_width': '120px'}
                ),
                widgets.FloatSlider(
                    description='q start trunc', min=0.0, max=1.0, step=0.01, value=0.1,
                    style={'description_width': '120px'}
                )
            ])
        ])
        
        # Prediction section
        prediction_section = widgets.VBox([
            widgets.HTML("<h4>Prediction Settings</h4>"),
            widgets.HBox([
                widgets.Checkbox(description='polish prediction', value=True),
                widgets.Checkbox(description='use sigmas for polishing', value=True)
            ])
        ])
        
        # Plotting section
        plotting_section = widgets.VBox([
            widgets.HTML("<h4>Plotting Options</h4>"),
            widgets.HBox([
                widgets.Checkbox(description='show error bars', value=True),
                widgets.Checkbox(description='show q-resolution', value=False),
                widgets.Checkbox(description='log x-axis', value=False),
                widgets.Checkbox(description='plot SLD profile', value=True)
            ]),
            widgets.HBox([
                widgets.FloatText(
                    description='SLD pad left', value=0.2, step=0.1,
                    style={'description_width': '120px'}
                ),
                widgets.FloatText(
                    description='SLD pad right', value=1.1, step=0.1,
                    style={'description_width': '120px'}
                )
            ])
        ])
        
        # Color section
        color_section = widgets.VBox([
            widgets.HTML("<h4>Colors</h4>"),
            widgets.HBox([
                widgets.ColorPicker(description='exp color', value='#0000FF'),  # blue
                widgets.ColorPicker(description='error color', value='#800080'),  # purple
            ]),
            widgets.HBox([
                widgets.ColorPicker(description='pred color', value='#FF0000'),  # red
                widgets.ColorPicker(description='polished color', value='#FFA500'),  # orange
            ]),
            widgets.HBox([
                widgets.ColorPicker(description='SLD pred color', value='#FF0000'),  # red
                widgets.ColorPicker(description='SLD pol color', value='#FFA500'),  # orange
            ])
        ])
        
        # Computation section
        compute_section = widgets.VBox([
            widgets.HTML("<h4>Computation</h4>"),
            widgets.HBox([
                widgets.Checkbox(description='calc curve', value=True),
                widgets.Checkbox(description='calc pred SLD', value=True),
                widgets.Checkbox(description='calc polished SLD', value=True)
            ])
        ])
        
        return widgets.VBox([
            preprocessing_section,
            prediction_section, 
            plotting_section,
            color_section,
            compute_section
        ])
    
    def _extract_widget_settings(self, priors_section, controls_section):
        """Extract all current widget settings into a dictionary"""
        settings = {}
        
        # Extract prior bounds from stored sliders (exact replication of original)
        def _current_priors():
            return np.array([s.value for s in self._priors_sliders], dtype=np.float32)
        
        settings['prior_bounds'] = _current_priors()
        
        # Extract settings by finding widgets with specific descriptions
        widget_map = {
            'truncate_left': ('truncate left', 'value'),
            'truncate_right': ('truncate right', 'value'),
            'enable_filtering': ('enable filtering', 'value'),
            'filter_remove_singles': ('remove singles', 'value'),
            'filter_remove_consecutives': ('remove consecutives', 'value'),
            'filter_threshold': ('threshold', 'value'),
            'filter_consecutive': ('consecutive', 'value'),
            'filter_q_start_trunc': ('q start trunc', 'value'),
            'polish_prediction': ('polish prediction', 'value'),
            'use_sigmas_for_polishing': ('use sigmas for polishing', 'value'),
            'show_error_bars': ('show error bars', 'value'),
            'show_q_resolution': ('show q-resolution', 'value'),
            'log_x_axis': ('log x-axis', 'value'),
            'plot_sld_profile': ('plot SLD profile', 'value'),
            'sld_pad_left': ('SLD pad left', 'value'),
            'sld_pad_right': ('SLD pad right', 'value'),
            'exp_color': ('exp color', 'value'),
            'exp_errcolor': ('error color', 'value'),
            'pred_color': ('pred color', 'value'),
            'pol_color': ('polished color', 'value'),
            'sld_pred_color': ('SLD pred color', 'value'),
            'sld_pol_color': ('SLD pol color', 'value'),
            'calc_pred_curve': ('calc curve', 'value'),
            'calc_pred_sld': ('calc pred SLD', 'value'),
            'calc_pol_sld': ('calc polished SLD', 'value'),
        }
        
        for setting_name, (description, attr) in widget_map.items():
            found_widgets = self._find_widgets_by_description(controls_section, [description])
            
            if found_widgets:
                settings[setting_name] = getattr(found_widgets[0], attr)
            else:
                # Set reasonable defaults if widget not found
                defaults = {
                    'truncate_left': 0,
                    'truncate_right': 100,
                    'enable_filtering': True,
                    'filter_remove_singles': True,
                    'filter_remove_consecutives': True,
                    'filter_threshold': 0.3,
                    'filter_consecutive': 3,
                    'filter_q_start_trunc': 0.1,
                    'polish_prediction': True,
                    'use_sigmas_for_polishing': True,
                    'show_error_bars': True,
                    'show_q_resolution': False,
                    'log_x_axis': False,
                    'plot_sld_profile': True,
                    'sld_pad_left': 0.2,
                    'sld_pad_right': 1.1,
                    'exp_color': '#0000FF',
                    'exp_errcolor': '#800080',
                    'pred_color': '#FF0000',
                    'pol_color': '#FFA500',
                    'sld_pred_color': '#FF0000',
                    'sld_pol_color': '#FFA500',
                    'calc_pred_curve': True,
                    'calc_pred_sld': True,
                    'calc_pol_sld': True,
                }
                settings[setting_name] = defaults.get(setting_name)
        
        return settings
    
    def _find_widgets_by_description(self, container, descriptions):
        """Helper to find widgets by their description"""
        found_widgets = []
        
        def search_widget(widget):
            if hasattr(widget, 'description') and widget.description in descriptions:
                found_widgets.append(widget)
            if hasattr(widget, 'children'):
                for child in widget.children:
                    search_widget(child)
        
        search_widget(container)
        return found_widgets


# Convenience functions for backward compatibility and ease of use
def create_basic_widget(model, reflectivity_curve, interactive_plots=True, **kwargs):
    """
    Convenience function to create and display a basic prediction widget
    
    Args:
        model: EasyInferenceModel instance
        reflectivity_curve: Experimental reflectivity data
        interactive_plots: Enable interactive plotting features (default: True)
        **kwargs: Additional arguments passed to basic_predict_widget
        
    Returns:
        ReflectJupyterWidget: The widget instance
    """
    widget = ReflectJupyterWidget(model, interactive_plots=interactive_plots)
    widget.basic_predict_widget(reflectivity_curve, **kwargs)
    return widget


def create_advanced_widget(model, reflectivity_curve, q_values, interactive_plots=True, **kwargs):
    """
    Convenience function to create and display an advanced prediction widget
    
    Args:
        model: EasyInferenceModel instance
        reflectivity_curve: Experimental reflectivity data
        q_values: Momentum transfer values
        interactive_plots: Enable interactive plotting features (default: True)
        **kwargs: Additional arguments passed to advanced_predict_widget
        
    Returns:
        ReflectJupyterWidget: The widget instance
    """
    widget = ReflectJupyterWidget(model, interactive_plots=interactive_plots)
    widget.advanced_predict_widget(reflectivity_curve, q_values, **kwargs)
    return widget