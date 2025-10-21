"""
Jupyter Widget Components for Reflectorch

This module contains reusable widget components that can be composed
to create different interfaces for reflectometry analysis.

Components:
    - ParameterTable: Interactive parameter table with sliders and results
    - PreprocessingControls: Data preprocessing options
    - PredictionControls: Prediction and computation settings
    - PlottingControls: Plotting and visualization options
    - AdditionalParametersControls: Controls for additional parameters like Q resolution
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import ipywidgets as widgets


class ParameterTable:
    """
    Interactive parameter table with sliders and result displays
    
    Features:
    - Structured table layout with aligned columns
    - Real-time result updates after predictions
    - Automatic slider validation
    - Professional styling
    """
    
    def __init__(self, param_labels: List[str], min_bounds: np.ndarray, 
                 max_bounds: np.ndarray, max_deltas: np.ndarray, 
                 initial_bounds: Optional[np.ndarray] = None,
                 additional_params_controls: Optional['AdditionalParametersControls'] = None,
                 predict_button: Optional[widgets.Button] = None):
        """
        Initialize parameter table
        
        Args:
            param_labels: List of parameter names
            min_bounds: Minimum values for each parameter
            max_bounds: Maximum values for each parameter  
            max_deltas: Maximum allowed range for each parameter
            initial_bounds: Initial bounds, shape (n_params, 2)
            additional_params_controls: Optional additional parameters component
            predict_button: Optional predict button to include in the table header
        """
        self.param_labels = param_labels
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.max_deltas = max_deltas
        self.sliders = []  # Store range sliders
        self.min_inputs = []  # Store min bound inputs
        self.max_inputs = []  # Store max bound inputs
        self.result_displays = {}
        self.additional_params_controls = additional_params_controls
        self.predict_button = predict_button
        
        self.widget = self._create_table(initial_bounds)
    
    def _create_table(self, initial_bounds: Optional[np.ndarray] = None) -> widgets.VBox:
        """Create the parameter table widget"""
        init_pb = np.array(initial_bounds) if initial_bounds is not None else None
        
        # Add custom CSS for styling
        custom_style = widgets.HTML("""
        <style>
            .widget-float-text input {
                text-align: center;
            }
            .widget-inline-hbox .widget-readout {
                margin-left: 20px !important;
            }
            .widget-slider .ui-slider {
                width: 280px !important;
            }
        </style>
        """)
        
        # Create header row
        header = widgets.HBox([
            widgets.HTML("<b>Parameter</b>", layout=widgets.Layout(width='150px')),
            widgets.HTML("<b>Prior Bounds</b>", layout=widgets.Layout(width='300px')),
            widgets.HTML("<b></b>", layout=widgets.Layout(width='50px')),
            widgets.HTML("<b></b>", layout=widgets.Layout(width='50px')),
            widgets.HTML("<b>Predicted</b>", layout=widgets.Layout(width='100px')),
            widgets.HTML("<b>Polished</b>", layout=widgets.Layout(width='100px')),
            widgets.HTML("<b>Uncertainty</b>", layout=widgets.Layout(width='100px'))
        ], layout=widgets.Layout(margin='5px 0px', align_items='center'))
        
        # Create parameter rows
        parameter_rows = []
        
        if not self.param_labels:
            # No parameters available - show message
            no_params_message = widgets.HTML(
                value="<i>No model loaded. Please load a model from the Models tab to see parameters.</i>",
                layout=widgets.Layout(margin='20px 5px')
            )
            parameter_rows.append(no_params_message)
        
        for i, label in enumerate(self.param_labels):
            init_min = float(init_pb[i, 0]) if init_pb is not None else float(self.min_bounds[i])
            init_max = float(init_pb[i, 1]) if init_pb is not None else float(min(self.min_bounds[i] + self.max_deltas[i], self.max_bounds[i]))
            
            # Parameter label
            param_label = widgets.HTML(
                value=f"<b>{label}</b>",
                layout=widgets.Layout(width='150px', display='flex', align_items='center', justify_content='flex-start')
            )
            
            # Range slider - updates continuously
            slider = widgets.FloatRangeSlider(
                value=[init_min, init_max],
                min=float(self.min_bounds[i]),
                max=float(self.max_bounds[i]),
                step=0.01,
                layout=widgets.Layout(width='180px'),
                readout=False,  # Hide readout since we have input boxes
                continuous_update=True,  # Update instantly while dragging
                style={'description_width': '0px', 'handle_color': '#2196F3'}
            )
            
            # Min bound input - updates continuously
            min_input = widgets.FloatText(
                value=round(init_min, 2),
                step=0.01,
                layout=widgets.Layout(width='65px'),
                continuous_update=True  # Update on every keystroke
            )
            
            # Max bound input - updates continuously
            max_input = widgets.FloatText(
                value=round(init_max, 2),
                step=0.01,
                layout=widgets.Layout(width='65px'),
                continuous_update=True  # Update on every keystroke
            )
            
            # Result displays
            predicted_display = widgets.HTML(
                value="<i>-</i>",
                layout=widgets.Layout(width='100px', display='flex', align_items='center', justify_content='flex-start')
            )
            polished_display = widgets.HTML(
                value="<i>-</i>",
                layout=widgets.Layout(width='100px', display='flex', align_items='center', justify_content='flex-start')
            )
            uncertainty_display = widgets.HTML(
                value="<i>-</i>",
                layout=widgets.Layout(width='100px', display='flex', align_items='center', justify_content='flex-start')
            )
            
            # Store references for updating results
            self.result_displays[i] = {
                'predicted': predicted_display,
                'polished': polished_display,
                'uncertainty': uncertainty_display
            }
            
            # Add synchronization between slider and inputs
            self._add_widget_synchronization(slider, min_input, max_input, i)
            self.sliders.append(slider)
            self.min_inputs.append(min_input)
            self.max_inputs.append(max_input)
            
            # Create row layout with vertical alignment
            row = widgets.HBox([
                param_label,
                slider,
                min_input,
                max_input,
                predicted_display,
                polished_display,
                uncertainty_display
            ], layout=widgets.Layout(margin='5px 0px', align_items='center'))
            
            parameter_rows.append(row)

        self.param_title = widgets.HTML("<h4>Parameter Configuration</h4>")
        
        # Create title row with optional predict button
        if self.predict_button is not None:
            title_row = widgets.HBox([
                self.param_title,
                self.predict_button
            ], layout=widgets.Layout(justify_content='space-between', align_items='center'))
        else:
            title_row = self.param_title
        
        # Create the main parameter table
        main_table = widgets.VBox([
            custom_style,
            title_row,
            header,
            widgets.HTML("<hr style='margin: 5px 0px;'>"),
            *parameter_rows
        ])
        
        # Create the complete widget with optional additional parameters
        table_components = [main_table]
        
        # Add additional parameters section if available
        if (self.additional_params_controls is not None and 
            self.additional_params_controls.additional_sliders):
            table_components.extend([
                widgets.HTML("<br>"),
                self.additional_params_controls.widget
            ])
        
        return widgets.VBox(table_components)
    
    def _add_widget_synchronization(self, slider: widgets.FloatRangeSlider, 
                                     min_input: widgets.FloatText, 
                                     max_input: widgets.FloatText, 
                                     param_idx: int):
        """Add synchronization between slider and input boxes with validation"""
        max_width = float(self.max_deltas[param_idx])
        global_min = float(self.min_bounds[param_idx])
        global_max = float(self.max_bounds[param_idx])
        
        # Flag to prevent infinite update loops
        updating = {'active': False}
        
        def validate_and_clamp(min_val, max_val, source='slider'):
            """Apply all constraints and return valid (min, max) pair
            
            Args:
                min_val: Minimum value
                max_val: Maximum value
                source: Which widget triggered the change ('slider', 'min', or 'max')
            """
            # Clamp to global bounds first
            min_val = max(global_min, min(min_val, global_max))
            max_val = max(global_min, min(max_val, global_max))
            
            # Ensure min < max with proper handling based on which widget changed
            min_step = 0.01  # Minimum separation between min and max
            
            if min_val >= max_val:
                if source == 'min':
                    # User changed min, adjust min to be less than max
                    min_val = max_val - min_step
                    # If this pushes min below global_min, adjust max instead
                    if min_val < global_min:
                        min_val = global_min
                        max_val = min_val + min_step
                        # If max exceeds global_max, we have a problem
                        if max_val > global_max:
                            max_val = global_max
                            min_val = max_val - min_step
                elif source == 'max':
                    # User changed max, adjust max to be greater than min
                    max_val = min_val + min_step
                    # If this pushes max above global_max, adjust min instead
                    if max_val > global_max:
                        max_val = global_max
                        min_val = max_val - min_step
                        # If min goes below global_min, we have a problem
                        if min_val < global_min:
                            min_val = global_min
                            max_val = min_val + min_step
                else:  # source == 'slider'
                    # Slider changed, just ensure there's a minimum gap
                    avg = (min_val + max_val) / 2
                    min_val = avg - min_step / 2
                    max_val = avg + min_step / 2
            
            # Ensure range doesn't exceed max_width
            if max_val - min_val > max_width:
                if source == 'min':
                    # User changed min, keep max fixed and adjust min
                    min_val = max_val - max_width
                    if min_val < global_min:
                        min_val = global_min
                        max_val = min_val + max_width
                elif source == 'max':
                    # User changed max, keep min fixed and adjust max
                    max_val = min_val + max_width
                    if max_val > global_max:
                        max_val = global_max
                        min_val = max_val - max_width
                else:  # source == 'slider'
                    # Slider changed, adjust based on center
                    center = (min_val + max_val) / 2
                    min_val = max(global_min, center - max_width / 2)
                    max_val = min(global_max, min_val + max_width)
                    if max_val - min_val > max_width:
                        max_val = min(global_max, center + max_width / 2)
                        min_val = max(global_min, max_val - max_width)
            
            return min_val, max_val
        
        def sync_all_widgets(min_val, max_val, source='slider'):
            """Update all three widgets if values changed"""
            if updating['active']:
                return
            
            updating['active'] = True
            
            # Validate values with source information
            min_val, max_val = validate_and_clamp(min_val, max_val, source)
            
            # Round to 2 decimal places for display
            min_val = round(min_val, 2)
            max_val = round(max_val, 2)
            
            # Update all widgets if needed
            if slider.value != (min_val, max_val):
                slider.value = (min_val, max_val)
            if min_input.value != min_val:
                min_input.value = min_val
            if max_input.value != max_val:
                max_input.value = max_val
            
            updating['active'] = False
        
        def on_slider_change(change):
            min_val, max_val = change['new']
            sync_all_widgets(min_val, max_val, source='slider')
        
        def on_min_input_change(change):
            min_val = change['new']
            max_val = max_input.value
            sync_all_widgets(min_val, max_val, source='min')
        
        def on_max_input_change(change):
            min_val = min_input.value
            max_val = change['new']
            sync_all_widgets(min_val, max_val, source='max')
        
        # Attach observers
        slider.observe(on_slider_change, names='value')
        min_input.observe(on_min_input_change, names='value')
        max_input.observe(on_max_input_change, names='value')
    
    def get_prior_bounds(self) -> np.ndarray:
        """Get current prior bounds from input boxes"""
        if not self.min_inputs or not self.max_inputs:
            return np.array([], dtype=np.float32).reshape(0, 2)
        return np.array([[min_inp.value, max_inp.value] 
                        for min_inp, max_inp in zip(self.min_inputs, self.max_inputs)], 
                       dtype=np.float32)
    
    def update_results(self, prediction_result: Dict[str, Any]):
        """Update parameter result displays"""
        if not prediction_result:
            return
        
        predicted_params = prediction_result.get('predicted_params_array', [])
        polished_params = prediction_result.get('polished_params_array', None)
        error_bars = prediction_result.get('polished_params_error_array', None)
        
        for i, displays in self.result_displays.items():
            # Update predicted value
            if i < len(predicted_params):
                pred_val = predicted_params[i]
                displays['predicted'].value = f"{pred_val:.2f}"
            else:
                displays['predicted'].value = "<i>-</i>"
            
            # Update polished value
            if polished_params is not None and i < len(polished_params):
                pol_val = polished_params[i]
                displays['polished'].value = f"{pol_val:.2f}"
            else:
                displays['polished'].value = "<i>-</i>"
            
            # Update uncertainty/error bars value
            if error_bars is not None and i < len(error_bars):
                err_val = error_bars[i]
                displays['uncertainty'].value = f"±{err_val:.2f}"
            else:
                displays['uncertainty'].value = "<i>-</i>"


class PreprocessingControls:
    """Data preprocessing controls for the widget"""
    
    def __init__(self, n_datapoints: int):
        """
        Initialize preprocessing controls
        
        Args:
            n_datapoints: Number of data points in the dataset
        """
        self.n_datapoints = n_datapoints
        self.widget = self._create_controls()
    
    def _create_controls(self) -> widgets.VBox:
        """Create preprocessing controls widget"""
        return widgets.VBox([
            widgets.HTML("<h4>Data Preprocessing</h4>"),
            
            # Truncation section
            widgets.HTML("<h5>Data Truncation</h5>"),
            widgets.HTML("<i>Specify which data points to include in the analysis</i>"),
            widgets.VBox([
                widgets.IntSlider(
                    description='Left index:', min=0, max=max(0, self.n_datapoints-1), 
                    step=1, value=0,
                ),
                widgets.IntSlider(
                    description='Right index:', min=1, max=self.n_datapoints, 
                    step=1, value=self.n_datapoints,
                )
            ]),
            
            widgets.HTML("<br>"),
            
            # Error bar filtering section  
            widgets.HTML("<h5>Error Bar Filtering</h5>"),
            widgets.HTML("<i>Filter out unreliable data points based on error bars</i>"),
            widgets.VBox([
                widgets.Checkbox(description='Enable filtering', value=True),
                widgets.Checkbox(description='Remove singles', value=True),
                widgets.Checkbox(description='Remove consecutives', value=True)
            ]),
            widgets.VBox([
                widgets.FloatSlider(
                    description='Threshold:', min=0.0, max=1.0, step=0.01, value=0.3,
                ),
                widgets.IntSlider(
                    description='Consecutive:', min=1, max=10, step=1, value=3,
                ),
                widgets.FloatSlider(
                    description='Q start trunc:', min=0.0, max=1.0, step=0.01, value=0.1,
                )
            ])
        ])


class PredictionControls:
    """Prediction and computation settings controls"""
    
    def __init__(self):
        self.widget = self._create_controls()
    
    def _create_controls(self) -> widgets.VBox:
        """Create prediction controls widget"""
        return widgets.VBox([
            widgets.HTML("<h4>Prediction & Computation Settings</h4>"),
            
            # Prediction settings
            widgets.HTML("<h5>Prediction Options</h5>"),
            widgets.VBox([
                widgets.Checkbox(description='Polish prediction', value=True),
                widgets.Checkbox(description='Use sigmas for polishing', value=True)
            ]),
            
            widgets.HTML("<br>"),
            
            # Computation settings
            widgets.HTML("<h5>Computation Settings</h5>"),
            widgets.HTML("<i>Choose what to calculate during prediction</i>"),
            widgets.VBox([
                widgets.Checkbox(description='Calculate curve', value=True),
                widgets.Checkbox(description='Calculate pred SLD', value=True),
                widgets.Checkbox(description='Calculate polished SLD', value=True)
            ])
        ])


class AdditionalParametersControls:
    """
    Controls for additional parameters that are not part of prior bounds
    
    These are fixed input parameters like Q resolution that are passed
    separately to the inference model.
    """
    
    def __init__(self, inference_model=None):
        """
        Initialize additional parameters controls
        
        Args:
            inference_model: The inference model to check for additional parameters (optional)
        """
        self.inference_model = inference_model
        self.additional_sliders = {}
        self.widget = self._create_controls()
    
    def _create_controls(self) -> widgets.VBox:
        """Create additional parameters controls widget"""
        controls = []
        
        # Check if model has smearing (Q resolution)
        if (self.inference_model is not None and
            hasattr(self.inference_model, 'trainer') and 
            hasattr(self.inference_model.trainer, 'loader') and
            hasattr(self.inference_model.trainer.loader, 'smearing') and
            self.inference_model.trainer.loader.smearing is not None):
            
            q_res_min = float(self.inference_model.trainer.loader.smearing.sigma_min)
            q_res_max = float(self.inference_model.trainer.loader.smearing.sigma_max)
            
            # Q resolution slider
            q_res_slider = widgets.FloatSlider(
                description='Q resolution (dq/q):',
                min=q_res_min,
                max=q_res_max,
                step=0.001,
                value=(q_res_min + q_res_max) / 2,  # Default to middle value
                readout_format='.4f',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='400px')
            )
            
            self.additional_sliders['q_resolution'] = q_res_slider
            controls.append(q_res_slider)
            controls.append(widgets.HTML("<br>"))
                
        return widgets.VBox(controls)
    
    def get_additional_params(self) -> Dict[str, float]:
        """Get current values of additional parameters"""
        return {name: slider.value for name, slider in self.additional_sliders.items()}


class PlottingControls:
    """Plotting and visualization controls"""
    
    def __init__(self):
        self.widget = self._create_controls()
    
    def _create_controls(self) -> widgets.VBox:
        """Create plotting controls widget"""
        return widgets.VBox([
            widgets.HTML("<h4>Plotting Settings</h4>"),
            
            # Display options
            widgets.HTML("<h5>Display Options</h5>"),
            widgets.VBox([
                widgets.HBox([
                    widgets.Checkbox(description='Show error bars', value=True),
                    widgets.Checkbox(description='Show q-resolution', value=False),
                ]),
                widgets.HBox([
                    widgets.Checkbox(description='Log x-axis', value=False),
                    widgets.Checkbox(description='Plot SLD profile', value=True)
                ])
            ]),
            
            # SLD padding
            widgets.HTML("<h5>SLD Profile Settings</h5>"),
            widgets.HBox([
                widgets.FloatText(
                    description='Left padding:', value=0.2, step=0.1,
                    style={'description_width': '100px'}, layout=widgets.Layout(width='200px')
                ),
                widgets.FloatText(
                    description='Right padding:', value=1.1, step=0.1,
                    style={'description_width': '100px'}, layout=widgets.Layout(width='200px')
                )
            ]),
            
            widgets.HTML("<br>"),
            
            # Color customization
            widgets.HTML("<h5>Color Customization</h5>"),
            widgets.VBox([
                widgets.ColorPicker(description='Data color:', value='#0000FF'),
                widgets.ColorPicker(description='Error bars:', value='#800080')
            ]),
            
            widgets.VBox([
                widgets.ColorPicker(description='Prediction:', value='#FF0000'),
                widgets.ColorPicker(description='Polished:', value='#FFA500')
            ]),
            
            widgets.VBox([
                widgets.ColorPicker(description='SLD pred:', value='#FF0000'),
                widgets.ColorPicker(description='SLD polish:', value='#FFA500')
            ])
        ])


class WidgetSettingsExtractor:
    """Utility class to extract settings from widget components"""
    
    @staticmethod
    def extract_settings(parameter_table: ParameterTable, 
                        preprocessing: PreprocessingControls,
                        prediction: PredictionControls, 
                        plotting: PlottingControls,
                        additional_params: Optional['AdditionalParametersControls'] = None,
                        data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract all current widget settings into a dictionary
        
        Args:
            parameter_table: Parameter table component
            preprocessing: Preprocessing controls component
            prediction: Prediction controls component
            plotting: Plotting controls component
            additional_params: Additional parameters controls component (optional)
            data: Data dictionary with fallback values (optional)
            
        Returns:
            Dictionary containing all current settings ready for preprocess_and_predict
        """
        settings = {}
        data = data or {}
        
        # Get prior bounds from parameter table
        settings['prior_bounds'] = parameter_table.get_prior_bounds()
        
        # Handle data parameters with widget override or fallback to data
        additional_param_values = {}
        if additional_params is not None:
            additional_param_values = additional_params.get_additional_params()
        
        # Set q_resolution: use widget value if available, otherwise data, otherwise None
        settings['q_resolution'] = additional_param_values.get('q_resolution', data.get('q_resolution'))
        
        # Set ambient_sld: use widget value if available, otherwise data, otherwise None  
        settings['ambient_sld'] = additional_param_values.get('ambient_sld', data.get('ambient_sld'))
        
        # Set other data parameters that are always from data
        settings['reflectivity_curve'] = data.get('reflectivity_curve')
        settings['q_values'] = data.get('q_values') 
        settings['sigmas'] = data.get('sigmas')
        
        # Set fixed prediction parameters
        settings['clip_prediction'] = True  # Always clip predictions
        
        # Find and extract settings from all components
        all_widgets = [preprocessing.widget, prediction.widget, plotting.widget]
        
        # Map widget descriptions to preprocess_and_predict parameter names
        widget_map = {
            # Preprocessing parameters (match preprocess_and_predict signature)
            'truncate_index_left': ('Left index:', 'value'),
            'truncate_index_right': ('Right index:', 'value'),
            'enable_error_bars_filtering': ('Enable filtering', 'value'),
            'filter_remove_singles': ('Remove singles', 'value'),
            'filter_remove_consecutives': ('Remove consecutives', 'value'),
            'filter_threshold': ('Threshold:', 'value'),
            'filter_consecutive': ('Consecutive:', 'value'),
            'filter_q_start_trunc': ('Q start trunc:', 'value'),
            
            # Prediction parameters (match preprocess_and_predict signature)
            'polish_prediction': ('Polish prediction', 'value'),
            'use_sigmas_for_polishing': ('Use sigmas for polishing', 'value'),
            'calc_pred_curve': ('Calculate curve', 'value'),
            'calc_pred_sld_profile': ('Calculate pred SLD', 'value'),
            'calc_polished_sld_profile': ('Calculate polished SLD', 'value'),
            'sld_profile_padding_left': ('Left padding:', 'value'),
            'sld_profile_padding_right': ('Right padding:', 'value'),
            
            # Plotting parameters (not passed to preprocess_and_predict, kept for plotting)
            'show_error_bars': ('Show error bars', 'value'),
            'show_q_resolution': ('Show q-resolution', 'value'),
            'log_x_axis': ('Log x-axis', 'value'),
            'plot_sld_profile': ('Plot SLD profile', 'value'),
            'exp_color': ('Data color:', 'value'),
            'exp_errcolor': ('Error bars:', 'value'),
            'pred_color': ('Prediction:', 'value'),
            'pol_color': ('Polished:', 'value'),
            'sld_pred_color': ('SLD pred:', 'value'),
            'sld_pol_color': ('SLD polish:', 'value'),
        }
        
        for setting_name, (description, attr) in widget_map.items():
            found_widgets = []
            for widget in all_widgets:
                found_widgets.extend(WidgetSettingsExtractor._find_widgets_by_description(widget, [description]))
            
            if found_widgets:
                settings[setting_name] = getattr(found_widgets[0], attr)
            else:
                # Set reasonable defaults
                defaults = {
                    # Preprocessing parameters
                    'truncate_index_left': 0,
                    'truncate_index_right': 100,
                    'enable_error_bars_filtering': True,
                    'filter_remove_singles': True,
                    'filter_remove_consecutives': True,
                    'filter_threshold': 0.3,
                    'filter_consecutive': 3,
                    'filter_q_start_trunc': 0.1,
                    
                    # Prediction parameters
                    'polish_prediction': True,
                    'use_sigmas_for_polishing': True,
                    'calc_pred_curve': True,
                    'calc_pred_sld_profile': True,
                    'calc_polished_sld_profile': True,
                    'sld_profile_padding_left': 0.2,
                    'sld_profile_padding_right': 1.1,
                    
                    # Plotting parameters
                    'show_error_bars': True,
                    'show_q_resolution': False,
                    'log_x_axis': False,
                    'plot_sld_profile': True,
                    'exp_color': '#0000FF',
                    'exp_errcolor': '#800080',
                    'pred_color': '#FF0000',
                    'pol_color': '#FFA500',
                    'sld_pred_color': '#FF0000',
                    'sld_pol_color': '#FFA500',
                }
                settings[setting_name] = defaults.get(setting_name)
        
        return settings
    
    @staticmethod
    def separate_settings(settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Separate settings into prediction parameters and plotting parameters
        
        Args:
            settings: Complete settings dictionary from extract_settings
            
        Returns:
            Tuple of (prediction_params, plotting_params)
        """
        # Parameters that go to preprocess_and_predict
        prediction_param_names = {
            'reflectivity_curve', 'q_values', 'prior_bounds', 'sigmas', 'q_resolution', 'ambient_sld',
            'clip_prediction', 'polish_prediction', 'use_sigmas_for_polishing',
            'calc_pred_curve', 'calc_pred_sld_profile', 'calc_polished_sld_profile',
            'sld_profile_padding_left', 'sld_profile_padding_right',
            'truncate_index_left', 'truncate_index_right', 'enable_error_bars_filtering',
            'filter_threshold', 'filter_remove_singles', 'filter_remove_consecutives',
            'filter_consecutive', 'filter_q_start_trunc'
        }
        
        # Parameters used for plotting
        plotting_param_names = {
            'show_error_bars', 'show_q_resolution', 'log_x_axis', 'plot_sld_profile',
            'exp_color', 'exp_errcolor', 'pred_color', 'pol_color', 
            'sld_pred_color', 'sld_pol_color'
        }
        
        prediction_params = {k: v for k, v in settings.items() if k in prediction_param_names}
        plotting_params = {k: v for k, v in settings.items() if k in plotting_param_names}
        
        return prediction_params, plotting_params
    
    @staticmethod
    def _find_widgets_by_description(container, descriptions):
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


