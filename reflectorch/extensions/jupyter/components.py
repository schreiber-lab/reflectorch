"""
Jupyter Widget Components for Reflectorch

This module contains reusable widget components that can be composed
to create different interfaces for reflectometry analysis.

Components:
    - ParameterTable: Interactive parameter table with sliders and results
    - PreprocessingControls: Data preprocessing options
    - PredictionControls: Prediction and computation settings
    - PlottingControls: Plotting and visualization options
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
                 initial_bounds: Optional[np.ndarray] = None):
        """
        Initialize parameter table
        
        Args:
            param_labels: List of parameter names
            min_bounds: Minimum values for each parameter
            max_bounds: Maximum values for each parameter  
            max_deltas: Maximum allowed range for each parameter
            initial_bounds: Initial bounds, shape (n_params, 2)
        """
        self.param_labels = param_labels
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.max_deltas = max_deltas
        self.sliders = []
        self.result_displays = {}
        
        self.widget = self._create_table(initial_bounds)
    
    def _create_table(self, initial_bounds: Optional[np.ndarray] = None) -> widgets.VBox:
        """Create the parameter table widget"""
        init_pb = np.array(initial_bounds) if initial_bounds is not None else None
        
        # Create header row
        header = widgets.HBox([
            widgets.HTML("<b>Parameter</b>", layout=widgets.Layout(width='140px')),
            widgets.HTML("<b>Prior Bounds</b>", layout=widgets.Layout(width='280px')),
            widgets.HTML("<b>Predicted</b>", layout=widgets.Layout(width='100px')),
            widgets.HTML("<b>Polished</b>", layout=widgets.Layout(width='100px'))
        ], layout=widgets.Layout(margin='5px 0px'))
        
        # Create parameter rows
        parameter_rows = []
        
        for i, label in enumerate(self.param_labels):
            init_min = float(init_pb[i, 0]) if init_pb is not None else float(self.min_bounds[i])
            init_max = float(init_pb[i, 1]) if init_pb is not None else float(min(self.min_bounds[i] + self.max_deltas[i], self.max_bounds[i]))
            
            # Parameter label
            param_label = widgets.HTML(
                value=f"<b>{label}</b>",
                layout=widgets.Layout(width='140px', height='35px', margin='2px 5px')
            )
            
            # Prior bounds slider
            slider = widgets.FloatRangeSlider(
                value=[init_min, init_max],
                min=float(self.min_bounds[i]),
                max=float(self.max_bounds[i]),
                step=0.01,
                layout=widgets.Layout(width='280px', height='35px'),
                readout_format='.3f',
                style={'description_width': '0px'}
            )
            
            # Result displays
            predicted_display = widgets.HTML(
                value="<i>-</i>",
                layout=widgets.Layout(width='100px', height='35px', margin='2px 5px')
            )
            polished_display = widgets.HTML(
                value="<i>-</i>",
                layout=widgets.Layout(width='100px', height='35px', margin='2px 5px')
            )
            
            # Store references for updating results
            self.result_displays[i] = {
                'predicted': predicted_display,
                'polished': polished_display
            }
            
            # Add slider validation
            self._add_slider_validation(slider, float(self.max_deltas[i]))
            self.sliders.append(slider)
            
            # Create row layout
            row = widgets.HBox([
                param_label,
                slider,
                predicted_display,
                polished_display
            ], layout=widgets.Layout(margin='2px 0px'))
            
            parameter_rows.append(row)
        
        # Create the complete table
        return widgets.VBox([
            widgets.HTML("<h4>Parameter Configuration</h4>"),
            header,
            widgets.HTML("<hr style='margin: 5px 0px;'>"),
            *parameter_rows
        ])
    
    def _add_slider_validation(self, slider: widgets.FloatRangeSlider, max_width: float):
        """Add validation to constrain slider range"""
        def validate_range(change):
            a, b = change['new']
            if b - a > max_width:
                oa, ob = change['old']
                if abs(oa - a) > abs(ob - b):
                    b = a + max_width
                else:
                    a = b - max_width
                slider.value = (a, b)
        
        slider.observe(validate_range, names='value')
    
    def get_prior_bounds(self) -> np.ndarray:
        """Get current prior bounds from sliders"""
        return np.array([s.value for s in self.sliders], dtype=np.float32)
    
    def update_results(self, prediction_result: Dict[str, Any]):
        """Update parameter result displays"""
        if not prediction_result:
            return
        
        predicted_params = prediction_result.get('predicted_params_array', [])
        polished_params = prediction_result.get('polished_params_array', None)
        
        for i, displays in self.result_displays.items():
            # Update predicted value
            if i < len(predicted_params):
                pred_val = predicted_params[i]
                displays['predicted'].value = f"<b>{pred_val:.3f}</b>"
            else:
                displays['predicted'].value = "<i>-</i>"
            
            # Update polished value
            if polished_params is not None and i < len(polished_params):
                pol_val = polished_params[i]
                displays['polished'].value = f"<b>{pol_val:.3f}</b>"
            else:
                displays['polished'].value = "<i>-</i>"


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
            widgets.HBox([
                widgets.IntSlider(
                    description='Left index:', min=0, max=max(0, self.n_datapoints-1), 
                    step=1, value=0, style={'description_width': '100px'},
                    layout=widgets.Layout(width='300px')
                ),
                widgets.IntSlider(
                    description='Right index:', min=1, max=self.n_datapoints, 
                    step=1, value=self.n_datapoints, style={'description_width': '100px'},
                    layout=widgets.Layout(width='300px')
                )
            ]),
            
            widgets.HTML("<br>"),
            
            # Error bar filtering section  
            widgets.HTML("<h5>Error Bar Filtering</h5>"),
            widgets.HTML("<i>Filter out unreliable data points based on error bars</i>"),
            widgets.HBox([
                widgets.Checkbox(description='Enable filtering', value=True, style={'description_width': '120px'}),
                widgets.Checkbox(description='Remove singles', value=True, style={'description_width': '120px'}),
                widgets.Checkbox(description='Remove consecutives', value=True, style={'description_width': '140px'})
            ]),
            widgets.HBox([
                widgets.FloatSlider(
                    description='Threshold:', min=0.0, max=1.0, step=0.01, value=0.3,
                    style={'description_width': '100px'}, layout=widgets.Layout(width='250px')
                ),
                widgets.IntSlider(
                    description='Consecutive:', min=1, max=10, step=1, value=3,
                    style={'description_width': '100px'}, layout=widgets.Layout(width='200px')
                ),
                widgets.FloatSlider(
                    description='Q start trunc:', min=0.0, max=1.0, step=0.01, value=0.1,
                    style={'description_width': '100px'}, layout=widgets.Layout(width='250px')
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
            widgets.HBox([
                widgets.Checkbox(description='Polish prediction', value=True, style={'description_width': '140px'}),
                widgets.Checkbox(description='Use sigmas for polishing', value=True, style={'description_width': '170px'})
            ]),
            
            widgets.HTML("<br>"),
            
            # Computation settings
            widgets.HTML("<h5>Computation Settings</h5>"),
            widgets.HTML("<i>Choose what to calculate during prediction</i>"),
            widgets.HBox([
                widgets.Checkbox(description='Calculate curve', value=True, style={'description_width': '120px'}),
                widgets.Checkbox(description='Calculate pred SLD', value=True, style={'description_width': '140px'}),
                widgets.Checkbox(description='Calculate polished SLD', value=True, style={'description_width': '160px'})
            ])
        ])


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
            widgets.HBox([
                widgets.Checkbox(description='Show error bars', value=True, style={'description_width': '120px'}),
                widgets.Checkbox(description='Show q-resolution', value=False, style={'description_width': '130px'}),
                widgets.Checkbox(description='Log x-axis', value=False, style={'description_width': '100px'}),
                widgets.Checkbox(description='Plot SLD profile', value=True, style={'description_width': '130px'})
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
            widgets.HTML("<i>Experimental data colors</i>"),
            widgets.HBox([
                widgets.ColorPicker(description='Data color:', value='#0000FF', style={'description_width': '100px'}),
                widgets.ColorPicker(description='Error bars:', value='#800080', style={'description_width': '100px'})
            ]),
            
            widgets.HTML("<i>Prediction colors</i>"),
            widgets.HBox([
                widgets.ColorPicker(description='Prediction:', value='#FF0000', style={'description_width': '100px'}),
                widgets.ColorPicker(description='Polished:', value='#FFA500', style={'description_width': '100px'})
            ]),
            
            widgets.HTML("<i>SLD profile colors</i>"),
            widgets.HBox([
                widgets.ColorPicker(description='SLD pred:', value='#FF0000', style={'description_width': '100px'}),
                widgets.ColorPicker(description='SLD polish:', value='#FFA500', style={'description_width': '100px'})
            ])
        ])


class WidgetSettingsExtractor:
    """Utility class to extract settings from widget components"""
    
    @staticmethod
    def extract_settings(parameter_table: ParameterTable, 
                        preprocessing: PreprocessingControls,
                        prediction: PredictionControls, 
                        plotting: PlottingControls) -> Dict[str, Any]:
        """
        Extract all current widget settings into a dictionary
        
        Args:
            parameter_table: Parameter table component
            preprocessing: Preprocessing controls component
            prediction: Prediction controls component
            plotting: Plotting controls component
            
        Returns:
            Dictionary containing all current settings
        """
        settings = {}
        
        # Get prior bounds from parameter table
        settings['prior_bounds'] = parameter_table.get_prior_bounds()
        
        # Find and extract settings from all components
        all_widgets = [preprocessing.widget, prediction.widget, plotting.widget]
        
        widget_map = {
            'truncate_left': ('Left index:', 'value'),
            'truncate_right': ('Right index:', 'value'),
            'enable_filtering': ('Enable filtering', 'value'),
            'filter_remove_singles': ('Remove singles', 'value'),
            'filter_remove_consecutives': ('Remove consecutives', 'value'),
            'filter_threshold': ('Threshold:', 'value'),
            'filter_consecutive': ('Consecutive:', 'value'),
            'filter_q_start_trunc': ('Q start trunc:', 'value'),
            'polish_prediction': ('Polish prediction', 'value'),
            'use_sigmas_for_polishing': ('Use sigmas for polishing', 'value'),
            'show_error_bars': ('Show error bars', 'value'),
            'show_q_resolution': ('Show q-resolution', 'value'),
            'log_x_axis': ('Log x-axis', 'value'),
            'plot_sld_profile': ('Plot SLD profile', 'value'),
            'sld_pad_left': ('Left padding:', 'value'),
            'sld_pad_right': ('Right padding:', 'value'),
            'exp_color': ('Data color:', 'value'),
            'exp_errcolor': ('Error bars:', 'value'),
            'pred_color': ('Prediction:', 'value'),
            'pol_color': ('Polished:', 'value'),
            'sld_pred_color': ('SLD pred:', 'value'),
            'sld_pol_color': ('SLD polish:', 'value'),
            'calc_pred_curve': ('Calculate curve', 'value'),
            'calc_pred_sld': ('Calculate pred SLD', 'value'),
            'calc_pol_sld': ('Calculate polished SLD', 'value'),
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


