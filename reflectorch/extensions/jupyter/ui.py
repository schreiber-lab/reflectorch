

"""
Jupyter Widget Interface for Reflectorch

This module provides interactive Jupyter widgets for reflectometry analysis using Reflectorch models.
The widgets offer a user-friendly interface for data preprocessing, prediction, and visualization.

Key Features:
    - Interactive parameter bounds adjustment with sliders
    - Real-time prediction and visualization
    - Comprehensive data preprocessing controls
    - Customizable plotting and color options
    - Easy integration with existing Reflectorch workflows

Quick Start:
    ```python
    # Basic widget usage
    from reflectorch.inference import EasyInferenceModel
    from reflectorch.extensions.jupyter import create_basic_widget
    
    model = EasyInferenceModel('your_config.yaml')
    widget = create_basic_widget(model, reflectivity_curve, q_values=q_values)
    
    # Advanced widget usage
    from reflectorch.extensions.jupyter import create_advanced_widget
    
    widget = create_advanced_widget(
        model, 
        reflectivity_curve, 
        q_values,
        sigmas=sigmas,
        initial_prior_bounds=my_bounds
    )
    
    # Access results
    results = widget.prediction_result
    ```

Widget Types:
    1. **Basic Widget**: Simple interface for adjusting prior bounds and making predictions
    2. **Advanced Widget**: Full-featured interface with preprocessing and visualization controls
"""

import numpy as np
from typing import Optional, Dict, Any, Union, List, Tuple
import ipywidgets as widgets
from IPython.display import display

from ...inference.plotting import plot_reflectivity, plot_prediction_results, print_prediction_results


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
    
    def __init__(self, model):
        """
        Initialize the widget with a Reflectorch model
        
        Args:
            model: EasyInferenceModel instance for making predictions
        """
        self.model = model
        self.prediction_result = None
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
        
        # Create parameter sliders
        sliders = self._create_parameter_sliders(
            param_labels, min_bounds, max_bounds, max_deltas, initial_prior_bounds
        )
        
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
        
        # Layout
        sliders_box = widgets.VBox([slider_row for _, slider_row in sliders])
        buttons_box = widgets.HBox([predict_button, close_button])
        container = widgets.VBox([
            widgets.HTML("<h3>Reflectorch Basic Prediction Widget</h3>"),
            sliders_box, 
            buttons_box, 
            output
        ])
        
        display(container)
        
        # Event handlers
        @output.capture(clear_output=True)
        def on_predict_click(_):
            """Handle predict button click"""
            try:
                # Get current slider values
                prior_bounds = np.array([[slider.value[0], slider.value[1]] 
                                       for slider, _ in sliders])
                
                # Make prediction
                prediction_result = self.model.predict(
                    reflectivity_curve=reflectivity_curve,
                    q_values=q_values,
                    prior_bounds=prior_bounds,
                    **predict_kwargs
                )
                
                # Display results
                print_prediction_results(prediction_result)
                plot_prediction_results(
                    prediction_result,
                    q_exp=q_values if q_values is not None else 
                          self.model.trainer.loader.q_generator.q.cpu().numpy(),
                    curve_exp=reflectivity_curve,
                )
                
                # Store results
                self.prediction_result = prediction_result
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")
        
        def on_close_click(_):
            """Handle close button click"""
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
            label_widget = widgets.Label(
                value=label,
                layout=widgets.Layout(width='150px')
            )
            slider = widgets.FloatRangeSlider(
                value=[init_min, init_max],
                min=float(min_bounds[i]),
                max=float(max_bounds[i]),
                step=0.01,
                layout=widgets.Layout(width='420px'),
                readout_format='.3f',
                description=label
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
        
        # Main layout
        main_controls = widgets.VBox([
            widgets.HTML("<h3>Reflectorch Advanced Prediction Widget</h3>"),
            priors_section,
            controls_section,
            widgets.HBox([predict_button, close_button]),
        ])
        
        container = widgets.VBox([main_controls, output])
        display(container)
        
        # Event handlers
        @output.capture(clear_output=True)
        def on_predict_click(_):
            """Handle predict button click with all current settings"""
            try:
                output.clear_output(wait=True)
                
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
                    clip_prediction=True,
                    polish_prediction=settings.get('polish_prediction', True),
                    calc_pred_curve=True,
                    calc_pred_sld_profile=True,
                    calc_polished_sld_profile=True,
                )
                
                # Display results
                print_prediction_results(prediction_result)
                
                # Simple plotting with default parameters
                plot_prediction_results(
                    prediction_result,
                    q_exp=q_values,
                    curve_exp=reflectivity_curve,
                    sigmas_exp=sigmas,
                )
                
                # Store results
                self.prediction_result = prediction_result
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        def on_close_click(_):
            """Handle close button click"""
            container.close()
            print("Advanced prediction widget closed.")
        
        # Connect event handlers
        predict_button.on_click(on_predict_click)
        close_button.on_click(on_close_click)
    
    def _create_priors_section(self, param_labels, min_bounds, max_bounds, max_deltas, initial_bounds=None):
        """Create the priors section with parameter sliders"""
        sliders = self._create_parameter_sliders(
            param_labels, min_bounds, max_bounds, max_deltas, initial_bounds
        )
        
        slider_rows = [slider_row for _, slider_row in sliders]
        return widgets.VBox([
            widgets.HTML("<h4>Prior Bounds</h4>"),
            *slider_rows
        ])
    
    def _create_controls_section(self, n_datapoints):
        """Create basic controls section"""
        return widgets.VBox([
            widgets.HTML("<h4>Settings</h4>"),
            widgets.Checkbox(description='Polish prediction', value=True),
            widgets.HTML("<i>Basic preprocessing and plotting options available</i>")
        ])
    
    def _extract_widget_settings(self, priors_section, controls_section):
        """Extract all current widget settings into a dictionary"""
        settings = {}
        
        # Extract prior bounds from sliders
        slider_widgets = []
        def find_sliders(widget):
            if isinstance(widget, widgets.FloatRangeSlider):
                slider_widgets.append(widget)
            elif hasattr(widget, 'children'):
                for child in widget.children:
                    find_sliders(child)
        
        find_sliders(priors_section)
        settings['prior_bounds'] = np.array([[s.value[0], s.value[1]] for s in slider_widgets])
        
        # Extract polish setting
        checkbox_widgets = []
        def find_checkboxes(widget):
            if isinstance(widget, widgets.Checkbox):
                checkbox_widgets.append(widget)
            elif hasattr(widget, 'children'):
                for child in widget.children:
                    find_checkboxes(child)
        
        find_checkboxes(controls_section)
        settings['polish_prediction'] = checkbox_widgets[0].value if checkbox_widgets else True
        
        return settings


# Convenience functions for backward compatibility and ease of use
def create_basic_widget(model, reflectivity_curve, **kwargs):
    """
    Convenience function to create and display a basic prediction widget
    
    Args:
        model: EasyInferenceModel instance
        reflectivity_curve: Experimental reflectivity data
        **kwargs: Additional arguments passed to basic_predict_widget
        
    Returns:
        ReflectJupyterWidget: The widget instance
    """
    widget = ReflectJupyterWidget(model)
    widget.basic_predict_widget(reflectivity_curve, **kwargs)
    return widget


def create_advanced_widget(model, reflectivity_curve, q_values, **kwargs):
    """
    Convenience function to create and display an advanced prediction widget
    
    Args:
        model: EasyInferenceModel instance
        reflectivity_curve: Experimental reflectivity data
        q_values: Momentum transfer values
        **kwargs: Additional arguments passed to advanced_predict_widget
        
    Returns:
        ReflectJupyterWidget: The widget instance
    """
    widget = ReflectJupyterWidget(model)
    widget.advanced_predict_widget(reflectivity_curve, q_values, **kwargs)
    return widget