"""
This module provides API for creating and using
Reflectorch widgets in Jupyter notebooks.

Key Functions:
- create_widget(): Main function to create a Reflectorch analysis widget
- ReflectorchWidget: Main widget class for direct usage
"""

import numpy as np
from typing import Optional, Union

from .widget import ReflectorchWidget


def create_widget(model, 
                 reflectivity_curve: np.ndarray,
                 q_values: np.ndarray,
                 sigmas: Optional[np.ndarray] = None,
                 q_resolution: Optional[Union[float, np.ndarray]] = None,
                 initial_prior_bounds: Optional[np.ndarray] = None,
                 ambient_sld: Optional[float] = None,
                 interactive_plots: bool = True) -> ReflectorchWidget:
    """
    Create and display a Reflectorch analysis widget
    
    This is the main function for creating Reflectorch widgets. It provides a clean,
    simple interface for reflectometry data analysis with comprehensive controls.
    
    Args:
        model: EasyInferenceModel instance for making predictions
        reflectivity_curve: Experimental reflectivity data
        q_values: Momentum transfer values
        sigmas: Experimental uncertainties (optional)
        q_resolution: Q-resolution, float or array (optional)
        initial_prior_bounds: Initial bounds for priors, shape (n_params, 2) (optional)
        ambient_sld: Ambient SLD value (optional)
        interactive_plots: Enable interactive plotting features (default: True)
        
    Returns:
        ReflectorchWidget instance with the widget displayed
        
    Features:
        - Professional tabbed interface (Parameters, Preprocessing, Prediction, Plotting)
        - Interactive parameter table with real-time result updates
        - Comprehensive data preprocessing controls
        - Full plotting customization
        - Interactive plots with coordinate display (when enabled)
        - Clean, modern UI design
        
    Example:
        ```python
        from reflectorch.inference import EasyInferenceModel
        from reflectorch.extensions.jupyter import create_widget
        
        # Load your model
        model = EasyInferenceModel('config.yaml')
        
        # Load your data
        q_values, reflectivity_curve, sigmas = load_data()
        
        # Create and display the widget
        widget = create_widget(
            model=model,
            reflectivity_curve=reflectivity_curve,
            q_values=q_values,
            sigmas=sigmas
        )
        
        # Access results after making predictions
        results = widget.prediction_result
        ```
        
    Interactive Features:
        - Real-time parameter result updates in structured table
        - Interactive plots with cursor coordinate display
        - Zoom, pan, and navigation controls
        - Figures update in-place (no accumulation)
        - Professional tabbed organization
        
    Tabs Overview:
        1. **Parameters**: Structured table with parameter sliders and live results
        2. **Preprocessing**: Data truncation and error bar filtering options  
        3. **Prediction**: Prediction polishing and computation settings
        4. **Plotting**: Display options and complete color customization
    """
    # Create widget instance
    widget = ReflectorchWidget(model, interactive_plots=interactive_plots)
    
    # Display the widget interface
    widget.display(
        reflectivity_curve=reflectivity_curve,
        q_values=q_values,
        sigmas=sigmas,
        q_resolution=q_resolution,
        initial_prior_bounds=initial_prior_bounds,
        ambient_sld=ambient_sld
    )
    
    return widget


# Export the main widget class for direct usage
__all__ = ['create_widget', 'ReflectorchWidget']
