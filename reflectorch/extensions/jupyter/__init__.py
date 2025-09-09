"""
Reflectorch Jupyter Extensions

This module provides interactive Jupyter widgets for reflectometry analysis.
The API has been simplified and modernized with a clean, modular architecture.

Main Functions:
    create_widget(): Create a comprehensive analysis widget with tabbed interface
    
Classes:
    ReflectorchWidget: Main widget class for direct usage
    JPlotLoss: Training loss plotting callback
    
Advanced Components (for custom usage):
    InteractivePlotManager: Manages interactive plots
    plot_reflectivity_interactive: Interactive plotting function
    
Example:
    ```python
    from reflectorch.inference import EasyInferenceModel
    from reflectorch.extensions.jupyter import create_widget
    
    model = EasyInferenceModel('config.yaml')
    widget = create_widget(model, data, q_values=q_values)
    ```
"""

# Main clean API
from .api import create_widget, ReflectorchWidget

# Training callback
from .callbacks import JPlotLoss

# Advanced components for custom usage
from .interactive_plotting import InteractivePlotManager, plot_reflectivity_interactive, plot_prediction_results_interactive

# Widget components for advanced customization
from .components import ParameterTable, PreprocessingControls, PredictionControls, PlottingControls

__all__ = [
    # Main API (recommended)
    'create_widget',
    'ReflectorchWidget',
    
    # Training utilities
    'JPlotLoss',
    
    # Advanced components
    'InteractivePlotManager',
    'plot_reflectivity_interactive', 
    'plot_prediction_results_interactive',
    
    # Widget components for customization
    'ParameterTable',
    'PreprocessingControls', 
    'PredictionControls',
    'PlottingControls',
]
