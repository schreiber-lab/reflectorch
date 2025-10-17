"""
This module provides API for creating and using
Reflectorch widgets and plots in Jupyter notebooks.
"""

import numpy as np
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from reflectorch.inference.inference_model import InferenceModel

from reflectorch.extensions.jupyter.widget import ReflectorchPlotlyWidget


def create_widget(
                 reflectivity_curve: np.ndarray,
                 q_values: np.ndarray,
                 model: Optional["InferenceModel"] = None,
                 sigmas: Optional[np.ndarray] = None,
                 q_resolution: Optional[Union[float, np.ndarray]] = None,
                 initial_prior_bounds: Optional[np.ndarray] = None,
                 ambient_sld: Optional[float] = None,
                 controls_width: int = 700,
                 plot_width: int = 400,
                 plot_height: int = 300,
                 ) -> ReflectorchPlotlyWidget:
    """
    Create and display a Reflectorch analysis widget
    
    This is the main function for creating Reflectorch widgets.
    
    Parameters:
    ----------
        reflectivity_curve: Experimental reflectivity data
        q_values: Momentum transfer values
        model: InferenceModel instance for making predictions (optional)
        sigmas: Experimental uncertainties (optional)
        q_resolution: Q-resolution, float or array (optional)
        initial_prior_bounds: Initial bounds for priors, shape (n_params, 2) (optional)
        ambient_sld: Ambient SLD value (optional)
        controls_width: Width of the controls area in pixels. Default is 700px.
        plot_width: Width of the plots in pixels. Default is 400px.
        plot_height: Height of the plots in pixels. Default is 300px.

    Returns:
    -------
        ReflectorchPlotlyWidget instance with the widget displayed
    
    Example:
    -------
        ```python
        # Load data
        from reflectorch.paths import ROOT_DIR
        data = np.loadtxt(ROOT_DIR / "exp_data/data_C60.txt")

        # create widget (displayed automatically)
        widget = create_widget(q_values=data[..., 0], reflectivity_curve=data[..., 1])
    ```
    """
    # Create widget instance
    widget = ReflectorchPlotlyWidget(
        reflectivity_curve=reflectivity_curve,
        q_values=q_values,
        sigmas=sigmas,
        q_resolution=q_resolution,
        initial_prior_bounds=initial_prior_bounds,
        ambient_sld=ambient_sld,
        model=model,
    )
    
    # Display the widget interface
    widget.display(
        controls_width=controls_width,
        plot_width=plot_width,
        plot_height=plot_height
    )
    
    return widget


# Export the main widget class for direct usage
__all__ = [
    'create_widget', 
    'ReflectorchPlotlyWidget'
]
