"""
Reflectorch Jupyter Widget
"""
import torch
import numpy as np
from typing import Optional, Union, TYPE_CHECKING

from torch.types import Device

if TYPE_CHECKING:
    from reflectorch.inference.inference_model import InferenceModel
import ipywidgets as widgets
from IPython.display import display

from reflectorch.extensions.jupyter.plotly_plot_manager import (
    PlotlyPlotManager,
    plot_reflectivity_only,
    plot_sld_only,
)
from reflectorch.extensions.jupyter.components import (
    ParameterTable,
    PreprocessingControls,
    PredictionControls,
    PlottingControls,
    AdditionalParametersControls,
    WidgetSettingsExtractor,
)
from reflectorch.extensions.jupyter.model_selection import ModelSelection
from reflectorch.extensions.jupyter.log_widget import LogWidget

from huggingface_hub.utils import disable_progress_bars

# that causes some Rust related errors when downloading models from Huggingface
disable_progress_bars()


class ReflectorchPlotlyWidget:
    """
    Interactive Jupyter Widget for Reflectometry Analysis using Plotly
        
    Attributes:
        model: The InferenceModel instance
        prediction_result: Latest prediction results
        plot_manager: PlotlyPlotManager for handling interactive plots
        
    Example:
        ```python
        from reflectorch.inference import InferenceModel
        from reflectorch.extensions.jupyter import ReflectorchPlotlyWidget
        
        model = InferenceModel('config.yaml')
        widget = ReflectorchPlotlyWidget(model)
        
        widget.display(
            reflectivity_curve=data,
            q_values=q_values,
            sigmas=sigmas
        )
        
        # Access results
        results = widget.prediction_result
        ```
    """
    
    def __init__(self,                
                reflectivity_curve: np.ndarray,
                q_values: np.ndarray,
                sigmas: Optional[np.ndarray] = None,
                q_resolution: Optional[Union[float, np.ndarray]] = None,
                initial_prior_bounds: Optional[np.ndarray] = None,
                ambient_sld: Optional[float] = None,
                model: Optional["InferenceModel"] = None,
                root_dir: Optional[str] = None,
):
        """
        Initialize the Reflectorch Plotly widget
        
        Args:
            reflectivity_curve: Experimental reflectivity data
            q_values: Momentum transfer values
            sigmas: Experimental uncertainties (optional)
            q_resolution: Q-resolution, float or array (optional)
            initial_prior_bounds: Initial bounds for priors, shape (n_params, 2)
            ambient_sld: Ambient SLD value (optional)
            model: InferenceModel instance for making predictions (optional)
            root_dir: Root directory for the model (optional)
        """
        self.model = model
        self.prediction_result = None
        self.plot_manager = PlotlyPlotManager()
        self.root_dir = root_dir
        # Store data for prediction
        self._data = {
            'reflectivity_curve': reflectivity_curve,
            'q_values': q_values,
            'sigmas': sigmas,
            'q_resolution': q_resolution,
            'ambient_sld': ambient_sld
        }

        self.initial_prior_bounds = initial_prior_bounds
        
        # Widget components (initialized when display is called)
        self.parameter_table = None
        self.preprocessing_controls = None
        self.prediction_controls = None
        self.plotting_controls = None
        self.additional_params_controls = None
        self.model_selection = None
        self.tabs_widget = None  # Store reference to tabs for updating
        self.predict_button = None  # Store reference to predict button
        self.log_widget = None  # Store reference to log widget
        
        if self.model is not None:
            self._validate_model()
    
    def _create_parameter_components(self, initial_prior_bounds=None, predict_button=None):
        """
        Create parameter table and additional parameters controls based on current model
        
        Args:
            initial_prior_bounds: Initial bounds for priors, shape (n_params, 2)
            predict_button: Optional predict button to include in parameter table
            
        Returns:
            Tuple of (parameter_table, additional_params_controls)
        """
        # Create additional params controls first so we can pass it to parameter table
        additional_params_controls = AdditionalParametersControls(self.model)
        
        # Get model parameters info if model is available
        if self.model is not None:
            param_labels = self.model.trainer.loader.prior_sampler.param_model.get_param_labels()
            min_bounds = self.model.trainer.loader.prior_sampler.min_bounds.cpu().numpy().flatten()
            max_bounds = self.model.trainer.loader.prior_sampler.max_bounds.cpu().numpy().flatten()
            max_deltas = self.model.trainer.loader.prior_sampler.max_delta.cpu().numpy().flatten()
        else:
            # Default empty parameters when no model is loaded
            param_labels = []
            min_bounds = np.array([])
            max_bounds = np.array([])
            max_deltas = np.array([])
        
        parameter_table = ParameterTable(
            param_labels, min_bounds, max_bounds, max_deltas, initial_prior_bounds,
            additional_params_controls=additional_params_controls,
            predict_button=predict_button
        )
        
        return parameter_table, additional_params_controls
    
    def _validate_model(self):
        """Validate that the model has required attributes"""
        required_attrs = ['trainer', 'preprocess_and_predict']
        for attr in required_attrs:
            if not hasattr(self.model, attr):
                raise ValueError(f"Model must have '{attr}' attribute")
    
    
    def display(self, 
                controls_width: int = 700,
                plot_width: int = 400,
                plot_height: int = 300
            ):
        """
        Display the widget interface
        
        Parameters:
        ----------
            reflectivity_curve: Experimental reflectivity data
            q_values: Momentum transfer values (required)
            sigmas: Experimental uncertainties (optional)
            q_resolution: Q-resolution, float or array (optional)
            initial_prior_bounds: Initial bounds for priors, shape (n_params, 2)
            ambient_sld: Ambient SLD value (optional)
            controls_width: Width of the controls area in pixels. Default is 700px.
            plot_width: Width of the plots in pixels. Default is 400px.
            plot_height: Height of the plots in pixels. Default is 300px.
        """

        # Create predict button (disabled by default if no model)
        self.predict_button = widgets.Button(
            description="Predict",
            button_style='primary',
            tooltip='Run prediction with current settings' if self.model else 'Load a model first',
            layout=widgets.Layout(width='120px'),
            disabled=(self.model is None)
        )
        
        # Create widget components
        self.parameter_table, self.additional_params_controls = self._create_parameter_components(
            self.initial_prior_bounds, self.predict_button
        )
        self.preprocessing_controls = PreprocessingControls(len(self._data['reflectivity_curve']))
        self.prediction_controls = PredictionControls()
        self.plotting_controls = PlottingControls()
        self.model_selection = ModelSelection()
        
        # Create log widget
        self.log_widget = LogWidget()
        
        # Create tabbed interface
        # if model is not provided, then Models tabs goes first.
        tabs = widgets.Tab()
        if self.model is None:
            tabs.children = [
                self.model_selection.widget,
                self.parameter_table.widget,
                self.preprocessing_controls.widget,
                self.prediction_controls.widget,
                self.plotting_controls.widget
            ]
            tabs.titles = ['Models', 'Parameters', 'Preprocessing', 'Prediction', 'Plotting']
        else:
            tabs.children = [
                self.parameter_table.widget,
                self.preprocessing_controls.widget,
                self.prediction_controls.widget,
                self.plotting_controls.widget,
                self.model_selection.widget
            ]
            tabs.titles = ['Parameters', 'Preprocessing', 'Prediction', 'Plotting', 'Models']

        self.tab_indices = dict(zip(tabs.titles, range(len(tabs.titles))))
        # Store reference to tabs for later updates
        self.tabs_widget = tabs
        
        # Create plot containers (initially empty)
        reflectivity_plot_container = widgets.VBox([])
        sld_plot_container = widgets.VBox([])
        
        # Combine plots vertically on the right
        plot_area = widgets.VBox([
            reflectivity_plot_container,
            sld_plot_container
        ], layout=widgets.Layout(margin='50px 0px 0px 0px'))
        
        # Main layout with controls on left, plots on right
        header = widgets.HTML("<h2>Reflectorch Widget</h2>")
        
        controls_area = widgets.VBox([
            header,
            tabs
        ], layout=widgets.Layout(width=f'{controls_width}px'))
        
        # Horizontal layout: controls on left, plots on right
        main_content = widgets.HBox([
            controls_area,
            plot_area
        ])
        
        # Complete layout with log at the bottom
        main_layout = widgets.VBox([
            main_content,
            self.log_widget.widget
        ])
        
        # Add border around the entire widget
        container = widgets.VBox([main_layout], layout=widgets.Layout(
            border='2px solid #d0d0d0',
            border_radius='8px',
            padding='15px',
            margin='10px',
            background_color='#fafafa'
        ))
        display(container)
        
        # Setup event handlers
        self._setup_event_handlers(self.predict_button, reflectivity_plot_container, sld_plot_container, container)
        
        # Setup model selection integration
        self._setup_model_selection_integration()
        
        # Setup truncation synchronization
        self._setup_truncation_sync()
        
        # Create initial plots with experimental data
        self._create_initial_plots(reflectivity_plot_container, sld_plot_container, plot_width, plot_height)
        
        # Setup reactive plot updates for plotting controls
        self._setup_reactive_plot_updates(reflectivity_plot_container, sld_plot_container)
    
    def _create_initial_plots(self, reflectivity_container, sld_container, plot_width, plot_height):
        """Create initial plots showing experimental data"""
        try:
            # Use default settings for initial plots
            settings = {
                'show_error_bars': True,
                'show_q_resolution': True,
                'exp_color': 'blue',
                'exp_errcolor': 'purple',
                'log_x_axis': False,
                'plot_sld_profile': True
            }
            
            # Plot initial experimental data
            self._plot_initial_data(settings, reflectivity_container, sld_container, plot_width, plot_height)
            
        except Exception as e:
            if self.log_widget:
                self.log_widget.log(f"⚠️  Could not create initial plots: {str(e)}")
            else:
                print(f"⚠️  Could not create initial plots: {str(e)}")
    
    def _plot_initial_data(self, settings, reflectivity_container, sld_container, plot_width, plot_height):
        """Plot only experimental data before any prediction"""
        # Prepare experimental data for plotting
        q_exp_plot = self._data['q_values']
        r_exp_plot = self._data['reflectivity_curve']
        yerr_plot = self._data['sigmas'] if settings['show_error_bars'] and self._data['sigmas'] is not None else None
        xerr_plot = self._data['q_resolution'] if settings['show_q_resolution'] and self._data['q_resolution'] is not None else None
        
        # Create reflectivity plot
        reflectivity_fig = plot_reflectivity_only(
            plot_manager=self.plot_manager,
            figure_id="reflectivity_plot",
            q_exp=q_exp_plot, r_exp=r_exp_plot, yerr=yerr_plot, xerr=xerr_plot,
            exp_color=settings['exp_color'],
            exp_errcolor=settings['exp_errcolor'],
            exp_label='experimental data',
            logx=settings['log_x_axis'], logy=True,
            width=plot_width, height=plot_height
        )
        
        # Get the reflectivity plotly widget and add it to container
        reflectivity_widget = self.plot_manager.get_widget("reflectivity_plot")
        reflectivity_container.children = [reflectivity_widget]
        
        # Create empty SLD plot (will be populated after prediction)
        if settings['plot_sld_profile']:
            sld_fig = plot_sld_only(
                plot_manager=self.plot_manager,
                figure_id="sld_plot",
                z_sld=None, sld_pred=None, sld_pol=None,
                width=plot_width, height=plot_height
            )
            
            # Get the SLD plotly widget and add it to container
            sld_widget = self.plot_manager.get_widget("sld_plot")
            sld_container.children = [sld_widget]
    
    def _setup_event_handlers(self, predict_button, reflectivity_container, sld_container, container):
        """Setup button event handlers"""
        
        def on_predict(_):
            """Handle predict button click"""
            with self.log_widget.capture_prints():
                # Store original button state
                original_description = predict_button.description
                original_disabled = predict_button.disabled
                
                try:
                    # Check if model is loaded
                    if self.model is None:
                        print("❌ No model loaded. Please load a model from the Models tab first.")
                        return
                    
                    # Disable button and show "Predicting..." 
                    predict_button.disabled = True
                    predict_button.description = "Predicting..."
                    
                    # Extract settings from all components with data fallback
                    settings = WidgetSettingsExtractor.extract_settings(
                        self.parameter_table,
                        self.preprocessing_controls,
                        self.prediction_controls,
                        self.plotting_controls,
                        self.additional_params_controls,
                        data=self._data
                    )
                    
                    # Separate prediction and plotting parameters
                    prediction_params, plotting_params = WidgetSettingsExtractor.separate_settings(settings)
                    
                    # Run prediction with all parameters
                    prediction_result = self.model.preprocess_and_predict(**prediction_params)
                    
                    # Update parameter table with results
                    self.parameter_table.update_results(prediction_result)
                    
                    # Plot results
                    self._plot_results(prediction_result, plotting_params, reflectivity_container, sld_container)
                    
                    # Store results
                    self.prediction_result = prediction_result
                    
                except Exception as e:
                    print(f"❌ Prediction error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always restore button state, even if there was an error
                    predict_button.description = original_description
                    predict_button.disabled = original_disabled
                
        # Connect event handlers
        predict_button.on_click(on_predict)
    
    def _setup_model_selection_integration(self):
        """Setup integration with model selection tab"""
        if not self.model_selection:
            return
        
        load_button = self.model_selection._widgets['download_button']
        
        def on_load_model(_):
            """Handle load model button click"""
            with self.log_widget.capture_prints():
                # Store original button state
                original_description = load_button.description
                original_disabled = load_button.disabled
                
                try:
                    selected_model_info = self.model_selection.get_selected_model_info()
                    if selected_model_info is None:
                        print("❌ No model selected")
                        return
                    
                    # Disable button and show "Downloading..."
                    load_button.disabled = True
                    load_button.description = "Downloading..."
                    
                    print(f"🔄 Loading model: {selected_model_info['model_name']} ...")
                    
                    from reflectorch.inference.inference_model import InferenceModel

                    device: Device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    
                    # Create new model instance
                    new_model = InferenceModel(
                        config_name=selected_model_info['config_name'],
                        repo_id=selected_model_info['repo_id'],
                        device=device,
                        root_dir=self.root_dir
                    )
                    
                    print(f"📥 Model downloaded and initialized successfully")
                    
                    # Replace current model
                    self.model = new_model
                    
                    # Enable the predict button since we now have a model
                    if self.predict_button:
                        self.predict_button.disabled = False
                        self.predict_button.tooltip = 'Run prediction with current settings'
                    
                    # Create new parameter components for the new model (reuse the same predict button)
                    new_parameter_table, new_additional_params_controls = self._create_parameter_components(predict_button=self.predict_button)
                    
                    # Update the parameter table title with the new model name
                    new_parameter_table.param_title.value = f"<h4>Parameter Configuration for {selected_model_info['model_name']}</h4>"
                    
                    # Update the parameter table in the tabs
                    if self.tabs_widget and hasattr(self.tabs_widget, 'children'):
                        children_list = list(self.tabs_widget.children)
                        children_list[self.tab_indices['Parameters']] = new_parameter_table.widget
                        self.tabs_widget.children = children_list
                        
                        # Update our references to the components
                        self.parameter_table = new_parameter_table
                        self.additional_params_controls = new_additional_params_controls
                        
                        # Switch to Parameters tab automatically
                        self.tabs_widget.selected_index = self.tab_indices['Parameters']
                    
                    # Clear previous prediction results
                    self.prediction_result = None
                    
                    # Get parameter info for success message
                    param_labels = self.model.trainer.loader.prior_sampler.param_model.get_param_labels()
                    max_layers = self.model.trainer.loader.prior_sampler.max_num_layers
                    
                    print(f"✅ Model loaded successfully: {selected_model_info['config_name']}")
                    print(f"Parameters: {len(param_labels)} parameters, {max_layers} max layers")
                    print("💡 Tip: Go to the Parameters tab to see the updated parameter ranges for the new model")
                    
                except Exception as e:
                    print(f"❌ Error loading model: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always restore button state, even if there was an error
                    load_button.description = original_description
                    load_button.disabled = original_disabled
        
        # Connect the load button
        load_button.on_click(on_load_model)
    
    def _plot_results(self, prediction_result, settings, reflectivity_container, sld_container):
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
        
        # Handle complex SLD
        if sld_pred is not None and np.iscomplexobj(sld_pred):
            sld_pred = sld_pred.real
        if sld_pol is not None and np.iscomplexobj(sld_pol):
            sld_pol = sld_pol.real
        
        # Update reflectivity plot
        reflectivity_fig = plot_reflectivity_only(
            plot_manager=self.plot_manager,
            figure_id="reflectivity_plot",
            q_exp=q_exp_plot, r_exp=r_exp_plot, yerr=yerr_plot, xerr=xerr_plot,
            exp_color=settings['exp_color'],
            exp_errcolor=settings['exp_errcolor'],
            q_pred=q_pred, r_pred=r_pred, pred_color=settings['pred_color'],
            q_pol=q_pol, r_pol=r_pol, pol_color=settings['pol_color'],
            logx=settings['log_x_axis'], logy=True,
            width=600, height=300
        )
        
        # Update SLD plot if requested
        if settings['plot_sld_profile']:
            sld_fig = plot_sld_only(
                plot_manager=self.plot_manager,
                figure_id="sld_plot",
                z_sld=z_sld, sld_pred=sld_pred, sld_pol=sld_pol,
                sld_pred_color=settings['sld_pred_color'],
                sld_pol_color=settings['sld_pol_color'],
                width=600, height=250
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
    
    def _setup_reactive_plot_updates(self, reflectivity_container, sld_container):
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
                    self._plot_results(self.prediction_result, settings, reflectivity_container, sld_container)
                    
                except Exception as e:
                    if self.log_widget:
                        self.log_widget.log(f"⚠️  Error updating plot: {str(e)}")
                    else:
                        print(f"⚠️  Error updating plot: {str(e)}")
            else:
                # If no prediction results yet, just update the initial plot
                try:
                    self._update_initial_plot_style(reflectivity_container, sld_container)
                except Exception as e:
                    if self.log_widget:
                        self.log_widget.log(f"⚠️  Error updating initial plot: {str(e)}")
                    else:
                        print(f"⚠️  Error updating initial plot: {str(e)}")
        
        # Setup observers for all reactive controls
        for control in reactive_controls:
            if hasattr(control, 'observe'):
                control.observe(update_plot_on_change, names='value')
    
    def _update_initial_plot_style(self, reflectivity_container, sld_container):
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
            self._plot_initial_data(settings, reflectivity_container, sld_container)
            
        except Exception as e:
            if self.log_widget:
                self.log_widget.log(f"⚠️  Error updating initial plot style: {str(e)}")
            else:
                print(f"⚠️  Error updating initial plot style: {str(e)}")
