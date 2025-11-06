"""
Jupyter Widget Model Selection Component for Reflectorch

This module contains the model browsing and selection component for
reflectometry analysis models from Hugging Face repositories.

Components:
    - ModelSelection: Model browsing and selection from Hugging Face
"""

from typing import Optional, Dict, Any, Callable
import ipywidgets as widgets
from huggingface_hub import HfApi

from reflectorch.extensions.jupyter.custom_select import CustomSelect

class ModelSelection:
    """Model selection component for Hugging Face models.
    """
    
    def __init__(self, organization_list: tuple[str, ...] = ('reflectorch-ILL', )):
        """
        Initialize model selection component
        
        Args:
            organization_list: Tuple of Hugging Face organizations to browse models from
        """
        assert len(organization_list) > 0, "At least one organization must be provided"
        self.organization = organization_list[0]
        self.organization_list = organization_list
        self.hf_api = HfApi()
        self.models_data = []
        self.selected_config = None
        self.selected_model = None
        self._model_cache = {}  # Cache model info by organization
        self._download_callback = None  # External download handler
        
        self.widget = self._create_model_browser()
        
    def _create_model_details_template(self) -> str:
        """Create a comprehensive HTML template for model details"""
        return """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.5;">
            <div style="margin-bottom: 12px;">
                <h4 style="margin: 0 0 8px 0; color: #0d6efd; font-size: 16px;">📋 {modelId}</h4>
                <div style="font-size: 13px; color: #6c757d;">
                    <b>Layers:</b> {num_layers} | <b>Parameterization:</b> {parameterization}
                </div>
            </div>
            
            <div style="margin-bottom: 12px;">
                <h5 style="margin: 0 0 6px 0; color: #495057; font-size: 14px;">Parameter Ranges</h5>
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; font-size: 12px;">
                    {param_ranges_table}
                </div>
            </div>
            
            <div style="margin-bottom: 12px;">
                <h5 style="margin: 0 0 6px 0; color: #495057; font-size: 14px;">Prior Bound Widths</h5>
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; font-size: 12px;">
                    {bound_width_ranges_table}
                </div>
            </div>
        </div>
        """
        
    def _create_model_browser(self) -> widgets.VBox:
        """Create the model browser widget"""
        # Title and description
        title = widgets.HTML(
            "<h3>🤗 Model Selection</h3>",
            layout=widgets.Layout(margin='0px 0px 10px 0px')
        )
        
        description = widgets.HTML(
            "<p>Browse and select reflectometry models from Hugging Face repositories.</p>",
            layout=widgets.Layout(margin='0px 0px 15px 0px')
        )

        # Download button - initially disabled
        download_model_button = widgets.Button(
            description="📥 Download Model",
            button_style='success',
            disabled=True,
            tooltip='Select a model to enable download',
            layout=widgets.Layout(width='150px')
        )
        
        refresh_button = widgets.Button(
            description="🔄 Refresh",
            button_style='info',
            tooltip='Refresh model list (uses cache after first load)',
            layout=widgets.Layout(width='100px')
        )
                
        # Organization selection
        org_dropdown = widgets.Dropdown(
            options=[(org, org) for org in self.organization_list],
            value=self.organization,
            description='Organization:',
            layout=widgets.Layout(width='300px')
        )
        
        # Controls row
        controls_row = widgets.HBox([
            org_dropdown,
            refresh_button,
            download_model_button,
        ], layout=widgets.Layout(justify_content='flex-start', margin='0px 0px 15px 0px'))
        
        # Status
        status_label = widgets.HTML(
            "Click 'Refresh' to load models",
            layout=widgets.Layout(margin='5px 0px')
        )
        
        
        # Add selection status widget
        selection_status = widgets.HTML(
            value="<i>Click a model row to select it</i>",
            layout=widgets.Layout(margin='5px 0px')
        )
        
        # Model selection dropdown
        model_selector = CustomSelect(
            data=[],
            columns=[
                ('Model ID', 'modelId'),
                ('Layers', 'num_layers'),
                ('Parameterization', 'parameterization')
            ],
            layout=widgets.Layout(width='600px'),
            show_details=True,
            details_template=self._create_model_details_template()
        )
        
        
        # Store widget references for event handlers
        self._widgets = {
            'refresh_button': refresh_button,
            'download_button': download_model_button,
            'org_dropdown': org_dropdown,
            'status_label': status_label,
            'model_selector': model_selector,
            'selection_status': selection_status
        }
        
        # Setup event handlers
        self._setup_model_browser_events()
        
        # Main content area
        return widgets.VBox([
            title,
            description,
            controls_row,
            status_label,
            model_selector.container,
            selection_status,
        ])
    
    def _setup_model_browser_events(self):
        """Setup event handlers for model browser"""
        widgets_dict = self._widgets
        
        def on_refresh_click(b):
            """Handle refresh button click"""
            self._load_models()
                
        def on_org_change(change):
            """Handle organization dropdown change"""
            self.organization = change['new']
            self._load_models()
            # Clear selection when organization changes
            self._clear_selection()
                
        def on_model_select(row):
            """Handle model selection from CustomSelect"""
            if row is None:
                self._clear_selection()
                return
            
            # Extract model data and index from the row
            model_index = row.get('index')
            model_data = row.get('_model_data')
            
            if model_index is not None and model_data is not None:
                self._select_model(model_data, model_index)
                widgets_dict['selection_status'].value = f"<b>Selected:</b> {row.get('modelId')}"
                # Enable download button
                widgets_dict['download_button'].disabled = False
                widgets_dict['download_button'].tooltip = f"Download {row.get('modelId')}"
        
        def on_download_click(b):
            """Handle download button click"""
            if self.selected_model and self._download_callback:
                self._download_callback(self.get_selected_model_info())
        
        # Connect event handlers
        widgets_dict['refresh_button'].on_click(on_refresh_click)
        widgets_dict['download_button'].on_click(on_download_click)
        widgets_dict['org_dropdown'].observe(on_org_change, names='value')
        widgets_dict['model_selector'].on_select(on_model_select)
    
    def select_model_by_index(self, index):
        """Public method to select a model by index"""
        if 0 <= index < len(self.models_data):
            model = self.models_data[index]
            self._select_model(model, index)
            model_id = model.get('modelId', 'Unknown')
            self._widgets['selection_status'].value = f"<b>Selected:</b> {model_id}"
            # Enable download button
            self._widgets['download_button'].disabled = False
            self._widgets['download_button'].tooltip = f"Download {model_id}"
            # Update CustomSelect to match
            self._widgets['model_selector'].set_selected_index(index)
        else:
            self._widgets['selection_status'].value = f"<i style='color: red;'>Invalid index: {index}</i>"
            self._clear_selection()
    
    def _load_models(self):
        """Load models synchronously"""
        try:
            self._widgets['status_label'].value = "🔄 Loading models..."
            
            # Get models from specific organization using HF API
            if self.organization in self._model_cache:
                self._widgets['status_label'].value = "🔄 Loading cached models..."
                models = self._model_cache[self.organization]
            else:
                self._widgets['status_label'].value = f"🔄 Loading models from {self.organization}..."
                
                # List all models from the organization
                hf_models = list(self.hf_api.list_models(author=self.organization, cardData=True))
                
                # Convert HF model objects to our format and try to get config info
                models = []
                for i, hf_model in enumerate(hf_models):
                    try:
                        metadata = hf_model.card_data.get("metadata", {})
                        parameterization = metadata.get("parameterization", "slab")
                        num_layers = metadata.get("number_of_layers", 0)
                        param_ranges = metadata.get("param_ranges", {})
                        bound_width_ranges = metadata.get("bound_width_ranges", {})
                        misalignment_included = metadata.get("shift_param_config", {})
                        
                        # Create model info
                        model_info = {
                            'index': i,
                            'modelId': hf_model.id,
                            'author': self.organization,
                            'hf_model': hf_model,
                            'metadata': hf_model.card_data.get("metadata", {}),
                            'num_layers': num_layers,
                            'param_ranges': param_ranges,
                            'bound_width_ranges': bound_width_ranges,
                            'misalignment_included': misalignment_included,
                            'parameterization': parameterization,
                        }
                        models.append(model_info)
                        
                    except Exception as e:
                        print(f"Warning: Could not process model {hf_model.id}: {e}")
                        continue
                
                # Cache the results
                self._model_cache[self.organization] = models
                            
            self.models_data = models
            
            # Update UI
            self._display_models(models)
            
            self._widgets['status_label'].value = f"✅ Loaded {len(models)} models"
            
        except Exception as e:
            self._widgets['status_label'].value = f"❌ Error loading models: {str(e)}"
            print(f"Error loading models: {e}")
    
    
    def _display_models(self, models):
        """Display models in the CustomSelect widget"""
        # Prepare data for CustomSelect
        model_data = []
        
        for i, model in enumerate(models):
            model_id = model.get('modelId') if isinstance(model, dict) else model.modelId
            num_layers = model.get('num_layers', 'Unknown')
            parameterization = model.get('parameterization', 'Unknown')
            
            # Format detailed information for the template
            param_ranges_table = self._format_parameter_ranges(model.get('param_ranges', {}))
            bound_width_ranges_table = self._format_bound_width_ranges(model.get('bound_width_ranges', {}))
            misalignment_info = self._format_misalignment_info(model.get('misalignment_included', {}))
            additional_metadata = self._format_additional_metadata(model.get('metadata', {}))
            
            # Create row data for CustomSelect
            row_data = {
                'index': i,
                'modelId': model_id,
                'num_layers': str(num_layers),
                'parameterization': parameterization,
                'param_ranges_table': param_ranges_table,
                'bound_width_ranges_table': bound_width_ranges_table,
                'misalignment_info': misalignment_info,
                'additional_metadata': additional_metadata,
                '_model_data': model  # Store full model data for selection
            }
            model_data.append(row_data)
        
        # Update CustomSelect widget with new data
        self._widgets['model_selector'].set_data(model_data)
    
    def _format_parameter_ranges(self, param_ranges: dict) -> str:
        """Format parameter ranges as an HTML table"""
        if not param_ranges:
            return "<i>No parameter range information available</i>"
        
        table_rows = []
        for param_type, ranges in param_ranges.items():
            if isinstance(ranges, list) and len(ranges) == 2:
                min_val, max_val = ranges
                table_rows.append(f"<tr><td style='padding: 2px 8px 2px 0;'><b>{param_type}:</b></td><td style='padding: 2px 0;'>[{min_val}, {max_val}]</td></tr>")
            else:
                table_rows.append(f"<tr><td style='padding: 2px 8px 2px 0;'><b>{param_type}:</b></td><td style='padding: 2px 0;'>{ranges}</td></tr>")
        
        if not table_rows:
            return "<i>No parameter ranges specified</i>"
        
        return f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
            {''.join(table_rows)}
        </table>
        """
    
    def _format_bound_width_ranges(self, bound_ranges: dict) -> str:
        """Format bound width ranges as an HTML table"""
        if not bound_ranges:
            return "<i>No bound width information available</i>"
        
        table_rows = []
        for param_type, ranges in bound_ranges.items():
            if isinstance(ranges, list) and len(ranges) == 2:
                min_val, max_val = ranges
                table_rows.append(f"<tr><td style='padding: 2px 8px 2px 0;'><b>{param_type}:</b></td><td style='padding: 2px 0;'>[{min_val}, {max_val}]</td></tr>")
            else:
                table_rows.append(f"<tr><td style='padding: 2px 8px 2px 0;'><b>{param_type}:</b></td><td style='padding: 2px 0;'>{ranges}</td></tr>")
        
        if not table_rows:
            return "<i>No bound width ranges specified</i>"
        
        return f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
            {''.join(table_rows)}
        </table>
        """
    
    def _format_misalignment_info(self, misalignment_config: dict) -> str:
        """Format misalignment/shift parameter information"""
        if not misalignment_config:
            return "<span style='color: #dc3545;'>❌ No misalignment support</span>"
        
        info_parts = []
        
        # Check for different types of misalignment
        if misalignment_config.get('enabled', False):
            info_parts.append("<span style='color: #198754;'>✅ Misalignment supported</span>")
            
            # Add specific shift types if available
            shift_types = []
            if misalignment_config.get('q_shift', False):
                shift_types.append("Q-shift")
            if misalignment_config.get('intensity_shift', False):
                shift_types.append("Intensity shift")
            if misalignment_config.get('background_shift', False):
                shift_types.append("Background shift")
            
            if shift_types:
                info_parts.append(f"<br><b>Types:</b> {', '.join(shift_types)}")
                
            # Add parameter ranges if available
            if 'param_ranges' in misalignment_config:
                ranges = misalignment_config['param_ranges']
                range_info = []
                for param, range_val in ranges.items():
                    if isinstance(range_val, list) and len(range_val) == 2:
                        range_info.append(f"{param}: [{range_val[0]}, {range_val[1]}]")
                if range_info:
                    info_parts.append(f"<br><b>Ranges:</b> {', '.join(range_info)}")
        else:
            info_parts.append("<span style='color: #dc3545;'>❌ No misalignment support</span>")
        
        return ''.join(info_parts) if info_parts else "<i>Misalignment information not specified</i>"
    
    def _format_additional_metadata(self, metadata: dict) -> str:
        """Format additional model metadata"""
        if not metadata:
            return "<i>No additional metadata available</i>"
        
        info_parts = []
        
        # Q-range information
        q_min = metadata.get('q_min')
        q_max = metadata.get('q_max')
        if q_min is not None and q_max is not None:
            info_parts.append(f"<b>Q-range:</b> [{q_min}, {q_max}]")
        
        # Discretization info
        num_points = metadata.get('num_q_points')
        if num_points:
            info_parts.append(f"<b>Q-points:</b> {num_points}")
        
        # Training info
        training_data = metadata.get('training_data_size')
        if training_data:
            info_parts.append(f"<b>Training size:</b> {training_data:,} samples")
        
        # Model architecture info
        model_type = metadata.get('model_type')
        if model_type:
            info_parts.append(f"<b>Architecture:</b> {model_type}")
        
        # Radiation type
        radiation = metadata.get('radiation_type', metadata.get('type'))
        if radiation:
            info_parts.append(f"<b>Radiation:</b> {radiation}")
        
        # Version info
        version = metadata.get('version')
        if version:
            info_parts.append(f"<b>Version:</b> {version}")
        
        # License info
        license_info = metadata.get('license')
        if license_info:
            info_parts.append(f"<b>License:</b> {license_info}")
        
        return '<br>'.join(info_parts) if info_parts else "<i>No additional information available</i>"
    
    @property
    def download_button(self) -> widgets.Button:
        """Get the download button widget for external access"""
        return self._widgets['download_button']
    
    def set_download_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the callback function for download button clicks.
        
        Args:
            callback: Function that will be called with model info when download is clicked
        """
        self._download_callback = callback
    
    def _clear_selection(self):
        """Clear the current model selection"""
        self.selected_config = None
        self.selected_model = None
        self._widgets['download_button'].disabled = True
        self._widgets['download_button'].tooltip = 'Select a model to enable download'
        self._widgets['selection_status'].value = "<i>Click a model row to select it</i>"
    
    def _select_model(self, model, index: int):
        """Handle model selection"""
        model_id = model.get('modelId') if isinstance(model, dict) else model.modelId
        
        # Store selection - the model ID is the full repo ID for HF models
        self.selected_config = model_id
        self.selected_model = model_id

    @property
    def selected_model_config_name(self) -> str:
        """Get the configuration name of the currently selected model"""
        if self.selected_model is None:
            return
        return self.selected_model.split('/')[-1]

    @property
    def selected_model_data(self) -> Optional[Dict[str, Any]]:
        """Get the data of the currently selected model"""
        if self.selected_model is None:
            return
        for model in self.models_data:
            if model.get('modelId') == self.selected_model:
                return model
        return None
        
    def get_selected_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently selected model"""
        if self.selected_config is None:
            return None
                
        return {
            'repo_id': self.organization ,
            'config_name': self.selected_model_config_name,
            'model_name': self.selected_model,
            'model_data': self.selected_model_data
        }
