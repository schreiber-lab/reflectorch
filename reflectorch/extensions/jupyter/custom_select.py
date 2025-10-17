import ipywidgets as W
from IPython.display import display, HTML
from typing import List, Dict, Tuple, Callable, Optional, Any


class CustomSelect:
    """
    A styled select widget for displaying tabular data with monospace formatting.
    
    Features:
    - Clean, modern styling with hover effects
    - Automatic column width calculation
    - Header display
    - Click callback support
    - Easy data updates
    """
    
    # Inject CSS once when class is first used
    _css_injected = False
    
    @classmethod
    def _inject_css(cls):
        if not cls._css_injected:
            display(HTML("""
            <style>
            .custom-select-container {
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            }
            
            .custom-select-header {
                margin: 0;
                padding: 8px 12px;
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                font-weight: 600;
                font-size: 13px;
                background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
                border: 1px solid #dee2e6;
                border-bottom: 2px solid #adb5bd;
                border-radius: 6px 6px 0 0;
                color: #495057;
                letter-spacing: 0.3px;
            }
            
            .custom-select select {
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                font-size: 13px;
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-top: none;
                border-radius: 0 0 6px 6px;
                padding: 6px 12px;
                color: #212529;
                line-height: 1.6;
                transition: all 0.2s ease;
            }
            
            .custom-select select:hover {
                background-color: #f8f9fa;
            }
            
            .custom-select select:focus {
                outline: none;
                border-color: #0d6efd;
                box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.1);
                background-color: #ffffff;
            }
            
            .custom-select select option {
                padding: 4px 8px;
            }
            
            .custom-select select option:hover {
                background-color: #e7f1ff;
            }
            
            .custom-select-details {
                margin-top: 12px;
                padding: 12px 16px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #495057;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            
            .custom-select-details i {
                color: #6c757d;
            }
            
            .custom-select-details b {
                color: #0d6efd;
                font-weight: 600;
            }
            </style>
            """))
            cls._css_injected = True
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        columns: List[Tuple[str, str]],
        width: str = "auto",
        max_rows: int = 10,
        show_details: bool = False,
        details_template: Optional[str] = None,
        layout: Optional[W.Layout] = None,
    ):
        """
        Initialize the CustomSelect widget.
        
        Args:
            data: List of dictionaries containing row data
            columns: List of tuples (header_label, data_key)
            width: CSS width for the select widget (e.g., '600px', 'auto')
            max_rows: Maximum number of visible rows
            show_details: Whether to show the details panel below
            details_template: Custom HTML template for details (receives row dict)
            layout: Layout for the VBox containing the widget
        """
        self._inject_css()
        
        self.data = data
        self.columns = columns
        self.width = width
        self.max_rows = max_rows
        self.show_details = show_details
        self.details_template = details_template
        self._callback: Optional[Callable] = None
        
        # Create widgets
        self._header_widget = W.HTML()
        self._select_widget = W.Select(layout=W.Layout(width=width))
        self._select_widget.add_class("custom-select")
        
        self._details_widget = W.HTML("<i>Select a row to view details…</i>")
        self._details_widget.add_class("custom-select-details")
        
        # Setup observer
        self._select_widget.observe(self._on_selection_change, names="value")
        
        # Build the widget
        self._update_display()
        
        # Container
        widgets = [self._header_widget, self._select_widget]
        if self.show_details:
            widgets.append(self._details_widget)

        if layout is not None:
            self.container = W.VBox(widgets, layout=layout)
        else:
            self.container = W.VBox(widgets)
        
        self.container.add_class("custom-select-container")
    
    def _calculate_column_widths(self) -> Dict[str, int]:
        """Calculate the width needed for each column."""
        widths = {}
        for header, key in self.columns:
            header_len = len(str(header))
            if self.data:
                max_data_len = max(len(str(row.get(key, ""))) for row in self.data)
                widths[key] = max(header_len, max_data_len)
            else:
                widths[key] = header_len
        return widths
    
    def _format_row(self, row: Dict[str, Any], widths: Dict[str, int]) -> str:
        """Format a row with proper column alignment."""
        parts = []
        for _, key in self.columns:
            value = str(row.get(key, ""))
            parts.append(value.ljust(widths[key]))
        return "  ".join(parts)
    
    def _update_display(self):
        """Update the header and select options based on current data."""
        if not self.data:
            self._header_widget.value = "<pre class='custom-select-header'><i>No data</i></pre>"
            self._select_widget.options = []
            return
        
        # Calculate widths
        widths = self._calculate_column_widths()
        
        # Build header
        header_parts = [h.ljust(widths[k]) for h, k in self.columns]
        header_text = "  ".join(header_parts)
        self._header_widget.value = f"<pre class='custom-select-header'>{header_text}</pre>"
        
        # Build options
        options = [(self._format_row(row, widths), row) for row in self.data]
        self._select_widget.options = options
        self._select_widget.rows = min(self.max_rows, len(self.data))
    
    def _on_selection_change(self, change):
        """Handle selection changes."""
        if change["name"] == "value" and change["new"] is not None:
            row = change["new"]
            
            # Update details
            if self.show_details:
                if self.details_template:
                    self._details_widget.value = self.details_template.format(**row)
                else:
                    # Default details display
                    details_parts = [f"<b>{k}:</b> {v}" for k, v in row.items()]
                    self._details_widget.value = " | ".join(details_parts)
            
            # Call user callback
            if self._callback:
                self._callback(row)
    
    def set_data(self, data: List[Dict[str, Any]]):
        """
        Update the data displayed in the select widget.
        
        Args:
            data: New list of dictionaries containing row data
        """
        self.data = data + [{}]
        self._update_display()
        if self.show_details:
            self._details_widget.value = "<i>Select a row to view details…</i>"
    
    def set_columns(self, columns: List[Tuple[str, str]]):
        """
        Update the columns definition.
        
        Args:
            columns: New list of tuples (header_label, data_key)
        """
        self.columns = columns
        self._update_display()
    
    def on_select(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback function to be called when a row is selected.
        
        Args:
            callback: Function that receives the selected row dictionary
        """
        self._callback = callback
        return self
    
    def get_selected(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected row, or None if nothing is selected."""
        return self._select_widget.value
    
    def set_selected_index(self, index: int):
        """Select a row by index."""
        if 0 <= index < len(self.data):
            self._select_widget.value = self.data[index]
    
    def clear_selection(self):
        """Clear the current selection."""
        self._select_widget.value = None
        if self.show_details:
            self._details_widget.value = "<i>Select a row to view details…</i>"
    
    def display(self):
        """Display the widget."""
        display(self.container)
    
    def __repr__(self):
        return f"CustomSelect(rows={len(self.data)}, columns={len(self.columns)})"
