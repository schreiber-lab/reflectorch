"""
Log Widget for Jupyter Interfaces

A clean, reusable log widget with print redirection capabilities.
"""

import sys
import contextlib
from io import StringIO
from typing import Optional
import ipywidgets as widgets


class LogWidget:
    """
    A clean log widget with print redirection capabilities.
    
    Features:
    - Hidden by default, auto-shows when messages arrive
    - Clear and toggle visibility controls
    - Context manager for print redirection
    - Clean, professional styling
    
    Example:
        ```python
        log_widget = LogWidget()
        
        # Use in layout
        layout = widgets.VBox([
            main_content,
            log_widget.widget
        ])
        
        # Redirect prints
        with log_widget.capture_prints():
            print("This goes to the log widget")
        
        # Or use the convenience method
        log_widget.log("Direct message to log")
        ```
    """
    
    def __init__(self, 
                 height: str = '150px',
                 hidden_by_default: bool = True,
                 auto_show_on_message: bool = True):
        """
        Initialize the log widget.
        
        Args:
            height: Height of the log output area
            hidden_by_default: Whether to start with log hidden
            auto_show_on_message: Whether to auto-show log when messages arrive
        """
        self.auto_show_on_message = auto_show_on_message
        self._create_widgets(height, hidden_by_default)
        self._setup_event_handlers()
    
    def _create_widgets(self, height: str, hidden_by_default: bool):
        """Create the log widget components"""
        # Create log output area
        self.output = widgets.Output(
            layout=widgets.Layout(
                height=height,
                width='100%',
                border='1px solid #ccc',
                overflow='auto',
                display='none' if hidden_by_default else ''
            )
        )
        
        # Create control buttons
        self.clear_button = widgets.Button(
            description="Clear Log",
            button_style='warning',
            tooltip='Clear all log messages',
            layout=widgets.Layout(width='100px')
        )
        
        self.toggle_button = widgets.Button(
            description="Show Log" if hidden_by_default else "Hide Log",
            button_style='info',
            tooltip='Toggle log visibility',
            layout=widgets.Layout(width='100px')
        )
        
        # Create label
        label = widgets.HTML("<b>Log Messages:</b>")
        
        # Create controls layout
        controls = widgets.HBox([
            label,
            widgets.HTML("&nbsp;" * 10),  # Spacer
            self.clear_button,
            self.toggle_button
        ])
        
        # Complete widget
        self.widget = widgets.VBox([
            controls,
            self.output
        ])
    
    def _setup_event_handlers(self):
        """Setup button event handlers"""
        def clear_log(_):
            self.output.clear_output()
        
        def toggle_log(_):
            if self.output.layout.display == 'none':
                self.show()
            else:
                self.hide()
        
        self.clear_button.on_click(clear_log)
        self.toggle_button.on_click(toggle_log)
    
    def show(self):
        """Show the log output area"""
        self.output.layout.display = ''
        self.toggle_button.description = "Hide Log"
    
    def hide(self):
        """Hide the log output area"""
        self.output.layout.display = 'none'
        self.toggle_button.description = "Show Log"
    
    def is_visible(self) -> bool:
        """Check if log is currently visible"""
        return self.output.layout.display != 'none'
    
    def clear(self):
        """Clear all log messages"""
        self.output.clear_output()
    
    def log(self, message: str):
        """
        Add a message directly to the log.
        
        Args:
            message: Message to add to the log
        """
        if self.auto_show_on_message and not self.is_visible():
            self.show()
        
        with self.output:
            print(message)
    
    @contextlib.contextmanager
    def capture_prints(self):
        """
        Context manager to redirect print statements to the log widget.
        
        Example:
            with log_widget.capture_prints():
                print("This goes to the log")
                some_function_that_prints()
        """
        if self.output is None:
            # Fallback to normal printing if widget not available
            yield
            return
        
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = StringIO()
        
        try:
            sys.stdout = captured_output
            yield
        finally:
            sys.stdout = old_stdout
            
            # Get captured content and add to log
            content = captured_output.getvalue()
            if content.strip():  # Only add if there's actual content
                if self.auto_show_on_message and not self.is_visible():
                    self.show()
                
                with self.output:
                    print(content.rstrip())  # Remove trailing newlines


class LoggedOperation:
    """
    Decorator/context manager for operations that should log to a specific log widget.
    
    Example:
        ```python
        log_widget = LogWidget()
        
        # As context manager
        with LoggedOperation(log_widget):
            print("This operation is logged")
            do_something()
        
        # As decorator
        @LoggedOperation(log_widget)
        def my_function():
            print("Function output goes to log")
        ```
    """
    
    def __init__(self, log_widget: LogWidget):
        self.log_widget = log_widget
    
    def __enter__(self):
        self.context = self.log_widget.capture_prints()
        return self.context.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.context.__exit__(exc_type, exc_val, exc_tb)
    
    def __call__(self, func):
        """Decorator functionality"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Convenience function for quick log widget creation
def create_log_widget(height: str = '150px', 
                     hidden_by_default: bool = True,
                     auto_show_on_message: bool = True) -> LogWidget:
    """
    Convenience function to create a log widget with common settings.
    
    Args:
        height: Height of the log output area
        hidden_by_default: Whether to start with log hidden
        auto_show_on_message: Whether to auto-show log when messages arrive
    
    Returns:
        Configured LogWidget instance
    """
    return LogWidget(
        height=height,
        hidden_by_default=hidden_by_default,
        auto_show_on_message=auto_show_on_message
    )
