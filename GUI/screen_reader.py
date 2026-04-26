from PyQt6.QtCore import QObject, QEvent, Qt
from PyQt6.QtWidgets import QPushButton, QCheckBox, QComboBox, QLabel, QDialog

class ScreenReaderFilter(QObject):
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.suppress_once = False

    def eventFilter(self, obj, event):
        if not self.app.screen_reader_enabled:
            return super().eventFilter(obj, event)

        # Handle keyboard focus and mouse hover
        if event.type() in [QEvent.Type.FocusIn, QEvent.Type.Enter]:
            if self.suppress_once:
                self.suppress_once = False
                return super().eventFilter(obj, event)
            
            # Avoid double-speaking if Enter and FocusIn happen together
            if hasattr(self, '_last_widget') and self._last_widget == obj:
                # If it's a hover after focus, or focus after hover, maybe skip?
                # Actually let's just use a small timer to prevent spam
                pass
            
            self._last_widget = obj
            self.narrate_widget(obj)
        
        return super().eventFilter(obj, event)

    def narrate_widget(self, widget):
        text_to_speak = ""
        
        if isinstance(widget, QPushButton):
            # Special case for gallery buttons
            if widget in getattr(self.app, 'poster_buttons', []):
                idx = self.app.poster_buttons.index(widget)
                poster = self.app.poster_data[idx]
                desc = poster.get("description", "Processing...")
                text_to_speak = f"Poster {idx + 1}. {desc}"
            else:
                text_to_speak = f"Button: {widget.text()}"
                
        elif hasattr(widget, 'accessible_name'):
            text_to_speak = widget.accessible_name
            
        elif isinstance(widget, QCheckBox):
            state = "checked" if widget.isChecked() else "unchecked"
            text_to_speak = f"Checkbox: {widget.text()}, {state}"
            
        elif isinstance(widget, QComboBox):
            text_to_speak = f"Combo box: {widget.currentText()}"

        if text_to_speak:
            self.app.speaker.speak(text_to_speak)
