import gradio as gr
from plugin_system import WAN2GPPlugin

class LoraSetterPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Lora Multiplier Setter"
        self.version = "1.0.0"
        self.request_component("loras_multipliers")

    def post_ui_setup(self, components: dict) -> dict:
        if hasattr(self, 'loras_multipliers'):
            return {
                self.loras_multipliers: gr.Textbox(value="test")
            }
        
        return {}