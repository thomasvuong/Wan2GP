import gradio as gr
from plugin_system import WAN2GPPlugin

class LoraSetterPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Lora Multiplier Setter"
        self.version = "1.0.0"
        self.request_component("loras_multipliers")

    def post_ui_setup(self, components: dict) -> dict:
        target_component = components.get("loras_multipliers")
        if target_component is None:
            return {}
            
        def create_test_component():
            return gr.HTML(
                value="<div style='padding: 10px; background: #f0f0f0; margin: 10px 0; border: 2px solid red;'>"
                      "Test Component - Inserted by LoraSetterPlugin"
                      "</div>"
            )

        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_test_component
        )

        return {
            target_component: gr.update(value="test")
        }