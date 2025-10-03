import gradio as gr
from plugin_system import WAN2GPPlugin

class ExamplePlugin(WAN2GPPlugin):
    
    def __init__(self):
        super().__init__()
        self.name = "Example Plugin"
        self.version = "1.0.1"
        
    def create_hello_world_tab(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Hello from the Example Plugin!")
            with gr.Row():
                name = gr.Textbox(label="Enter your name")
                output = gr.Textbox(label="Output")
            greet_btn = gr.Button("Greet")
            
            greet_btn.click(
                fn=lambda n: f"Hello, {n}!",
                inputs=[name],
                outputs=output
            )
        return demo
        
    def setup_ui(self) -> None:
        self.add_tab(
            tab_id="example_plugin_tab",
            label="Example Plugin",
            component_constructor=self.create_hello_world_tab
        )