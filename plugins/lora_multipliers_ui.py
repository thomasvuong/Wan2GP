import gradio as gr
from plugin_system import WAN2GPPlugin

class LoraMultipliersUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Lora Multipliers UI"
        self.version = "1.0.0"
        self.request_component("loras_multipliers")
        self.request_component("loras_choices")
        self.request_component("guidance_phases")
        self.request_component("num_inference_steps")
        self.request_component("main")

    def post_ui_setup(self, components: dict) -> dict:
        loras_multipliers = components.get("loras_multipliers")
        loras_choices = components.get("loras_choices")
        guidance_phases = components.get("guidance_phases")
        num_inference_steps = components.get("num_inference_steps")
        main_ui_block = components.get("main")

        if not all([loras_multipliers, loras_choices, guidance_phases, num_inference_steps, main_ui_block]):
            print("LoraMultipliersUIPlugin: Could not find all required components. UI will not be created.")
            return {}

        def create_and_wire_ui():
            MAX_LORA_SLIDERS = 15
            MAX_STEP_SPLITS = 5

            css = """
            <style>
                #lora_builder_main_group {
                    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
                }
                .lora-main-container {
                    border: 1px solid var(--border-color-primary);
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px !important;
                    background-color: var(--background-fill-secondary);
                }
                .lora-main-container > .gr-row {
                    margin-bottom: 8px;
                    justify-content: space-between;
                    align-items: center;
                }
                .lora-step-split-container {
                    border: 1px dashed var(--border-color-accent);
                    border-radius: 6px;
                    padding: 10px;
                    margin-top: 8px;
                }
                .lora-main-container button {
                    padding: 4px 12px !important;
                    font-size: 1.2em !important;
                    min-width: fit-content !important;
                    flex-grow: 0;
                    background: var(--button-secondary-background-fill);
                    color: var(--button-secondary-text-color);
                    border: 1px solid var(--button-secondary-border-color);
                }
                .lora-main-container button:hover {
                    background: var(--button-secondary-background-fill-hover);
                    border-color: var(--button-secondary-border-color-hover);
                }
                .lora-step-split-container > .gr-row {
                    gap: 16px;
                    align-items: end; 
                }
                .lora-step-split-container .form {
                    flex-grow: 1;
                }
            </style>
            """
            
            def update_slider_ui_and_textbox(selected_lora_indices, guidance_phases_val, current_multipliers_str, total_steps, current_split_counts, triggered_lora_index=-1):
                if triggered_lora_index != -1:
                    if current_split_counts[triggered_lora_index] < MAX_STEP_SPLITS:
                        current_split_counts[triggered_lora_index] += 1
                elif triggered_lora_index == -1:
                    current_split_counts = [1] * MAX_LORA_SLIDERS
                
                ui_updates = []
                textbox_strings = []
                multipliers_per_lora = current_multipliers_str.split(' ')

                for i in range(MAX_LORA_SLIDERS):
                    if i < len(selected_lora_indices):
                        lora_name = selected_lora_indices[i]
                        ui_updates.extend([gr.update(visible=True), gr.update(value=f"### {lora_name}")])
                        
                        num_splits_for_this_lora = current_split_counts[i]
                        steps_and_phases_str = multipliers_per_lora[i] if i < len(multipliers_per_lora) else ""
                        multipliers_per_step = steps_and_phases_str.split(',')

                        steps_per_split = total_steps
                        remainder = total_steps % num_splits_for_this_lora
                        start_step = 0
                        
                        lora_step_strings = []
                        for j in range(MAX_STEP_SPLITS):
                            if j < num_splits_for_this_lora:
                                end_step = start_step + steps_per_split + (1 if j < remainder else 0)
                                step_title = f"**Steps {start_step + 1} to {end_step}**"
                                start_step = end_step
                                ui_updates.extend([gr.update(visible=True), gr.update(value=step_title)])
                                
                                multipliers_per_phase = multipliers_per_step[j].split(';') if j < len(multipliers_per_step) else ['1.0'] * 3
                                phase_values_for_textbox = []
                                
                                for k in range(3):
                                    try: phase_value = float(multipliers_per_phase[k])
                                    except (ValueError, IndexError): phase_value = 1.0
                                    
                                    is_visible = (k + 1) <= guidance_phases_val
                                    ui_updates.append(gr.update(visible=is_visible, value=phase_value))
                                    if is_visible:
                                        formatted_value = str(int(phase_value)) if phase_value == int(phase_value) else f"{phase_value:.2f}".rstrip('0').rstrip('.')
                                        phase_values_for_textbox.append(formatted_value)
                                
                                if phase_values_for_textbox:
                                    lora_step_strings.append(";".join(phase_values_for_textbox))
                            else:
                                ui_updates.extend([gr.update(visible=False), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                        
                        if lora_step_strings:
                            textbox_strings.append(",".join(lora_step_strings))
                    else:
                        ui_updates.extend([gr.update(visible=False), gr.update(value="")])
                        for _ in range(MAX_STEP_SPLITS):
                            ui_updates.extend([gr.update(visible=False), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                
                new_textbox_value = " ".join(textbox_strings)
                return [current_split_counts, gr.update(value=new_textbox_value)] + ui_updates

            def update_textbox_from_sliders(selected_loras, guidance_phases_val, split_counts, *all_slider_values_flat):
                textbox_strings = []
                slider_cursor = 0
                for i in range(MAX_LORA_SLIDERS):
                    if i < len(selected_loras):
                        num_splits_for_this_lora = split_counts[i]
                        lora_step_strings = []
                        for j in range(MAX_STEP_SPLITS):
                            phase_values_for_textbox = []
                            if j < num_splits_for_this_lora:
                                for k in range(3):
                                    is_visible = (k + 1) <= guidance_phases_val
                                    if is_visible:
                                        phase_value = all_slider_values_flat[slider_cursor + k]
                                        formatted_value = str(int(phase_value)) if phase_value == int(phase_value) else f"{phase_value:.2f}".rstrip('0').rstrip('.')
                                        phase_values_for_textbox.append(formatted_value)
                            if phase_values_for_textbox:
                                lora_step_strings.append(";".join(phase_values_for_textbox))
                            slider_cursor += 3
                        if lora_step_strings:
                            textbox_strings.append(",".join(lora_step_strings))
                    else:
                        slider_cursor += MAX_STEP_SPLITS * 3
                new_textbox_value = " ".join(textbox_strings)
                return gr.update(value=new_textbox_value)

            with gr.Accordion("Dynamic Lora Multipliers Adjustments", open=True) as main_accordion:
                gr.HTML(value=css)
                lora_split_counts = gr.State([1] * MAX_LORA_SLIDERS)
                lora_slider_ui_groups = []

                update_mults_btn = gr.Button(visible=False, elem_id="lora_mults_update_btn")

                with gr.Group(elem_id="lora_builder_main_group"):
                    for i in range(MAX_LORA_SLIDERS):
                        with gr.Column(visible=False, elem_classes="lora-main-container") as lora_main_group:
                            with gr.Row(variant="compact"):
                                lora_name_md = gr.Markdown()
                                split_steps_btn = gr.Button("Split Steps")
                            
                            split_groups = []
                            for j in range(MAX_STEP_SPLITS):
                                with gr.Column(visible=False, elem_classes="lora-step-split-container") as lora_step_group:
                                    step_range_md = gr.Markdown()
                                    with gr.Row():
                                        phase1_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Phase 1", interactive=True)
                                        phase2_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Phase 2", interactive=True, visible=False)
                                        phase3_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Phase 3", interactive=True, visible=False)
                                    split_groups.append({ "group": lora_step_group, "title": step_range_md, "sliders": [phase1_slider, phase2_slider, phase3_slider] })
                            
                            lora_slider_ui_groups.append({ "main_group": lora_main_group, "name": lora_name_md, "split_button": split_steps_btn, "splits": split_groups })

            slider_ui_outputs = []
            all_sliders_flat = []
            for group in lora_slider_ui_groups:
                slider_ui_outputs.extend([group["main_group"], group["name"]])
                for split in group["splits"]:
                    slider_ui_outputs.extend([split["group"], split["title"], *split["sliders"]])
                    all_sliders_flat.extend(split["sliders"])
            
            events_to_trigger_ui_update = [loras_choices.change, guidance_phases.change, num_inference_steps.change]
            for event in events_to_trigger_ui_update:
                event(
                    fn=update_slider_ui_and_textbox,
                    inputs=[loras_choices, guidance_phases, loras_multipliers, num_inference_steps, lora_split_counts],
                    outputs=[lora_split_counts, loras_multipliers] + slider_ui_outputs,
                    show_progress="hidden"
                )

            for i, group in enumerate(lora_slider_ui_groups):
                group["split_button"].click(
                    fn=update_slider_ui_and_textbox,
                    inputs=[loras_choices, guidance_phases, loras_multipliers, num_inference_steps, lora_split_counts, gr.State(i)],
                    outputs=[lora_split_counts, loras_multipliers] + slider_ui_outputs,
                    show_progress="hidden"
                )

            update_mults_btn.click(
                fn=update_textbox_from_sliders,
                inputs=[loras_choices, guidance_phases, lora_split_counts] + all_sliders_flat,
                outputs=[loras_multipliers],
                show_progress="hidden"
            )

            for slider in all_sliders_flat:
                slider.release(fn=None, js="() => { document.getElementById('lora_mults_update_btn').click() }")

            main_ui_block.load(
                fn=update_slider_ui_and_textbox,
                inputs=[loras_choices, guidance_phases, loras_multipliers, num_inference_steps, lora_split_counts],
                outputs=[lora_split_counts, loras_multipliers] + slider_ui_outputs,
                show_progress="hidden"
            )

            return main_accordion

        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_and_wire_ui
        )
        
        return {
            loras_multipliers: gr.update(elem_id="loras_multipliers_textbox", interactive=True)
        }