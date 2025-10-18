import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import json
import traceback

class LoraMultipliersUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Lora Multipliers UI"
        self.version = "1.0.9"
        self.description = "Dynamically set lora multipliers with a fast, JavaScript-powered UI."
        self.target_tabs = ['generate', 'edit']
        self.request_component("loras_multipliers")
        self.request_component("loras_choices")
        self.request_component("guidance_phases")
        self.request_component("num_inference_steps")
        self.request_component("main")
        self.previous_loras_state = {}

    def post_ui_setup(self, components: dict) -> dict:
        tab_id = components.get('__tab_id__', 'unknown_tab')
        if tab_id not in self.previous_loras_state:
            self.previous_loras_state[tab_id] = {'loras': [], 'accelerators': []}
        
        try:
            loras_multipliers = components["loras_multipliers"]
            loras_choices = components["loras_choices"]
            guidance_phases = components["guidance_phases"]
            num_inference_steps = components["num_inference_steps"]
            main_ui_block = components["main"]

            def create_and_wire_ui():
                container_id = f"lora_multiplier_ui_container_{tab_id}"
                update_btn_id = f"lora_mults_update_btn_{tab_id}"
                hidden_input_id = f"lora_mults_hidden_input_{tab_id}"
                js_renderer_func = f"wgpLoraUIRenderer_{tab_id}"
                
                css = f"""
                <style>
                    #{container_id} {{ font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif; }}
                    .lora-main-container {{ border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px; margin-bottom: 8px !important; background-color: var(--background-fill-secondary); }}
                    .lora-main-container .gr-row {{ margin-bottom: 8px; justify-content: space-between; align-items: center; }}
                    .lora-main-container .gr-row h3 {{ margin-top: 0; margin-bottom: 10;}}
                    .lora-step-split-container {{ border: 1px dashed var(--border-color-accent); border-radius: 6px; padding: 10px; margin-top: 8px; }}
                    .lora-slider-row {{ display: flex; gap: 16px; align-items: end; }}
                    .lora-main-container button, .lora-main-container .wgp-lora-button {{ padding: 4px 12px !important; font-size: 1em !important; min-width: fit-content !important; flex-grow: 0; background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); border: 1px solid var(--button-secondary-border-color); border-radius: 4px; cursor: pointer; }}
                    .lora-main-container button:hover, .lora-main-container .wgp-lora-button:hover {{ background: var(--button-secondary-background-fill-hover); border-color: var(--button-secondary-border-color-hover); }}
                    .lora-section-header h3 {{ border-bottom: 1px solid var(--border-color-primary); padding-bottom: 4px; margin-top: 16px; margin-bottom: 8px; }}
                    #{container_id} > .lora-section-header:first-child h3 {{
                        margin-top: 0;
                    }}
                    .lora-main-container > h3:first-child {{ margin-top: 0; }}
                    .lora-slider-group {{
                        flex: 1;
                        display: flex;
                        flex-direction: column;
                    }}
                    .lora-slider-input-wrapper {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }}
                    .lora-slider-group label {{ display: block; color: var(--body-text-color); font-size: 0.9em; margin-bottom: 4px; }}
                    .lora-slider-group input[type=range] {{
                        flex-grow: 1;
                        width: auto;
                    }}
                    .lora-slider-group input[type=number] {{
                        width: 60px;
                        padding: 4px;
                        border: 1px solid var(--border-color-primary);
                        border-radius: 4px;
                        background-color: var(--input-background-fill);
                        color: var(--input-text-color);
                        font-size: 0.9em;
                        text-align: center;
                    }}
                    .lora-slider-group input[type=number]::-webkit-inner-spin-button, 
                    .lora-slider-group input[type=number]::-webkit-outer-spin-button {{ 
                      -webkit-appearance: none; 
                      margin: 0; 
                    }}
                    .lora-slider-group input[type=number] {{
                      -moz-appearance: textfield;
                    }}
                    .hidden {{ display: none !important; }}
                    .lora-split-title {{ margin-bottom: 8px; }}
                </style>
                """

                main_js_script = f"""
                () => {{
                    const debounce = (func, delay) => {{
                        let timeout;
                        return (...args) => {{
                            clearTimeout(timeout);
                            timeout = setTimeout(() => func.apply(this, args), delay);
                        }};
                    }};

                    const updatePythonTextbox = debounce(() => {{
                        const container = document.getElementById('{container_id}');
                        if (!container) return;

                        const loras = Array.from(container.querySelectorAll('.lora-main-container'));
                        const textboxStrings = [];

                        for (const loraEl of loras) {{
                            const splits = Array.from(loraEl.querySelectorAll('.lora-step-split-container'));
                            const loraStepStrings = [];
                            for (const splitEl of splits) {{
                                const sliders = Array.from(splitEl.querySelectorAll('input[type=range]'));
                                const phaseValues = sliders
                                    .filter(s => !s.closest('.lora-slider-group').classList.contains('hidden'))
                                    .map(s => {{
                                        const val = parseFloat(s.value);
                                        return val % 1 === 0 ? String(val) : val.toFixed(2).replace(/\\.?0+$/, '');
                                    }});
                                if (phaseValues.length > 0) {{
                                    loraStepStrings.push(phaseValues.join(';'));
                                }}
                            }}
                            if (loraStepStrings.length > 0) {{
                               textboxStrings.push(loraStepStrings.join(','));
                            }}
                        }}

                        const separatorIndex = parseInt(container.dataset.separatorIndex || '-1');
                        let finalString = "";
                        if (separatorIndex > 0 && separatorIndex <= textboxStrings.length) {{
                            const part1 = textboxStrings.slice(0, separatorIndex).join(' ');
                            const part2 = textboxStrings.slice(separatorIndex).join(' ');
                            finalString = part1 + '|' + part2;
                        }} else {{
                            finalString = textboxStrings.join(' ');
                        }}
                        
                        const hiddenInput = document.querySelector('#{hidden_input_id} textarea');
                        const updateButton = document.getElementById('{update_btn_id}');
                        if (hiddenInput && updateButton) {{
                            hiddenInput.value = finalString;
                            hiddenInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            updateButton.click();
                        }}
                    }}, 200);

                    function createSlider(phase, value, isVisible) {{
                        const container = document.createElement('div');
                        container.className = 'lora-slider-group';
                        if (!isVisible) container.classList.add('hidden');

                        const initialValue = parseFloat(value);

                        container.innerHTML = `
                            <label>Phase ${{phase}}</label>
                            <div class="lora-slider-input-wrapper">
                                <input type="range" min="0" max="1" step="0.05" value="${{initialValue}}">
                                <input type="number" min="0" max="1" step="0.05" value="${{initialValue.toFixed(2)}}">
                            </div>
                        `;

                        const rangeInput = container.querySelector('input[type="range"]');
                        const numberInput = container.querySelector('input[type="number"]');

                        const syncAndUpdate = (source) => {{
                            let val = parseFloat(source.value);
                            if (isNaN(val)) val = 0;
                            val = Math.max(0, Math.min(1, val));

                            if (source === rangeInput) {{
                                numberInput.value = val.toFixed(2);
                            }} else {{
                                rangeInput.value = val;
                                if (source.value !== val.toFixed(2)) {{
                                    numberInput.value = val.toFixed(2);
                                }}
                            }}
                            updatePythonTextbox();
                        }};

                        rangeInput.addEventListener('input', () => syncAndUpdate(rangeInput));
                        numberInput.addEventListener('input', () => syncAndUpdate(numberInput));

                        return container;
                    }}

                    function createStepSplit(loraIndex, splitIndex, values, guidancePhases, stepText) {{
                        const splitContainer = document.createElement('div');
                        splitContainer.className = 'lora-step-split-container';
                        const sliderRow = document.createElement('div');
                        sliderRow.className = 'lora-slider-row';
                        const title = document.createElement('div');
                        title.className = 'lora-split-title';
                        title.innerHTML = `<strong>${{stepText}}</strong>`;
                        splitContainer.appendChild(title);
                        for (let i = 0; i < 3; i++) {{
                            const isVisible = (i + 1) <= guidancePhases;
                            const sliderValue = values[i] !== undefined ? values[i] : 1.0;
                            sliderRow.appendChild(createSlider(i + 1, sliderValue, isVisible));
                        }}
                        splitContainer.appendChild(sliderRow);
                        return splitContainer;
                    }}

                    function updateRejoinVisibility(loraContainer) {{
                        if (!loraContainer) return;
                        const splits = loraContainer.querySelectorAll('.lora-step-split-container');
                        const rejoinBtn = loraContainer.querySelector('.rejoin-btn');
                        if (rejoinBtn) {{ rejoinBtn.style.display = splits.length > 1 ? 'inline-block' : 'none'; }}
                    }}

                    function handleSplit(e) {{
                        const loraIndex = parseInt(e.target.dataset.loraIndex);
                        const loraContainer = document.getElementById(`lora-container-{tab_id}-${{loraIndex}}`);
                        const newSplit = createStepSplit(loraIndex, -1, [1.0, 1.0, 1.0], window.wgp_guidance_phases_{tab_id}, "");
                        loraContainer.appendChild(newSplit);
                        recalculateStepRanges(loraContainer);
                        updateRejoinVisibility(loraContainer);
                        updatePythonTextbox();
                    }}

                    function handleRejoin(e) {{
                        const loraIndex = parseInt(e.target.dataset.loraIndex);
                        const loraContainer = document.getElementById(`lora-container-{tab_id}-${{loraIndex}}`);
                        const splits = loraContainer.querySelectorAll('.lora-step-split-container');
                        if (splits.length > 1) {{
                            splits[splits.length - 1].remove();
                            recalculateStepRanges(loraContainer);
                            updateRejoinVisibility(loraContainer);
                            updatePythonTextbox();
                        }}
                    }}

                    function recalculateStepRanges(loraContainer) {{
                        const totalSteps = window.wgp_total_steps_{tab_id} || 1;
                        const splits = loraContainer.querySelectorAll('.lora-step-split-container');
                        const numSplits = splits.length;
                        if (numSplits === 0) return;
                        const stepsPerSplit = Math.floor(totalSteps / numSplits);
                        const remainder = totalSteps % numSplits;
                        let startStep = 0;
                        splits.forEach((split, i) => {{
                            const stepsInThisSplit = stepsPerSplit + (i < remainder ? 1 : 0);
                            const endStep = startStep + stepsInThisSplit;
                            const titleStrong = split.querySelector('.lora-split-title strong');
                            if(titleStrong) {{
                                const displayEnd = Math.max(startStep + 1, endStep);
                                titleStrong.textContent = `Steps ${{startStep + 1}} to ${{displayEnd}}`;
                            }}
                            startStep = endStep;
                        }});
                    }}

                    window.{js_renderer_func} = (jsonData) => {{
                        let data;
                        try {{ data = JSON.parse(jsonData); }} catch (e) {{ return; }}
                        if (!data) return;
                        const container = document.getElementById('{container_id}');
                        if (!container) return;
                        
                        container.innerHTML = '';
                        window.wgp_guidance_phases_{tab_id} = data.guidance_phases;
                        window.wgp_total_steps_{tab_id} = data.total_steps;
                        container.dataset.separatorIndex = data.separator_index;

                        const createHeader = (text) => {{
                            const headerDiv = document.createElement('div');
                            headerDiv.className = 'lora-section-header';
                            headerDiv.innerHTML = `<h3>${{text}}</h3>`;
                            return headerDiv;
                        }};

                        if(data.separator_index > 0 && data.loras.length > 0) {{
                            container.appendChild(createHeader('Accelerator LoRAs'));
                        }}

                        data.loras.forEach((lora, i) => {{
                            if ((data.separator_index === -1 && i === 0) || data.separator_index === i) {{
                               container.appendChild(createHeader('User LoRAs'));
                            }}
                            const loraContainer = document.createElement('div');
                            loraContainer.className = 'lora-main-container';
                            loraContainer.id = `lora-container-{tab_id}-${{i}}`;
                            loraContainer.innerHTML = `<div class="gr-row"><h3>${{lora.name}}</h3><div style="display:flex; gap: 8px;"><button class="wgp-lora-button split-btn" data-lora-index="${{i}}" type="button">Split Steps</button><button class="wgp-lora-button rejoin-btn" data-lora-index="${{i}}" type="button" style="display:none;">Rejoin Step</button></div></div>`;
                            lora.splits.forEach((split) => {{
                                loraContainer.appendChild(createStepSplit(i, -1, split.values, data.guidance_phases, ""));
                            }});
                            container.appendChild(loraContainer);
                            recalculateStepRanges(loraContainer);
                            updateRejoinVisibility(loraContainer);
                            loraContainer.querySelector('.split-btn').addEventListener('click', handleSplit);
                            loraContainer.querySelector('.rejoin-btn').addEventListener('click', handleRejoin);
                        }});
                    }};
                }}
                """

                def update_ui_data_from_python(selected_lora_names, multipliers_str, guidance_phases_val, total_steps):
                    try:
                        lora_names = selected_lora_names if selected_lora_names else []
                        num_selected_loras = len(lora_names)

                        all_stale_multipliers = [s for s in (multipliers_str or "").replace('|', ' ').split(' ') if s]
                        num_stale_multipliers = len(all_stale_multipliers)
                        
                        is_desynced = num_selected_loras != num_stale_multipliers
                        
                        original_accelerator_count = 0
                        if multipliers_str and '|' in multipliers_str:
                            parts = multipliers_str.split('|')
                            original_accelerator_count = len([s for s in parts[0].split(' ') if s])

                        lora_names_for_ui = lora_names
                        
                        if is_desynced:
                            multipliers_per_lora_str = ["1.0"] * num_selected_loras
                            
                            previous_state = self.previous_loras_state.get(tab_id, {'accelerators': []})
                            old_accelerators = set(previous_state.get('accelerators', []))
                            
                            remaining_accelerators = [lora for lora in lora_names if lora in old_accelerators]
                            user_loras = [lora for lora in lora_names if lora not in old_accelerators]
                            
                            lora_names_for_ui = remaining_accelerators + user_loras
                            new_separator_index = len(remaining_accelerators)

                            if new_separator_index == 0 or new_separator_index > len(lora_names_for_ui):
                                new_separator_index = -1
                        else:
                            multipliers_per_lora_str = all_stale_multipliers
                            new_separator_index = original_accelerator_count if original_accelerator_count > 0 else -1
                            lora_names_for_ui = lora_names

                        current_accelerators = lora_names_for_ui[:new_separator_index if new_separator_index != -1 else 0]
                        self.previous_loras_state[tab_id] = {
                            'loras': lora_names_for_ui,
                            'accelerators': current_accelerators
                        }

                        loras_data = []
                        for i, lora_name in enumerate(lora_names_for_ui):
                            lora_obj = {"name": lora_name, "splits": []}
                            steps_and_phases_str = multipliers_per_lora_str[i]
                            
                            for step_str in steps_and_phases_str.split(','):
                                phase_values = []
                                phase_strs = step_str.split(';')
                                for k in range(3):
                                    try:
                                        phase_values.append(float(phase_strs[k]))
                                    except (ValueError, IndexError):
                                        phase_values.append(1.0)
                                lora_obj["splits"].append({"values": phase_values})
                            loras_data.append(lora_obj)

                        payload = {
                            "loras": loras_data,
                            "guidance_phases": guidance_phases_val,
                            "total_steps": total_steps or 1,
                            "separator_index": new_separator_index,
                        }
                        
                        return json.dumps(payload)
                    except Exception:
                        traceback.print_exc()
                        return "{}"

                def update_textbox_from_js(new_value):
                    return gr.update(value=new_value)

                with gr.Accordion("Dynamic Lora Multipliers", open=True) as main_accordion:
                    gr.HTML(value=css)
                    gr.HTML(f"<div id='{container_id}'></div>")
                    
                    with gr.Row(visible=False):
                        hidden_input = gr.Text(elem_id=hidden_input_id)
                        update_button = gr.Button(elem_id=update_btn_id)
                    
                    ui_data_json = gr.Text(elem_id=f"ui_data_json_{tab_id}", visible=False)

                main_ui_block.load(fn=None, js=main_js_script)

                ui_data_json.change(
                    fn=None,
                    inputs=[ui_data_json],
                    js=f"(jsonData) => {{ if(window.{js_renderer_func}) window.{js_renderer_func}(jsonData); }}",
                    show_progress="hidden"
                )

                input_components = [loras_choices, loras_multipliers, guidance_phases, num_inference_steps]

                main_ui_block.load(
                    fn=update_ui_data_from_python,
                    inputs=input_components,
                    outputs=[ui_data_json],
                    show_progress="hidden"
                )

                events_to_trigger = [loras_choices.change, guidance_phases.change, num_inference_steps.change, loras_multipliers.blur]
                for event_fn in events_to_trigger:
                    event_fn(
                        fn=update_ui_data_from_python,
                        inputs=input_components,
                        outputs=[ui_data_json],
                        show_progress="hidden"
                    )

                update_button.click(
                    fn=update_textbox_from_js,
                    inputs=[hidden_input],
                    outputs=[loras_multipliers],
                    show_progress="hidden"
                )

                return main_accordion

            self.insert_after(
                target_component_id="loras_multipliers",
                new_component_constructor=create_and_wire_ui
            )
        
        except Exception:
            traceback.print_exc()
        
        return {}