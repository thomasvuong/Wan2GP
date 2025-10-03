import os
import sys
import importlib.util
import inspect
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import gradio as gr

@dataclass
class PluginTab:
    id: str
    label: str
    component_constructor: callable
    position: int = -1

class WAN2GPPlugin:
    def __init__(self):
        self.tabs: Dict[str, PluginTab] = {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self._component_requests: List[str] = []
        
    def setup_ui(self) -> None:
        pass
        
    def add_tab(self, tab_id: str, label: str, component_constructor: callable, position: int = -1):
        self.tabs[tab_id] = PluginTab(id=tab_id, label=label, component_constructor=component_constructor, position=position)

    def request_component(self, component_id: str) -> None:
        if component_id not in self._component_requests:
            self._component_requests.append(component_id)
            
    @property
    def component_requests(self) -> List[str]:
        return self._component_requests.copy()

    def post_ui_setup(self, components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        return {}

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self._load_builtin_plugins()
        
    def _load_builtin_plugins(self) -> None:
        pass
    
    def load_plugin(self, plugin_path: str) -> bool:
        try:
            module_name = os.path.splitext(os.path.basename(plugin_path))[0]
            
            spec = importlib.util.spec_from_file_location(f"wan2gp_plugins.{module_name}", plugin_path)
            if spec is None or spec.loader is None:
                print(f"Error: Could not load plugin {plugin_path}")
                return False
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                    plugin = obj()
                    plugin.setup_ui()
                    self.plugins[module_name] = plugin
                    print(f"Loaded plugin: {plugin.name}")
                    return True
                    
            print(f"No plugin class found in {plugin_path}")
            return False
            
        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_plugins_from_directory(self, directory: str) -> None:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
            return
            
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                self.load_plugin(os.path.join(directory, filename))
    
    def get_plugin(self, plugin_id: str) -> Optional[WAN2GPPlugin]:
        return self.plugins.get(plugin_id)
    
    def get_all_plugins(self) -> Dict[str, WAN2GPPlugin]:
        return self.plugins.copy()
    
    def setup_ui(self) -> Dict[str, Dict[str, Any]]:
        tabs = {}
        
        for plugin_id, plugin in self.plugins.items():
            try:
                for tab_id, tab in plugin.tabs.items():
                    tabs[tab_id] = {
                        'label': tab.label,
                        'component_constructor': tab.component_constructor,
                        'position': tab.position
                    }
            except Exception as e:
                print(f"Error in setup_ui for plugin {plugin_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return {'tabs': tabs}

    def run_post_ui_setup(self, all_components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        all_updates: Dict[gr.components.Component, Union[gr.update, Any]] = {}
        for plugin_id, plugin in self.plugins.items():
            try:
                for comp_id in plugin._component_requests:
                    if comp_id in all_components:
                        setattr(plugin, comp_id, all_components[comp_id])

                requested_components = {
                    comp_id: all_components[comp_id]
                    for comp_id in plugin._component_requests
                    if comp_id in all_components
                }

                if not requested_components and plugin._component_requests:
                    print(f"Warning: Plugin '{plugin.name}' requested components that were not found: "
                          f"{[c for c in plugin._component_requests if c not in all_components]}")

                updates = plugin.post_ui_setup(requested_components)
                if updates:
                    all_updates.update(updates)
            except Exception as e:
                print(f"Error in post_ui_setup for plugin {plugin_id}: {str(e)}")
                import traceback
                traceback.print_exc()

        return all_updates