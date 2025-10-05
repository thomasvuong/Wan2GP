import os
import sys
import importlib.util
import inspect
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import gradio as gr
import traceback
import subprocess
import git

@dataclass
class InsertAfterRequest:
    target_component_id: str
    new_component_constructor: callable

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
        self._global_requests: List[str] = []
        self._insert_after_requests: List[InsertAfterRequest] = []
        self._setup_complete = False
        
    def setup_ui(self) -> None:
        pass
        
    def add_tab(self, tab_id: str, label: str, component_constructor: callable, position: int = -1):
        self.tabs[tab_id] = PluginTab(id=tab_id, label=label, component_constructor=component_constructor, position=position)

    def request_component(self, component_id: str) -> None:
        if component_id not in self._component_requests:
            self._component_requests.append(component_id)
            
    def request_global(self, global_name: str) -> None:
        if global_name not in self._global_requests:
            self._global_requests.append(global_name)
            
    @property
    def component_requests(self) -> List[str]:
        return self._component_requests.copy()

    @property
    def global_requests(self) -> List[str]:
        return self._global_requests.copy()

    def insert_after(self, target_component_id: str, new_component_constructor: callable) -> None:
        if not hasattr(self, '_insert_after_requests'):
            self._insert_after_requests = []
        self._insert_after_requests.append(
            InsertAfterRequest(
                target_component_id=target_component_id,
                new_component_constructor=new_component_constructor
            )
        )
        
    def post_ui_setup(self, components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        return {}

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self.plugins_dir = plugins_dir
        os.makedirs(self.plugins_dir, exist_ok=True)
        
    def install_plugin_from_url(self, git_url: str):
        if not git_url or not git_url.startswith("https://github.com/"):
            return "[Error] Invalid GitHub URL."

        try:
            repo_name = git_url.split('/')[-1].replace('.git', '')
            target_dir = os.path.join(self.plugins_dir, repo_name)

            if os.path.exists(target_dir):
                return f"[Warning] Plugin '{repo_name}' already exists. Please remove it manually to reinstall."

            gr.Info(f"Cloning '{repo_name}' into '{target_dir}'...")
            git.Repo.clone_from(git_url, target_dir)

            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                gr.Info(f"Installing dependencies for '{repo_name}'...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
                except subprocess.CalledProcessError as e:
                    return f"[Error] Failed to install dependencies for {repo_name}. Check console for details. Error: {e}"

            setup_path = os.path.join(target_dir, 'setup.py')
            if os.path.exists(setup_path):
                gr.Info(f"Running setup for '{repo_name}'...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', target_dir])
                except subprocess.CalledProcessError as e:
                    return f"[Error] Failed to run setup.py for {repo_name}. Check console for details. Error: {e}"
            
            init_path = os.path.join(target_dir, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    pass

            return f"[Success] Plugin '{repo_name}' installed. Please enable it in the list and restart WanGP."

        except git.exc.GitCommandError as e:
            return f"[Error] Git clone failed: {e.stderr}"
        except Exception as e:
            return f"[Error] An unexpected error occurred: {str(e)}"

    def discover_plugins(self) -> List[str]:
        discovered = []
        for item in os.listdir(self.plugins_dir):
            path = os.path.join(self.plugins_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, '__init__.py')):
                discovered.append(item)
        return sorted(discovered)

    def load_plugins_from_directory(self, enabled_plugins: List[str]) -> None:
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, self.plugins_dir)

        for plugin_dir_name in self.discover_plugins():
            if plugin_dir_name not in enabled_plugins:
                continue
            try:
                module = importlib.import_module(f"{plugin_dir_name}.plugin")

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                        plugin = obj()
                        plugin.setup_ui()
                        self.plugins[plugin_dir_name] = plugin
                        print(f"Loaded plugin: {plugin.name} (from {plugin_dir_name})")
                        break
            except Exception as e:
                print(f"Error loading plugin from directory {plugin_dir_name}: {e}")
                traceback.print_exc()

    def get_all_plugins(self) -> Dict[str, WAN2GPPlugin]:
        return self.plugins.copy()

    def inject_globals(self, global_references: Dict[str, Any]) -> None:
        for plugin_id, plugin in self.plugins.items():
            try:
                for global_name in plugin.global_requests:
                    if global_name in global_references:
                        setattr(plugin, global_name, global_references[global_name])
            except Exception as e:
                print(f"  [!] ERROR injecting globals for {plugin_id}: {str(e)}")

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
        return {'tabs': tabs}

    def run_post_ui_setup(self, all_components: Dict[str, gr.components.Component]) -> None:
        all_insert_requests: List[InsertAfterRequest] = []

        for plugin_id, plugin in self.plugins.items():
            try:
                for comp_id in plugin.component_requests:
                    if comp_id in all_components:
                        setattr(plugin, comp_id, all_components[comp_id])

                requested_components = {
                    comp_id: all_components[comp_id]
                    for comp_id in plugin.component_requests
                    if comp_id in all_components
                }
                plugin.post_ui_setup(requested_components)
                all_insert_requests.extend(getattr(plugin, '_insert_after_requests', []))
                
            except Exception as e:
                print(f"[PluginManager] Error in post_ui_setup for {plugin_id}: {str(e)}")
                traceback.print_exc()

        if all_insert_requests:
            for request in all_insert_requests:
                try:
                    target = all_components.get(request.target_component_id)
                    parent = getattr(target, 'parent', None)
                    if not target or not parent or not hasattr(parent, 'children'):
                        print(f"[PluginManager] ERROR: Target '{request.target_component_id}' for insertion not found or invalid.")
                        continue
                        
                    target_index = parent.children.index(target)
                    with parent:
                        new_component = request.new_component_constructor()
                    
                    newly_added = parent.children.pop(-1)
                    parent.children.insert(target_index + 1, newly_added)
                    print(f"[PluginManager] Successfully inserted component after '{request.target_component_id}'")

                except Exception as e:
                    print(f"[PluginManager] Error processing insert_after for {request.target_component_id}: {str(e)}")

class WAN2GPApplication:
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.ui_components = {}

    def _create_plugin_tabs(self, main_module_globals: Dict[str, Any]):
        if not self.plugin_manager:
            return
        server_config = main_module_globals.get('server_config', {})
        enabled_plugins = server_config.get("enabled_plugins", [])
        self.plugin_manager.load_plugins_from_directory(enabled_plugins)
        self.plugin_manager.inject_globals(main_module_globals)
        plugin_ui = self.plugin_manager.setup_ui()
        plugin_tabs = plugin_ui.get('tabs', {})
        
        sorted_plugin_tabs = sorted(
            plugin_tabs.items(),
            key=lambda item: item[1].get('position', -1)
        )

        for tab_id, tab_info in sorted_plugin_tabs:
            with gr.Tab(tab_info['label'], id=f"plugin_{tab_id}"):
                tab_info['component_constructor']()

    def finalize_ui_setup(self, main_module_globals: Dict[str, Any], all_ui_components: Dict[str, Any]):
        self._create_plugin_tabs(main_module_globals)
        self.ui_components = { 
            name: obj for name, obj in all_ui_components.items() 
            if isinstance(obj, (gr.Blocks, gr.components.Component, gr.Row, gr.Column, gr.Tabs, gr.Group, gr.Accordion)) 
        }
        if self.plugin_manager:
            self.plugin_manager.run_post_ui_setup(self.ui_components)