import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import gradio as gr
import traceback
import subprocess
import git
import shutil

SYSTEM_PLUGINS = [
    "wan2gp-about",
    "wan2gp-configuration",
    "wan2gp-downloads",
    "wan2gp-guides",
    "wan2gp-plugin-manager",
    "wan2gp-video-mask-creator",
]

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
        self.description = "No description provided."
        self._component_requests: List[str] = []
        self._global_requests: List[str] = []
        self._insert_after_requests: List[InsertAfterRequest] = []
        self._setup_complete = False
        self._data_hooks: Dict[str, List[callable]] = {}
        self.tab_ids: List[str] = []
        self.target_tabs: Optional[List[str]] = None
        
    def setup_ui(self) -> None:
        pass
        
    def add_tab(self, tab_id: str, label: str, component_constructor: callable, position: int = -1):
        self.tabs[tab_id] = PluginTab(id=tab_id, label=label, component_constructor=component_constructor, position=position)

    def post_ui_setup(self, components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        return {}

    def on_tab_select(self, state: Dict[str, Any]) -> None:
        pass

    def on_tab_deselect(self, state: Dict[str, Any]) -> None:
        pass

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
        
    def register_data_hook(self, hook_name: str, callback: callable):
        if hook_name not in self._data_hooks:
            self._data_hooks[hook_name] = []
        self._data_hooks[hook_name].append(callback)

    def insert_after(self, target_component_id: str, new_component_constructor: callable) -> None:
        if not hasattr(self, '_insert_after_requests'):
            self._insert_after_requests = []
        self._insert_after_requests.append(
            InsertAfterRequest(
                target_component_id=target_component_id,
                new_component_constructor=new_component_constructor
            )
        )

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self.plugins_dir = plugins_dir
        os.makedirs(self.plugins_dir, exist_ok=True)
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, self.plugins_dir)
        self.data_hooks: Dict[str, List[callable]] = {}

    def get_plugins_info(self) -> List[Dict[str, str]]:
        plugins_info = []
        for dir_name in self.discover_plugins():
            plugin_path = os.path.join(self.plugins_dir, dir_name)
            is_system = dir_name in SYSTEM_PLUGINS
            info = {'id': dir_name, 'name': dir_name, 'version': 'N/A', 'description': 'No description provided.', 'path': plugin_path, 'system': is_system}
            try:
                module = importlib.import_module(f"{dir_name}.plugin")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                        instance = obj()
                        info['name'] = instance.name
                        info['version'] = instance.version
                        info['description'] = instance.description
                        break
            except Exception as e:
                print(f"Could not load metadata for plugin {dir_name}: {e}")
            plugins_info.append(info)
        
        plugins_info.sort(key=lambda p: (not p['system'], p['name']))
        return plugins_info

    def uninstall_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for uninstallation."
        
        if plugin_id in SYSTEM_PLUGINS:
            return f"[Error] Cannot uninstall system plugin '{plugin_id}'."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' directory not found."

        try:
            shutil.rmtree(target_dir)
            return f"[Success] Plugin '{plugin_id}' uninstalled. Please restart WanGP."
        except Exception as e:
            return f"[Error] Failed to remove plugin '{plugin_id}': {e}"

    def update_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for update."
            
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(os.path.join(target_dir, '.git')):
            return f"[Error] '{plugin_id}' is not a git repository and cannot be updated automatically."

        try:
            gr.Info(f"Attempting to update '{plugin_id}'...")
            repo = git.Repo(target_dir)
            origin = repo.remotes.origin
            origin.fetch()
            
            local_commit = repo.head.commit
            remote_commit = origin.refs[repo.active_branch.name].commit

            if local_commit == remote_commit:
                 return f"[Info] Plugin '{plugin_id}' is already up to date."

            origin.pull()
            
            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                gr.Info(f"Re-installing dependencies for '{plugin_id}'...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

            return f"[Success] Plugin '{plugin_id}' updated. Please restart WanGP for changes to take effect."
        except git.exc.GitCommandError as e:
            return f"[Error] Git update failed for '{plugin_id}': {e.stderr}"
        except Exception as e:
            return f"[Error] An unexpected error occurred during update of '{plugin_id}': {str(e)}"

    def reinstall_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for reinstallation."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' not found."

        git_url = None
        if os.path.isdir(os.path.join(target_dir, '.git')):
            try:
                repo = git.Repo(target_dir)
                git_url = repo.remotes.origin.url
            except Exception as e:
                return f"[Error] Could not get remote URL for '{plugin_id}': {e}"
        
        if not git_url:
            return f"[Error] Could not determine remote URL for '{plugin_id}'. Cannot reinstall."

        gr.Info(f"Reinstalling '{plugin_id}'...")
        uninstall_msg = self.uninstall_plugin(plugin_id)
        if "[Error]" in uninstall_msg:
            return uninstall_msg
        
        install_msg = self.install_plugin_from_url(git_url)
        if "[Success]" in install_msg:
            return f"[Success] Plugin '{plugin_id}' reinstalled. Please restart WanGP."
        else:
            return f"[Error] Reinstallation failed during install step: {install_msg}"

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

    def load_plugins_from_directory(self, enabled_user_plugins: List[str]) -> None:
        plugins_to_load = SYSTEM_PLUGINS + [p for p in enabled_user_plugins if p not in SYSTEM_PLUGINS]

        for plugin_dir_name in self.discover_plugins():
            if plugin_dir_name not in plugins_to_load:
                continue
            try:
                # Reload the module to pick up code changes without restarting the whole app
                module_name = f"{plugin_dir_name}.plugin"
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                        plugin = obj()
                        plugin.setup_ui()
                        self.plugins[plugin_dir_name] = plugin
                        for hook_name, callbacks in plugin._data_hooks.items():
                            if hook_name not in self.data_hooks:
                                self.data_hooks[hook_name] = []
                            self.data_hooks[hook_name].extend(callbacks)
                        if plugin_dir_name not in SYSTEM_PLUGINS:
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
        
    def run_data_hooks(self, hook_name: str, *args, **kwargs):
        if hook_name not in self.data_hooks:
            return kwargs.get('configs')

        callbacks = self.data_hooks[hook_name]
        data = kwargs.get('configs')

        if 'configs' in kwargs:
            kwargs.pop('configs')

        for callback in callbacks:
            try:
                data = callback(data, **kwargs)
            except Exception as e:
                print(f"[PluginManager] Error running hook '{hook_name}' from {callback.__module__}: {e}")
                traceback.print_exc()
        return data

    def run_post_ui_setup(self, all_tab_components: List[Dict[str, Any]]) -> None:
        for tab_components in all_tab_components:
            tab_id = tab_components.get('__tab_id__')
            if not tab_id:
                print("[PluginManager] WARNING: A tab component dictionary is missing '__tab_id__'. Skipping.")
                continue

            tab_insert_requests: List[InsertAfterRequest] = []

            for plugin_id, plugin in self.plugins.items():
                target_tabs = getattr(plugin, 'target_tabs', None)
                if target_tabs is not None and tab_id not in target_tabs:
                    continue

                if not all(comp_id in tab_components for comp_id in plugin.component_requests):
                    continue
                
                original_attrs = {}
                try:
                    # Temporarily set attributes for the current tab's context
                    for comp_id in plugin.component_requests:
                        if hasattr(plugin, comp_id):
                            original_attrs[comp_id] = getattr(plugin, comp_id)
                        setattr(plugin, comp_id, tab_components[comp_id])

                    requested_components = { comp_id: tab_components[comp_id] for comp_id in plugin.component_requests }
                    requested_components['__tab_id__'] = tab_id
                    
                    plugin._insert_after_requests = []
                    plugin.post_ui_setup(requested_components)
                    tab_insert_requests.extend(getattr(plugin, '_insert_after_requests', []))
                    
                except Exception as e:
                    print(f"[PluginManager] Error in post_ui_setup for {plugin_id} on tab '{tab_id}': {str(e)}")
                    traceback.print_exc()
                finally:
                    # Clean up attributes to prevent state leakage
                    for comp_id in plugin.component_requests:
                        if hasattr(plugin, comp_id):
                            delattr(plugin, comp_id)
                    # Restore any original attributes that were overwritten
                    for comp_id, value in original_attrs.items():
                        setattr(plugin, comp_id, value)

            if tab_insert_requests:
                for request in tab_insert_requests:
                    try:
                        target = tab_components.get(request.target_component_id)
                        parent = getattr(target, 'parent', None)
                        if not target or not parent or not hasattr(parent, 'children'):
                            print(f"[PluginManager] ERROR on tab '{tab_id}': Target '{request.target_component_id}' for insertion not found or invalid.")
                            continue
                            
                        target_index = parent.children.index(target)
                        with parent:
                            new_component = request.new_component_constructor()
                        
                        newly_added = parent.children.pop(-1)
                        parent.children.insert(target_index + 1, newly_added)
                        print(f"[PluginManager] Successfully inserted component after '{request.target_component_id}' on tab '{tab_id}'")

                    except Exception as e:
                        print(f"[PluginManager] Error processing insert_after for {request.target_component_id} on tab '{tab_id}': {str(e)}")


class WAN2GPApplication:
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.tab_to_plugin_map: Dict[str, WAN2GPPlugin] = {}

    def _create_plugin_tabs(self, main_module_globals: Dict[str, Any]):
        if not self.plugin_manager:
            return
        server_config = main_module_globals.get('server_config', {})
        enabled_user_plugins = server_config.get("enabled_plugins", [])
        
        self.plugin_manager.load_plugins_from_directory(enabled_user_plugins)
        self.plugin_manager.inject_globals(main_module_globals)

        loaded_plugins = self.plugin_manager.get_all_plugins()

        system_tabs = []
        user_tabs = []

        for plugin_id, plugin in loaded_plugins.items():
            for tab_id, tab in plugin.tabs.items():
                self.tab_to_plugin_map[f"plugin_{tab_id}"] = plugin
                tab_info = {
                    'id': tab_id,
                    'label': tab.label,
                    'component_constructor': tab.component_constructor,
                    'position': tab.position
                }
                if plugin_id in SYSTEM_PLUGINS:
                    system_tabs.append(tab_info)
                else:
                    user_tabs.append((plugin_id, tab_info))

        system_tabs.sort(key=lambda t: (t.get('position', -1), t['label']))
        
        sorted_user_tabs = []
        for plugin_id in enabled_user_plugins:
            for pid, tab_info in user_tabs:
                if pid == plugin_id:
                    sorted_user_tabs.append(tab_info)
        
        all_tabs_to_render = system_tabs + sorted_user_tabs
        
        for tab_info in all_tabs_to_render:
            with gr.Tab(tab_info['label'], id=f"plugin_{tab_info['id']}"):
                tab_info['component_constructor']()

    def _handle_tab_selection(self, state: dict, evt: gr.SelectData):
        if not hasattr(self, 'previous_tab_id'):
            self.previous_tab_id = None
        
        new_tab_id = evt.value
        
        if self.previous_tab_id == new_tab_id:
            return

        if self.previous_tab_id and self.previous_tab_id in self.tab_to_plugin_map:
            plugin_to_deselect = self.tab_to_plugin_map[self.previous_tab_id]
            try:
                plugin_to_deselect.on_tab_deselect(state)
            except Exception as e:
                print(f"[PluginManager] Error in on_tab_deselect for plugin {plugin_to_deselect.name}: {e}")
                traceback.print_exc()

        if new_tab_id and new_tab_id in self.tab_to_plugin_map:
            plugin_to_select = self.tab_to_plugin_map[new_tab_id]
            try:
                plugin_to_select.on_tab_select(state)
            except Exception as e:
                print(f"[PluginManager] Error in on_tab_select for plugin {plugin_to_select.name}: {e}")
                traceback.print_exc()

        self.previous_tab_id = new_tab_id

    def finalize_ui_setup(self, main_module_globals: Dict[str, Any], all_tab_components: List[Dict[str, Any]]):
        self._create_plugin_tabs(main_module_globals)
        
        main_tabs = main_module_globals.get('main_tabs')
        state = main_module_globals.get('state')
        if main_tabs and state:
             main_tabs.select(
                self._handle_tab_selection, 
                inputs=[state], 
                outputs=None,
                show_progress="hidden"
            )
            
        if self.plugin_manager:
            self.plugin_manager.run_post_ui_setup(all_tab_components)