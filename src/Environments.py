import sys
import json
import inspect
import logging
from copy import deepcopy
from typing import get_origin, get_args, Union
from abc import ABC, abstractmethod
from ast import literal_eval

from src.utils import get_schema_from_annotation, get_class_tool_infos

sys.path.append('Agent_SafetyBench/environments')
from Agent_SafetyBench.environments.EnvManager import EnvManager

sys.path.append('SHADE_Arena')
from SHADE_Arena.environments.banking_environment import BankingEnvironment
from SHADE_Arena.environments.spam_filter_update_environment import SpamFilterUpdateEnvironment
from SHADE_Arena.environments.travel_environment import TravelEnvironment
from SHADE_Arena.environments.workspace_environment import WorkspaceEnvironment
from SHADE_Arena.utils.functions_runtime import FunctionsRuntime, make_function
from SHADE_Arena.utils.pipeline_elements import is_string_list


class BaseEnvironment(ABC):
    """
    Abstract base class for all environment implementations.
    
    This class defines the interface that all environment implementations must follow.
    Environments handle tool execution and provide context information to agents.
    """
    
    def __init__(self):
        """Initialize the base environment."""
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state.
        
        This method should be implemented by subclasses to reinitialize
        the environment for a new episode or task.
        """
        pass
    
    @abstractmethod
    def get_env_info(self):
        """
        Get information about the current environment state.
        
        Returns:
            str: Description of the environment and its current state
        """
        pass
    
    @abstractmethod
    def get_tool_info(self):
        """
        Get information about available tools in this environment.
        
        Returns:
            str: Description of available tools and their usage
        """
        pass
    
    @abstractmethod
    def step(self):
        """
        Execute a tool call or action in the environment.
        
        Args:
            completion (dict): Tool call information including name and arguments
            
        Returns:
            List[dict]: List of message objects containing tool execution results
        """
        pass
    
    
class AgentSafetyBenchEnvironment(BaseEnvironment):
    """
    Environment implementation for Agent-SafetyBench scenarios.
    
    This environment wraps the Agent-SafetyBench framework, providing access to
    various safety-related environments and their associated tools. It handles
    tool configuration formatting for different model types.
    
    Attributes:
        env_manager (EnvManager): Manager for Agent-SafetyBench environments
        model_id (str): Identifier for the language model being used
        data (dict): Environment configuration data
        envs (List): List of initialized environments
        tool_config (dict or List): Tool configuration formatted for the model type
    """
    
    def __init__(self, data, model_id, allow_all_tools=False):
        """
        Initialize the Agent-SafetyBench environment.
        
        Args:
            data (dict): Configuration data containing environment and tool specifications
            model_id (str): Identifier for the language model to format tools for
        """
        super().__init__()
        self.env_manager = EnvManager()
        self.model_id = model_id
        self.allow_all_tools = allow_all_tools
        self.reset(data)
        
    def reset(self, data):
        """
        Reset the environment with new configuration data.
        
        Initializes environments based on the provided data and configures tools
        according to the model type (Claude/LLaMA, GPT, or other models).
        
        Args:
            data (dict): Configuration containing environment specifications and tool lists
        """
        self.data = data
        envs = []
        tool_descs = []
        for env_info in data['environments']:
            env_name = env_info['name']
            if env_name == '':
                continue
            if 'parameters' in env_info:
                env_params = env_info['parameters'] or None
            else:
                env_params = None
            env = self.env_manager.init_env(env_name, env_params)
            if env is None:
                print(data['environments'])
                print(f"Warning: Environment {env_name} not found.")
                continue
            envs.append(env)
            tool_names = env_info['tools']
            tool_desc = env.get_tool_descs(tool_names)
            tool_descs.extend(tool_desc)
            
        self.envs = envs
            
        for item in tool_descs:
            item['type'] = "object"
            
        for i, item in enumerate(tool_descs):
            newitem = {'type': 'function', 'function': item}
            tool_descs[i] = newitem
            
        tools_list = []

        if self.allow_all_tools:
            for env in self.envs:
                tools_list += get_class_tool_infos(env)
            if 'claude' in self.model_id.lower() or 'llama' in self.model_id.lower():
                self.tool_config = []
                tool_names = []
                for tool in tools_list:
                    if 'toolSpec' in tool:
                        self.tool_config.append(tool)
                    else:
                        if tool['function']['name'] in tool_names:
                            continue
                        else:
                            tool_names.append(tool['function']['name'])
                        self.tool_config.append(
                                {
                                    'toolSpec': 
                                        {
                                            'name': tool['function']['name'],
                                            'description': tool['function']['description'],
                                            'inputSchema': {
                                                'json': tool['function']['parameters']
                                            }
                                        }
                                }
                            )
                self.tool_config = {'tools': self.tool_config}
            else:
                self.tool_config = json.dumps(tools_list)
        else:
            if 'claude' in self.model_id.lower() or 'llama' in self.model_id.lower():
                for tool_desc in tool_descs:
                    tool_info = tool_desc[tool_desc['type']]
                    name = tool_info['name']
                    description = tool_info['description']
                    parameters = tool_info['parameters']
                    tools_list.append(
                        {
                            'toolSpec': 
                                {
                                    'name': name,
                                    'description': description,
                                    'inputSchema': {
                                        'json': parameters
                                    }
                                }
                        }
                    )
                    
                self.tool_config = {'tools': tools_list} if len(tool_descs) > 0 else None
                
            elif 'gpt' in self.model_id.lower():
                for item in tool_descs:
                    item['type'] = "object"
                    
                for i, item in enumerate(tool_descs):
                    newitem = {'type': 'function', 'function': item}
                    tool_descs[i] = newitem
                    
                self.tool_config = tool_descs
                
            else:
                for tool_desc in tool_descs:
                    tool_info = tool_desc[tool_desc['type']]
                    name = tool_info['name']
                    description = tool_info['description']
                    parameters = tool_info['parameters']
                    tool_info = {
                            'name': name,
                            'description': description,
                            'parameters': parameters
                        }
                    tools_list.append(tool_info)
                self.tool_config = json.dumps(tools_list) # str
        
    def get_env_info(self):
        """
        Get environment information as JSON string.
        
        Returns:
            str: JSON representation of environment configuration
        """
        return json.dumps(self.data['environments'])
    
    def get_tool_info(self):
        """
        Get tool information for the current configuration.
        
        Returns:
            str or None: JSON string of available tools, or None if no tools configured
        """
        if self.tool_config:
            if isinstance(self.tool_config, str):
                return self.tool_config
            else:
                return json.dumps(self.tool_config['tools'])
        return None
    
    def get_failure_mode_ids(self):
        return [int(f) for f in self.data['failure_modes'].split(',')]
    
    def step(self, completion):
        """
        Execute a tool call in the appropriate environment.
        
        Searches through all environments to find one that supports the requested tool,
        then executes the tool call and formats the result according to the model's
        expected message format.
        
        Args:
            completion (dict): Tool call information containing:
                - tool_call_id: Unique identifier for the tool call
                - tool_name: Name of the tool to execute
                - arguments: Dictionary of arguments for the tool
                
        Returns:
            List[dict]: Message list containing tool execution results
        """
        tool_call_id = completion['tool_call_id']
        tool_name = completion['tool_name']
        arguments = completion['arguments']
        tool_call_result = None
        for env_idx, env in enumerate(self.envs):
            provided_tools = self.data['environments'][env_idx]['tools']
            if tool_name not in provided_tools: continue
            if not env.has_tool(tool_name): continue
            tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
            if isinstance(tool_call_result, int):
                tool_call_result = f"{tool_call_result}"
            elif isinstance(tool_call_result, tuple):
                tool_call_result = list(tool_call_result)
            break
        
        def remove_tuples_from_dict(tool_call_result):
            if isinstance(tool_call_result, dict):
                for k, v in tool_call_result.items():
                    if isinstance(v, tuple):
                        if len(v) == 1:
                            tool_call_result[k] = v[0]
                        else:
                            tool_call_result[k] = list(v)
                    elif isinstance(v, dict):
                        remove_tuples_from_dict(v)
                    
        remove_tuples_from_dict(tool_call_result)
        content_type = 'text' 
        if not isinstance(tool_call_result, str):
        # if content_type == 'json' and isinstance(tool_call_result, list):
            tool_call_result = str(tool_call_result)

        if tool_call_result:
            tool_result = {
                    "toolResult": {
                        "toolUseId": (tool_call_id),
                        "content": [{content_type: deepcopy(tool_call_result)}]
                    }
                }
        else:
            tool_result = {
                    "toolResult": {
                        "toolUseId": (tool_call_id),
                        "content": [{"text": f"Tool {tool_name} doesn't exist."}]
                    }
                }
        
        if 'claude' in self.model_id.lower() or 'deepseek' in self.model_id.lower() or 'llama' in self.model_id.lower():
            tool_call_result_message = [{
                "role": "tool",
                "content": [tool_result]
            }]
        else:
            tool_call_result_message = [{
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result['toolResult']['content'][0]['text']
            }]
            
        return tool_call_result_message
    

class SHADEArenaEnvironment(BaseEnvironment):
    """
    Environment implementation for SHADE Arena scenarios.
    
    This environment provides access to various SHADE Arena environments including
    travel, banking, spam filter, and workspace scenarios. It uses a FunctionsRuntime
    to execute tools and handles model-specific formatting.
    
    Attributes:
        model_id (str): Identifier for the language model being used
        env_name (str): Name of the SHADE Arena environment
        env: The initialized SHADE Arena environment instance
        runtime (FunctionsRuntime): Runtime for executing environment tools
        tool_config (dict or str): Tool configuration formatted for the model type
    """
    
    def __init__(self, model_id: str, env_name: str):
        """
        Initialize the SHADE Arena environment.
        
        Args:
            model_id (str): Identifier for the language model to format tools for
            env_name (str): Name of the specific SHADE Arena environment to initialize
        """
        super().__init__()
        self.model_id = model_id
        self.env_name = env_name
        self.env = self.init_env(env_name)
        self.runtime = FunctionsRuntime([])
        for tool in self.env.available_tools:
            self.runtime.register_function(make_function(tool))
        self.reset()
        
    def init_env(self, env_name: str):
        """
        Initialize a specific SHADE Arena environment by name.
        
        Creates and configures environment instances, limiting the number of emails
        and files in certain environments to manage complexity.
        
        Args:
            env_name (str): Name of the environment to initialize
            
        Returns:
            Environment instance (TravelEnvironment, BankingEnvironment, 
                                SpamFilterUpdateEnvironment, or WorkspaceEnvironment)
        """
        if 'travel' in env_name.lower():
            env = TravelEnvironment.create()
            env.inbox.initial_emails = env.inbox.initial_emails[:50]
            env.inbox.emails = dict(list(env.inbox.emails.items())[:50])
            return env
        if 'banking' in env_name.lower():
            env = BankingEnvironment.create()
            return env
        if 'spam' in env_name.lower():
            env = SpamFilterUpdateEnvironment.create()
            n_emails_to_keep = 5
            for user in env.email_manager.users:
                env.email_manager.users[user].inbox.initial_emails = env.email_manager.users[user].inbox.initial_emails[-n_emails_to_keep:]
                env.email_manager.users[user].inbox.emails = dict(list(env.email_manager.users[user].inbox.emails.items())[-n_emails_to_keep:])
                env.email_manager.users[user].spam.spam_emails = env.email_manager.users[user].spam.spam_emails[-n_emails_to_keep:]
                env.email_manager.users[user].spam.spam_data['initial_emails'] = env.email_manager.users[user].spam.spam_data['initial_emails'][-n_emails_to_keep:]
            n_files_to_keep = 10
            env.cloud_drive.files = dict(list(env.cloud_drive.files.items())[:n_files_to_keep])
            env.cloud_drive.initial_files = env.cloud_drive.initial_files[:n_files_to_keep]
            return env
        if 'workspace' in env_name.lower():
            env = WorkspaceEnvironment.create()
            return env
    
    def reset(self):
        """
        Reset the environment and configure tools for the current model type.
        
        Analyzes available tools from the environment, extracts their signatures and
        documentation, then formats them according to the requirements of different
        model types (Claude/LLaMA, GPT, or other models).
        """
        tool_list = []
        
        for tool in self.env.available_tools:
            signature = inspect.signature(tool)
            name = tool.__name__
            description = inspect.getdoc(tool) or f"A tool to perform the '{name}' operation."
            
            properties = {}
            required = []

            for param_name, param in signature.parameters.items():
                # Generate the full schema for the parameter's type annotation
                param_schema = get_schema_from_annotation(param.annotation)

                # If a description wasn't provided via Annotated, create a default one
                if "description" not in param_schema:
                    param_schema["description"] = f"The {param_name.replace('_', ' ')}."
                
                properties[param_name] = param_schema

                # A parameter is required if it has no default value AND is not Optional.
                is_optional = get_origin(param.annotation) is Union and type(None) in get_args(param.annotation)
                if param.default is inspect.Parameter.empty and not is_optional:
                    required.append(param_name)

            if 'claude' in self.model_id.lower() or 'llama' in self.model_id.lower() or 'deepseek' in self.model_id.lower():
                tool_list.append({
                    "toolSpec": {
                        "name": name,
                        "description": description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            }
                        }
                    }
                })
            elif 'gpt' in self.model_id.lower():
                tool_info = {'type': 'function',
                            'function': {'name': name,
                                        'description': description,
                                        'parameters': {
                                            "type": "object",
                                            "properties": properties,
                                            "required": required
                                        },
                                        'type': 'object'}}
                tool_list.append(tool_info)
            else:
                tool_info = {
                        'name': name,
                        'description': description,
                        'parameters': properties
                    }
                tool_list.append(tool_info)
                
                
        if 'claude' in self.model_id.lower() or 'llama' in self.model_id.lower() or 'deepseek' in self.model_id.lower():
            self.tool_config = {"tools": tool_list}
        else:
            self.tool_config = json.dumps(tool_list) # str
            
    def get_env_info(self):
        """
        Get environment information as string representation.
        
        Returns:
            str: String representation of the environment state
        """
        return repr(self.env)
            
    def get_tool_info(self):
        """
        Get tool information for the current configuration.
        
        Returns:
            str or None: JSON string of available tools, or None if no tools configured
        """
        if self.tool_config:
            if isinstance(self.tool_config, str):
                return self.tool_config
            else:
                return json.dumps(self.tool_config['tools'])
        return None

    def step(self, completion):
        """
        Execute a tool call using the FunctionsRuntime.
        
        Validates the tool exists, processes arguments (including converting string
        lists to actual lists), executes the tool, and formats the result according
        to the model's expected message format.
        
        Args:
            completion (dict): Tool call information containing:
                - tool_call_id: Unique identifier for the tool call
                - tool_name: Name of the tool to execute
                - arguments: Dictionary of arguments for the tool
                
        Returns:
            List[dict]: Message list containing tool execution results
        """
        tool_call_id = completion['tool_call_id']
        tool_name = completion['tool_name']
        arguments = completion['arguments']
        
        if tool_name not in (tool.name for tool in self.runtime.functions.values()):
            tool_result = {
                    "toolResult": {
                        "toolUseId": (tool_call_id),
                        "content": [{"text": f"Tool {tool_name} doesn't exist."}]
                    }
                }
        else:
            # Converts type of input lists from string to list type
            for arg_k, arg_v in arguments.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    arguments[arg_k] = literal_eval(arg_v)
                    
            tool_call_result, error = self.runtime.run_function(self.env, tool_name, arguments)
            if tool_name == 'end_task':
                tool_call_result = "Task ended."
            
            if error:
                tool_result = {
                    "toolResult": {
                        "toolUseId": (tool_call_id),
                        "content": [{"text": str(error)}]
                    }
                }
                logging.error(f"\nEnvironment: {tool_result}")
            else:
                tool_result = {
                        "toolResult": {
                            "toolUseId": (tool_call_id),
                            "content": [{"text": str(tool_call_result)}]
                        }
                    }

        if 'claude' in self.model_id.lower() or 'deepseek' in self.model_id.lower() or 'llama' in self.model_id.lower():
            tool_call_result_message = [{
                "role": "tool",
                "content": [tool_result]
            }]
        else:
            tool_call_result_message = [{
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result['toolResult']['content'][0]['text']
            }]
        logging.info(f"\nEnvironment: {tool_call_result_message}")
        return tool_call_result_message
