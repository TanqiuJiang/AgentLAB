import json
import logging
import re
from typing import List
from copy import deepcopy

import numpy as np
import json_repair
import ray

from src.Environments import BaseEnvironment
from src.LanguageModels import BedrockLM, VllmLM, OpenAILM
from src.utils import str2json, gen_tool_call_id, ensure_ray_initialized


class Agent():
    """
    Main agent class that manages language model interactions and handles multiple environments.
    
    This class serves as a wrapper around different language model implementations (Bedrock, OpenAI, vLLM)
    and manages communication with various environments through tool calls. It supports multi-agent
    scenarios and handles message formatting for different model types.
    
    Attributes:
        temperature (float): Sampling temperature for model generation
        top_p (float): Nucleus sampling parameter
        max_tokens (int): Maximum tokens to generate
        sys_prompt_path (str): Path to system prompt file
        n_agents (int): Number of agents to manage
        model_id (str): Identifier for the language model
        sys_prompt (str): System prompt content
        model: Language model instance (BedrockLM, OpenAILM, or VllmLM)
        envs (List[BaseEnvironment]): List of environments for each agent
        tool_configs (List): Tool configurations for each environment
        running (np.ndarray): Boolean array tracking which agents are active
        messages (List): Message history for each agent
    """
    
    def __init__(self, 
                 model_id: str, 
                 model: VllmLM = None,
                 envs: List[BaseEnvironment] = [], 
                 temperature: float = 0.0, 
                 top_p: float = 0.9, 
                 max_tokens: int = None, 
                 sys_prompt_path: str = None, 
                 n_gpus: int = 8,
                 region: str = 'us-east-1',
                 n_agents: int = 1) -> None:
        """
        Initialize the Agent with specified parameters and language model.
        
        Args:
            model_id (str): Identifier for the language model to use
            model (VllmLM, optional): Pre-initialized model instance
            envs (List[BaseEnvironment]): List of environments for the agent
            temperature (float): Sampling temperature for generation (0.0-1.0)
            top_p (float): Nucleus sampling parameter (0.0-1.0)
            max_tokens (int, optional): Maximum tokens to generate
            sys_prompt_path (str, optional): Path to system prompt file
            n_gpus (int): Number of GPUs to use for vLLM models
            region (str): AWS region for Bedrock models
            n_agents (int): Number of agents to manage
        """
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sys_prompt_path = sys_prompt_path
        self.n_agents = n_agents
        self.model_id = model_id
        
        self.sys_prompt_path = sys_prompt_path
        self.load_sys_prompt()

        if model:
            self.model = model
        else:
            if 'claude' in model_id.lower() or 'llama' in model_id.lower() or 'deepseek' in model_id.lower():
                self.model = BedrockLM(model_id, sys_prompt_paths=[sys_prompt_path], region=region)
            elif 'gpt' in model_id.lower():
                self.model = OpenAILM(model_id, sys_prompt_paths=[sys_prompt_path])
            else:
                ensure_ray_initialized()
                self.model = VllmLM.remote(model_id, n_gpus=n_gpus, sys_prompt_paths=[sys_prompt_path])
        
        self.reset(envs)
        
    def load_sys_prompt(self):
        """
        Load system prompt from file specified in sys_prompt_path.
        
        Sets self.sys_prompt to the content of the file, or empty string if no path provided.
        """
        if not self.sys_prompt_path:
            self.sys_prompt = ""
        else:
            with open(self.sys_prompt_path, 'r', encoding='utf-8') as f:
                self.sys_prompt = f.read()
    
    def extend_sys_prompt(self, prompt_to_append) -> None:
        """
        Extend the current system prompt with additional content.
        
        Args:
            prompt_to_append (str or list): Content to append to system prompt.
                If string, appends directly. If list, creates list of extended prompts.
        """
        if isinstance(prompt_to_append, str):
            self.sys_prompt = self.sys_prompt + prompt_to_append if self.sys_prompt else prompt_to_append
        elif isinstance(prompt_to_append, list):
            self.sys_prompt = [self.sys_prompt + p for p in prompt_to_append] # list
        
    def reset_running(self, running=None):
        """
        Reset the running status for all agents.
        
        Args:
            running (np.ndarray, optional): Boolean array specifying which agents should be running.
                If None, sets all agents to running (True).
        """
        self.running = np.full(self.n_agents, True) # running tools
        if running is not None:
            self.running[~running] = False
        
    def format_tool_config(self, tool_config):
        """
        Format tool configuration for different model types.
        
        For non-GPT/Claude/LLaMA models, wraps the tool configuration in XML tags
        with instructions for proper JSON tool call formatting.
        
        Args:
            tool_config (str or dict): Tool configuration to format
            
        Returns:
            str: Formatted tool configuration with instructions
        """
        if 'gpt' not in self.model_id.lower() and 'o3' not in self.model_id.lower() and 'o4' not in self.model_id.lower() and 'claude' not in self.model_id.lower() and 'llama' not in self.model_id.lower():
            prompt_parts = [
                "You have access to the following tools to assist the user.",
                "You must respond ONLY with a valid JSON object of the tool call if you decide to use a tool.",
                "Make sure to only call one tool at a time.",
                "\n\n# Tools\n",
                "You are provided with function signatures within <tools></tools> XML tags.",
                "<tools>"
            ]
            prompt_parts.append(tool_config if isinstance(tool_config, str) else json.dumps(tool_config))
            prompt_parts.append("</tools>")
            prompt_parts.append(
                "\nFor each function call, return a json object with function name and "
                "arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": "<function-name>", "arguments": {"<arg_name>": "<arg_value>"}}\n'
                "</tool_call>\n\n"
                "Make sure to only call the next tool in your response (ONLY ONE TOOL) and wait for the tool output to be returned to you before calling the next tool, even if you want to tall multiple tools.\n"
                "If you need to think or reason, keep your thinking or reasoning concise and do not return too much text.\n"
            )
            return "".join(prompt_parts)
        else:
            return tool_config
                    
            
    def reset(self, envs: List[BaseEnvironment]):
        """
        Reset the agent with new environments and initialize message histories.
        
        Args:
            envs (List[BaseEnvironment]): List of environments to reset with
        """
        self.n_agents = len(envs)
        self.envs = envs
        self.tool_configs = [env.tool_config for env in self.envs]
        self.reset_running()
        if isinstance(self.model, ray.actor.ActorHandle):
            self.messages = [[{'role': 'system', 'content': self.sys_prompt + self.format_tool_config(tool_config)}] for tool_config in self.tool_configs]
        elif 'gpt' in self.model_id.lower() or 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
            self.messages = [[{'role': 'system', 'content': self.sys_prompt}] for _ in self.tool_configs]
        else:
            self.messages = [[]] * self.n_agents
        
    def get_env_info(self):
        """
        Get environment information from all environments.
        
        Returns:
            List[str]: Environment information for each environment
        """
        return [env.get_env_info() for env in self.envs]
        
    def get_tool_info(self):
        """
        Get tool information from all environments.
        
        Returns:
            List[str]: Tool information for each environment
        """
        return [env.get_tool_info() for env in self.envs]

    def extend_messages(self, new_messages, idx=None):
        """
        Extend message histories with new messages.
        
        Args:
            new_messages (List[List[dict]]): New messages to append
            idx (int, optional): Specific agent index to extend. If None, extends all agents.
        """
        if idx is not None:
            self.messages[idx].extend(new_messages[0])
        else:
            for i in range(self.n_agents):
                self.messages[i].extend(new_messages[i])

    def step(self, user_prompts, spotlighting=False):
        """
        Execute one step of agent interaction with the language model.
        
        This method handles the complete interaction cycle:
        1. Formats and sends user prompts to the language model
        2. Processes model responses and extracts tool calls
        3. Executes tool calls in the appropriate environments
        4. Updates message histories with results
        
        The method handles different model types (Bedrock, OpenAI, vLLM) with their
        specific message formats and tool call structures.
        
        Args:
            user_prompts (List[str]): User prompts for each agent
            spotlighting (bool): Whether to apply spotlighting to OpenAI prompts
        """
        if isinstance(self.model, BedrockLM):
            self.model.set_sys_prompt(self.sys_prompt)
            if 'deepseek' in self.model_id.lower():
                # assumes batch_size = 1
                self.model.extend_sys_prompt(json.dumps(self.format_tool_config(self.tool_configs[0])))
            for agent_i in range(self.n_agents):
                formatted_user_prompts = self.model.format_prompts(user_prompts)
                if user_prompts[agent_i]:
                    self.messages[agent_i].extend(formatted_user_prompts[agent_i])
            was_running = np.where(self.running)[0]
            all_completion_raw = self.model.generate([m for i, m in enumerate(self.messages) if self.running[i]],  
                                                    temperature=self.temperature, 
                                                    top_p=self.top_p, 
                                                    max_tokens=self.max_tokens, 
                                                    tool_configs=[t for i, t in enumerate(self.tool_configs) if self.running[i]],
                                                    return_raw_output=True,
                                                    role='Agent')
            
            for agent_i, completion_raw in enumerate(all_completion_raw):
                if not completion_raw:
                    continue
                agent_idx = was_running[agent_i]
                completion_raw = completion_raw['content']
                if completion_raw == []:
                    completion_raw = [{"text": "[empty]"}]
                self.messages[agent_idx].append({'role': 'assistant', 'content': deepcopy(completion_raw)})
                self.running[agent_idx] = False
                all_env_messages = []
                for this_completion_raw in completion_raw:
                    if 'deepseek' in self.model_id.lower():
                        if 'reasoningContent' in this_completion_raw:
                            continue
                        try:
                            this_completion = this_completion_raw['text']
                            this_completion = this_completion.split('</tool_call>')[0]
                            completion_json = str2json(this_completion)
                            params = completion_json['parameters'] if 'parameters' in this_completion else completion_json['arguments']
                            tool_name = re.sub(r'[^a-zA-Z0-9_-]', '', completion_json['name'])
                            completion = {'type': 'tool', 
                                        'tool_call_id': completion_json['tool_call_id'] if 'tool_call_id' in completion_json else "NA", 
                                        'tool_name': tool_name,
                                        'arguments': params}
                            self.running[agent_idx] = completion_json['name'] != "end_task"
                            env_messages = self.envs[agent_idx].step(completion)
                            all_env_messages.extend(env_messages)
                        except:
                            self.messages[agent_idx].append({'role': 'tool', 'content': f'Function call failed. Try again and make sure to use the correct JSON format and not to include any other text in your response.'})

                    elif 'toolUse' in this_completion_raw:
                        self.running[agent_idx] = this_completion_raw['toolUse']['name'] != "end_task"
                        tool_name = re.sub(r'[^a-zA-Z0-9_-]', '', this_completion_raw['toolUse']['name'])
                        completion = {'type': 'tool', 
                                    'tool_call_id': this_completion_raw['toolUse']['toolUseId'], 
                                    'tool_name': tool_name,
                                    'arguments': this_completion_raw['toolUse']['input']}
                        
                        env_messages = self.envs[agent_idx].step(completion)
                        all_env_messages.extend(deepcopy(env_messages))
                if len(all_env_messages) > 0:
                    env_messages = all_env_messages[0]
                    if len(all_env_messages) > 1:
                        for i in range(len(all_env_messages)-1):
                            env_messages['content'].extend(all_env_messages[i+1]['content'])
                    self.messages[agent_idx].append(env_messages)
                    
        elif isinstance(self.model, OpenAILM):
            self.model.set_sys_prompt(self.sys_prompt)
            prev_running = np.where(self.running)[0]
            for agent_i in range(self.n_agents):
                add_sys_prompt = self.messages[agent_i] == []
                formatted_user_prompts = self.model.format_prompts(user_prompts, add_sys_prompt=add_sys_prompt)
                if user_prompts[agent_i]:
                    self.messages[agent_i].extend(formatted_user_prompts[agent_i])
            all_completion_raw = self.model.generate([m for i, m in enumerate(self.messages) if self.running[i]],  
                                                    temperature=self.temperature, 
                                                    top_p=self.top_p, 
                                                    max_tokens=self.max_tokens, 
                                                    tool_configs=[t for i, t in enumerate(self.tool_configs) if self.running[i]],
                                                    return_raw_output=True,
                                                    spotlighting=spotlighting,
                                                    role='Agent')
            for i, completion_raw in enumerate(all_completion_raw):
                agent_idx = prev_running[i]
                self.messages[agent_idx].append(completion_raw)
                self.running[agent_idx] = False
                all_env_messages = []
                if 'tool_calls' in completion_raw and len(completion_raw['tool_calls']) > 0:
                    if len(completion_raw['tool_calls']) > 1:
                        completion_raw['tool_calls'] = [completion_raw['tool_calls'][0]]
                        self.messages[agent_idx][-1] = completion_raw
                    tool_call = completion_raw['tool_calls'][0]
                    self.running[agent_idx] = True
                    completion = {'type': 'tool', 
                                'tool_call_id': tool_call['id'], 
                                'tool_name': tool_call['function']['name'],
                                'arguments': json_repair.loads(tool_call['function']['arguments'])}
                    env_messages = self.envs[agent_idx].step(completion)
                    all_env_messages.extend(deepcopy(env_messages))
                if len(all_env_messages) > 0:
                    env_messages = all_env_messages[0]
                    if len(all_env_messages) > 1:
                        for i in range(len(all_env_messages)-1):
                            env_messages['content'].extend(all_env_messages[i+1]['content'])
                    self.messages[agent_idx].append(env_messages)
            
        elif isinstance(self.model, ray.actor.ActorHandle):
            self.model.set_sys_prompt.remote(self.sys_prompt)
            logging.info(f"\n[SYSTEM PROMPT]: {self.model.get_sys_prompt.remote()}")
            logging.info(f"\n[USER MESSAGES]: {self.messages}")
            for agent_i in range(len(user_prompts)):
                if user_prompts[agent_i]:
                    self.messages[agent_i].append({'role': 'user', 'content': user_prompts[agent_i]})
            prev_running_agents = np.where(self.running[:len(user_prompts)])[0]
            all_completions = self.model.generate.remote([m for i, m in enumerate(self.messages[:len(user_prompts)]) if self.running[i]],  
                                                        temperature=self.temperature, 
                                                        top_p=self.top_p, 
                                                        max_tokens=self.max_tokens, 
                                                        tool_configs=[t for i, t in enumerate(self.tool_configs[:len(user_prompts)]) if self.running[i]],
                                                        return_raw_output=False,
                                                        role='Agent')
            all_completions = ray.get(all_completions)
            logging.info(f"\n[AGENT RESPONSE] {all_completions}")
            for i, completion in enumerate(all_completions):
                agent_i = prev_running_agents[i]
                self.messages[agent_i].append({'role': 'assistant', 'content': completion})
                self.running[agent_i] = False
                if '{' in completion and "name" in completion and ('parameters' in completion or 'arguments' in completion) and '}' in completion and "tool_call_result" not in completion: # function call attempt
                    self.running[agent_i] = True
                    try:
                        completion_json = str2json(completion)
                        completion_json['tool_call_id'] = gen_tool_call_id()
                        params = completion_json['parameters'] if 'parameters' in completion else completion_json['arguments']
                        tool_completion = {'type': 'tool', 
                                            'tool_call_id': completion_json['tool_call_id'], 
                                            'tool_name': completion_json['name'],
                                            'arguments': params}
                        self.messages[agent_i][-1] = {"role": "assistant", "content": "<tool_call>"+json.dumps(completion_json)+"</tool_call>"}
                        env_messages = self.envs[agent_i].step(tool_completion)
                        self.messages[agent_i].extend(env_messages)
                    except:
                        self.messages[agent_i].append({'role': 'tool', 'content': f'Function call failed. Try again and make sure to use the correct JSON format and not to include any other text in your response.'})
