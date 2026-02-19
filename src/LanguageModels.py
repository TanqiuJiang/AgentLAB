from __future__ import annotations
import json
import os
import time
import re
import tempfile
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import json_repair
import openai
import ray
import vllm
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from tqdm import tqdm


class LM:
    """
    Base class for language model implementations.
    
    This class provides the basic interface and functionality for interacting with
    different language model backends. It handles system prompts, model loading,
    and generation functionality.
    
    Attributes:
        model_id (str): Identifier for the language model to use
        sys_prompt (str): System prompt to be used with the model
        model (Any): The loaded language model instance
    """
    
    def __init__(self, model_id: str, sys_prompt_paths: List[str] = None) -> None:
        """
        Initialize the language model.
        
        Args:
            model_id (str): Identifier for the model to use
            sys_prompt_path (Optional[str]): Path to the system prompt file
        """
        self.model_id = model_id
        self.sys_prompt = self.load_sys_prompt(sys_prompt_paths)
        self.model = self.load_model()
        
    def load_sys_prompt(self, sys_prompt_paths: List[str]) -> str:
        """
        Load the system prompt from a file.
        
        Args:
            sys_prompt_path (Optional[str]): Path to the system prompt file
            
        Returns:
            str: Content of the system prompt
            
        Raises:
            FileNotFoundError: If the system prompt file doesn't exist
        """
        if not sys_prompt_paths:
            return ""
        
        sys_prompt = ""
        for sys_prompt_path in sys_prompt_paths:
            if sys_prompt_path:
                with open(sys_prompt_path, 'r', encoding='utf-8') as f:
                    sys_prompt += '\n' + f.read()
        
        self.sys_prompt = sys_prompt
        return sys_prompt
        
    def get_sys_prompt(self):
        return self.sys_prompt
    
    def set_sys_prompt(self, sys_prompt):
        self.sys_prompt = sys_prompt

    def extend_sys_prompt(self, prompt_to_append) -> None:
        if isinstance(prompt_to_append, str):
            self.sys_prompt = self.sys_prompt + prompt_to_append if self.sys_prompt else prompt_to_append
        elif isinstance(prompt_to_append, list):
            self.sys_prompt = [self.sys_prompt + p for p in prompt_to_append] # list
        
    def load_model(self) -> Any:
        """
        Load the language model.
        
        This is an abstract method that should be implemented by subclasses.
        
        Returns:
            Any: The loaded model instance
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def format_output(self, id: int, output: str) -> Dict[str, Union[int, str]]:
        """
        Format the model output into a standard dictionary.
        
        Args:
            id (int): The identifier for the output
            output (str): The model's generated text
            
        Returns:
            Dict[str, Union[int, str]]: Formatted output with id and text
        """
        return {'id': id, 'output': output}
    
    def write_outputs(self, outputs, output_path):
        if output_path:
            with open(output_path, 'a') as f:
                for output in outputs:
                    f.write(json.dumps(output) + '\n')
    
    def batched_generate(self, 
                         user_prompts: List[str], 
                         output_path: str = None, 
                         batch_size: int = 128, 
                         temperature: float = 1.0, 
                         top_p: float = 0.9, 
                         max_tokens: Optional[int] = None) -> Any:
        """
        Generate responses for a large number of prompts in batches and save to a file.
        
        Args:
            user_prompts (List[str]): List of user prompts to process
            output_path (str): Path to save the results
            batch_size (int): Number of prompts to process in each batch
            temperature (float): Sampling temperature (higher = more random)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            max_tokens (Optional[int]): Maximum number of tokens to generate
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement batched_generate()")
            
    def generate(self, 
                user_prompts: List[str], 
                temperature: float = 1.0, 
                top_p: float = 0.9, 
                max_tokens: Optional[int] = None) -> List[str]:
        """
        Generate responses for a batch of user prompts.
        
        This is an abstract method that should be implemented by subclasses.
        
        Args:
            user_prompts (List[str]): List of user prompts to process
            temperature (float): Sampling temperature (higher = more random)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            max_tokens (Optional[int]): Maximum number of tokens to generate
            
        Returns:
            List[str]: Generated responses for each prompt
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement generate()")


class BedrockLM(LM):
    """
    AWS Bedrock language model implementation.
    
    This class interfaces with AWS Bedrock to provide language model functionality.
    
    Attributes:
        model_id (str): Identifier for the Bedrock model to use
        region (str): AWS region where the model is deployed
        sys_prompt (List[Dict[str, str]]): System prompt formatted for Bedrock
        model (boto3.client): Bedrock client for API calls
    """
    
    def __init__(self, model_id: str, region: str = 'us-east-1', sys_prompt_paths: List[str] = None) -> None:
        """
        Initialize the AWS Bedrock language model.
        
        Args:
            model_id (str): Identifier for the model to use
            region (str): AWS region where the model is deployed
            sys_prompt_path (Optional[str]): Path to the system prompt file
        """
        self.region = region
        super().__init__(model_id, sys_prompt_paths)
        
    def load_sys_prompt(self, sys_prompt_paths: List[str]) -> List[Dict[str, str]]:
        """
        Load and format the system prompt for Bedrock models.
        
        Args:
            sys_prompt_path (Optional[str]): Path to the system prompt file
            
        Returns:
            List[Dict[str, str]]: System prompt formatted for Bedrock
        """
        raw_prompt = super().load_sys_prompt(sys_prompt_paths)
        return [{"text": raw_prompt}] if len(raw_prompt) > 0 else ""
    
    def extend_sys_prompt(self, prompt_to_append):
        if self.sys_prompt is None:
            self.sys_prompt = [{"text": prompt_to_append}]
        else:
            self.sys_prompt[0]["text"] += prompt_to_append
            
    def set_sys_prompt(self, sys_prompt):
        self.sys_prompt = [{"text": sys_prompt}] 
    
    def format_prompts(self, user_prompts: List[str]) -> Any:
        """
        Format user prompts for Bedrock API.
        
        Args:
            user_prompts (List[str]): Raw user prompts
            
        Returns:
            List[List[Dict[str, List[Dict[str, str]]]]]: Formatted prompts for Bedrock
        """
        formatted_prompts = []
        for user_prompt in user_prompts:
            if not user_prompt:
                new_formatted_prompt = None
            elif isinstance(user_prompt, str):
                new_formatted_prompt = [{"role": "user", "content": [{"text": user_prompt}]}]
            elif isinstance(user_prompt, list):
                formatted_prompt = deepcopy(user_prompt)
                new_formatted_prompt = []
                for p in formatted_prompt:
                    if p['role'] == 'assistant':
                        if 'content' in p:
                            p['content'] = [c for c in p['content'] if 'reasoningContent' not in c]
                            if 'llama' in self.model_id.lower() and len([1 for c in p['content'] if 'toolUse' in c]) > 0 and len([1 for c in p['content'] if 'text' in c]) > 0:
                                p['content'] = [c for c in p['content'] if 'toolUse' in c]
                            new_formatted_prompt.append(p)
                        elif 'tool_calls' in p:
                            tool_call = p['tool_calls'][0]
                            new_p = {
                                "role": "assistant",
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId" : tool_call['id'],
                                            "name": tool_call['function']['name'],
                                            "input": json.loads(tool_call['function']['arguments'])
                                        }
                                    }
                                ]
                            }
                            new_formatted_prompt.append(new_p)
                    elif p['role'] == 'tool':
                        try:
                            tool_result_content = json.loads[p['content']]
                            content_type = 'json'
                        except:
                            tool_result_content = p['content']
                            content_type = 'text'
                        p['role'] = 'user'
                        if 'tool_call_id' in p:
                            new_p = {"role": "user",
                                     "content": [{
                                         "toolResult" : {
                                             "toolUseId": str(p['tool_call_id']),
                                             "content": [{content_type: tool_result_content}]
                                         }
                                     }]}
                        else:
                            new_p = p
                        if 'deepseek' in self.model_id.lower():
                            new_p['content'] = [{"text": json.dumps(new_p['content'])}]
                        if 'llama' in self.model_id.lower() and len([1 for c in new_p['content'] if 'toolResult' in c]) > 0 and len([1 for c in new_p['content'] if 'text' in c]) > 0:
                            new_p['content'] = [c for c in new_p['content'] if 'toolResult' in c]
                        new_formatted_prompt.append(new_p)
                        if 'llama' in self.model_id.lower() and 'toolResult' in p['content'][0] and p['content'][0]['toolResult']['content'][0]['text'] == 'end_task':
                            new_formatted_prompt.append({"role": "assistant",
                                                         "content": [{"text": "I have completed the task."}]})
                    elif isinstance(p['content'], str):
                        new_formatted_prompt.append({"role": p["role"],
                                                     "content": [{"text": p['content']}]})
                    else:
                        new_formatted_prompt.append(p)
                        
            if 'llama' in self.model_id.lower() and new_formatted_prompt:
                no_consecutive_user_prompts = False
                while not no_consecutive_user_prompts:
                    no_consecutive_user_prompts = True
                    for i in range(len(new_formatted_prompt)-1):
                        if new_formatted_prompt[i]['role'] == 'user' and new_formatted_prompt[i+1]['role'] == 'user':
                            # new_formatted_prompt[i]['content'] += new_formatted_prompt[i+1]['content']
                            # new_formatted_prompt = new_formatted_prompt[:i+1] + new_formatted_prompt[i+1:]
                            new_formatted_prompt = new_formatted_prompt[:i+1] + [{"role": "assistant", "content": [{"text": "No action."}]}] + new_formatted_prompt[i+1:]
                            no_consecutive_user_prompts = False
                            break
            formatted_prompts.append(new_formatted_prompt)        
        return formatted_prompts
        
    def convert_messages_format(self, all_messages):
        all_converted_messages = []
        for messages in all_messages:
            converted_messages = []
            for message in messages: 
                if message['role'] == 'user' or message['role'] == 'environment' or message['role'] == 'tool':
                    converted_message = {
                        "role": "user", 
                        "content": [{"text": message["content"]}]
                        }
                elif message['role'] == 'assistant':
                    content = []
                    if "content" in message:
                        content.append({"text": message["content"]})
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            content.append({
                                "toolUse": {
                                    "toolUseId": tool_call["id"],
                                    "name": tool_call["function"]["name"],
                                    "input": json_repair.loads(tool_call["function"]["arguments"])
                                }
                            })
                    converted_message = {
                        "role": "assistant",
                        "content": content
                    }
                elif message["role"] == "tool":
                    converted_message = {
                        "role": "user",
                        "content": [
                            {
                            "toolResult": 
                                {
                                    "toolUseId": (message["tool_call_id"]),
                                    "content": [{"text": message['content']}]
                                }
                            }
                        ]
                    }
                    
                converted_messages.append(converted_message)
            all_converted_messages.append(converted_messages)
        return all_converted_messages
    
    def load_model(self) -> Any:
        """
        Initialize the Bedrock client.
        
        Returns:
            boto3.client: Configured Bedrock runtime client
        """
        model = boto3.client("bedrock-runtime", 
                             region_name=self.region,
                             config=Config(
                                 read_timeout=300,  # Increase timeout to 300 seconds (5 minutes)
                                 connect_timeout=60,
                                 retries={'max_attempts': 5}
))
        return model

    def generate(self, 
                user_prompts: List[str], 
                temperature: float = 1.0, 
                top_p: float = 0.9, 
                max_tokens: Optional[int] = None,
                tool_configs: List = None,
                format_user_prompts: bool = True,
                return_raw_output: bool = False,
                role: str = '',
                batched: bool = False,
                batch_size: Optional[int] = None) -> List[str]:
        """
        Generate responses using AWS Bedrock.
        
        Args:
            user_prompts (List[str]): List of user prompts to process
            temperature (float): Sampling temperature (higher = more random)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            max_tokens (Optional[int]): Maximum number of tokens to generate
            
        Returns:
            List[str]: Generated responses for each prompt
        """
        formatted_prompts = self.format_prompts(user_prompts) if format_user_prompts else user_prompts
        inference_config = {
            "temperature": temperature,
            "topP": top_p
        }
        
        # Only include maxTokens if it's provided
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        
        outputs = []
        prompts_iterable = tqdm(formatted_prompts, desc=role) if role != '' or role.lower() == 'agent' else formatted_prompts
        for prompt_i, formatted_prompt in enumerate(prompts_iterable):
            logging.info(f"\n\nROLE: {role}")
            # logging.info(f"\n\n[SYSTEM PROMPT]: {self.sys_prompt}")
            logging.info(f"\n\n[USER PROMPT]: {formatted_prompt}")
            tool_config = tool_configs[prompt_i] if tool_configs else None
            success = False
            while not success:
                try:
                    if self.sys_prompt and self.sys_prompt != "" and tool_config and 'deepseek' not in self.model_id:
                        response = self.model.converse(
                            modelId=self.model_id,
                            messages=formatted_prompt,
                            system=self.sys_prompt,
                            toolConfig=tool_config,
                            inferenceConfig=inference_config
                        )
                    elif self.sys_prompt and self.sys_prompt != "":
                        response = self.model.converse(
                            modelId=self.model_id,
                            messages=formatted_prompt,
                            system=self.sys_prompt,
                            inferenceConfig=inference_config
                        )
                    elif tool_config:
                        response = self.model.converse(
                            modelId=self.model_id,
                            messages=formatted_prompt,
                            toolConfig=tool_config,
                            inferenceConfig=inference_config
                        )
                    else:
                        response = self.model.converse(
                            modelId=self.model_id,
                            messages=formatted_prompt,
                            inferenceConfig=inference_config
                        )
                    # Extract the text from the response
                    output = response["output"]["message"]
                    if not return_raw_output:
                        if len(output['content']) > 1 and 'reasoningContent' in output["content"][1]:
                            output = f"<think>{output["content"][1]['reasoningContent']['reasoningText']['text']}</think>{output["content"][0]["text"]}"
                        else:
                            output = output["content"][0]["text"]
                    outputs.append(output)
                    logging.info(f"\n\n[OUTPUT]: {output}")
                    success = True
                except (ClientError, Exception) as e:
                    print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
                    time.sleep(300 if 'deepseek' in self.model_id.lower() else 60)
                    
        return outputs


@ray.remote        
class VllmLM(LM):
    """
    Hugging Face language model implementation using Ray for distributed processing.
    
    This class uses vLLM to efficiently run Hugging Face models with Ray for parallelization.
    
    Attributes:
        model_id (str): HuggingFace model identifier
        n_gpus (int): Number of GPUs to use for tensor parallelism
        tokenizer: HuggingFace tokenizer for the model
    """
    
    def __init__(self, model_id: str, tokenizer_id: Optional[str] = None, n_gpus: int = 8, sys_prompt_paths: List[str] = None) -> None:
        """
        Initialize the Hugging Face language model.
        
        Args:
            model_id (str): HuggingFace model identifier
            tokenizer_id (Optional[str]): Specific tokenizer to use (uses model_id if None)
            n_gpus (int): Number of GPUs to use for tensor parallelism
            sys_prompt_path (Optional[str]): Path to the system prompt file
        """
        self.init_resources(n_gpus)
        super().__init__(model_id, sys_prompt_paths)
        
    def init_resources(self, n_gpus: int) -> None:
        """
        Initialize GPU resources for the model.
        
        Args:
            n_gpus (int): Number of GPUs to use
            
        Raises:
            AssertionError: If not enough GPUs are available
        """
        resources = ray.cluster_resources()
        n_devices = int(resources.get("GPU", 0))
        available_devices = ",".join([str(i) for i in range(n_gpus)]) 
        os.environ['CUDA_VISIBLE_DEVICES'] = available_devices
        print(f"Available devices: {available_devices}")
        if n_devices < n_gpus:
            print(f"Not enough GPUs available. Requested {n_gpus}, but only {n_devices} available. Using {n_devices} GPUs.")
            n_gpus = n_devices
        self.n_gpus = n_gpus
        
    def load_model(self) -> vllm.LLM:
        """
        Load the Hugging Face model using vLLM.
        
        Returns:
            vllm.LLM: Loaded language model instance
        """
        hf_overrides = None
        if 'qwen' in self.model_id.lower() or 'large' in self.model_id.lower() or 'magistral' in self.model_id.lower():
            max_model_len = 40960
        else:
            max_model_len = 131072
            
        use_mistral_format = True if 'mistralai/Mistral-Small-3.2-24B-Instruct-2506' in self.model_id or 'magistral' in self.model_id.lower() else False

        model = vllm.LLM(model=self.model_id,
                         dtype='auto',
                         trust_remote_code=True,
                         quantization=None,
                         tensor_parallel_size=self.n_gpus,
                         tokenizer_mode='mistral' if use_mistral_format else 'auto',
                         config_format='mistral' if use_mistral_format else 'auto',
                         load_format='mistral' if use_mistral_format else 'auto',
                         max_model_len=max_model_len,
                         hf_overrides=hf_overrides)
        return model
    
    def get_tokenizer(self):
        return self.model.get_tokenizer()
    
    def count_tokens(self, prompt: str):
        tokenizer = self.get_tokenizer()
        token_ids = tokenizer.encode(prompt)
        return len(token_ids)
    
    def convert_messages_format(self, all_messages):
        all_converted_messages = []
        for messages in all_messages:
            converted_messages = []
            for message in messages: 
                if message['role'] == 'user':
                    converted_message = {
                        "role": "user", 
                        "content": message["content"]
                        }
                    converted_messages.append(converted_message)
                elif message['role'] == 'assistant':
                    converted_message = []
                    if "content" in message:
                        converted_message.append({
                            "role": "assistant",
                            "content": message["content"]
                        })
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            converted_message.append({
                                "role": "assistant",
                                "content": json.dumps({
                                    "id": tool_call["id"],
                                    "function": 
                                        {
                                            "name": tool_call["function"]["name"],
                                            "arguments": json_repair.loads(tool_call["function"]["arguments"])
                                        }
                                    })
                            })
                    converted_messages.extend(converted_message)
                elif message["role"] == "tool":
                    converted_message = {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "name": message['name'],
                        "content": json.dumps(message['content'])
                        }
                    converted_messages.append(converted_message)
            all_converted_messages.append(converted_messages)
        return all_converted_messages
    
    def format_prompts(self, user_prompts: List) -> List[str]:
        """
        Format user prompts to include system prompt and special tokens.
        
        Args:
            user_prompts (List[str]): Raw user prompts
            
        Returns:
            List[str]: Formatted prompts ready for the model
        """
        formatted_prompts = []
        for user_prompt in user_prompts:
            if 'gemma' in self.model_id.lower():
                formatted_prompt = "<bos>"
                if isinstance(user_prompt, str):
                    formatted_prompt += f"<start_of_turn>user{self.sys_prompt}\n{user_prompt}<end_of_turn>\n"
                elif isinstance(user_prompt, list):
                    for p in user_prompt:
                        role = p['role']
                        if role == 'assistant':
                            role = 'model'
                        formatted_prompt += f"<start_of_turn>{role}\n{p['content']}<end_of_turn>\n"
                formatted_prompt += "<start_of_turn>model\n"
            elif 'stral' in self.model_id.lower():
                formatted_prompt = "<s>"
                if isinstance(user_prompt, str):
                    formatted_prompt += f"[SYSTEM_PROMPT] {self.sys_prompt} [/SYSTEM_PROMPT]"
                    formatted_prompt += f"[INST] {user_prompt} [/INST]"
                elif isinstance(user_prompt, list):
                    for p in user_prompt:
                        role = p['role']
                        if role == 'system':
                            formatted_prompt += f"[SYSTEM_PROMPT] {p['content']} [/SYSTEM_PROMPT]"
                        if role == 'user':
                            formatted_prompt += f"[INST] {p['content']} [/INST]"
                        if role == 'assistant':
                            formatted_prompt += f"{p['content']}</s>"
                        if role == 'tool':
                            formatted_prompt += f"[INST] {p['content']} [/INST]"
            elif 'qwen' in self.model_id.lower():
                formatted_prompt = ""
                if isinstance(user_prompt, str):
                    formatted_prompt += f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
                    formatted_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                elif isinstance(user_prompt, list):
                    for p in user_prompt:
                        role = p['role']
                        if 'content' in p:
                            content = p['content']
                        elif 'tool_calls' in p:
                            content = json.dumps(p['tool_calls'])
                        elif role == 'tool' and 'name' in p and 'tool_call_id' in p:
                            content = f"tool call id: {p['tool_call_id']}, tool name: {p['name']}, content: {p['content']}"
                        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                formatted_prompt += "<|im_start|>assistant\n"
            else:
                if not user_prompt:
                    formatted_prompt = "<|assistant|>\n"
                elif isinstance(user_prompt, str):
                    formatted_prompt = f"<|system|>\n{self.sys_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
                elif isinstance(user_prompt, list):
                    formatted_prompt = ''
                    for p in user_prompt:
                        formatted_prompt += f"<|{p['role']}|>\n{p['content']}\n"
                    formatted_prompt += "<|assistant|>\n"
            formatted_prompts.append(formatted_prompt)           
        return formatted_prompts

    def generate(self, 
                user_prompts: List[str], 
                temperature: float = 1.0, 
                top_p: float = 0.9, 
                max_tokens: Optional[int] = None,
                tool_configs: List = None,
                format_user_prompts: bool = True,
                return_raw_output: bool = False,
                role: str = None) -> List[str]:
        """
        Generate responses using vLLM.
        
        Args:
            user_prompts (List[str]): List of user prompts to process
            temperature (float): Sampling temperature (higher = more random)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            max_tokens (Optional[int]): Maximum number of tokens to generate
            
        Returns:
            List[str]: Generated responses for each prompt
        """
        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        formatted_prompts = self.format_prompts(user_prompts) if format_user_prompts else user_prompts
        
        logging.info(f"\n\nROLE: {role}")
        logging.info(f"\n\n[SYSTEM PROMPT]: {self.sys_prompt}")
        logging.info(f"\n\n[USER PROMPTS]: {formatted_prompts}")
        
        valid = [False for _ in formatted_prompts]
        n_tries = 0
        outputs = ["" for _ in formatted_prompts]
        while (False in valid) and (n_tries < 10):
            new_outputs = self.model.generate(prompts=[f for f_i, f in enumerate(formatted_prompts) if not valid[f_i]],
                                                sampling_params=sampling_params,
                                                use_tqdm=True)
            for output_i, output in enumerate(new_outputs):
                if not valid[output_i]:
                    outputs[output_i] = new_outputs if return_raw_output else new_outputs[output_i].outputs[0].text
            if 'qwen' in self.model_id.lower():
                new_outputs = []
                for output in outputs:
                    if '</think>' in output:
                        output = output.split('</think>')[-1]
                    elif '<think>' in output:
                        output = "Sorry, I didn't finish thinking within the allowed tokens."
                    new_outputs.append(output)
                outputs = new_outputs
            valid = [False if output == "Sorry, I didn't finish thinking within the allowed tokens." else True for output in outputs]
            n_tries += 1
        logging.info(f"\n\n[MODEL OUTPUTS]: {outputs}")
        return outputs
    
    def chat(self, 
            user_prompts: List[str], 
            temperature: float = 1.0, 
            top_p: float = 0.9, 
            max_tokens: Optional[int] = None,
            tool_configs: List = None,
            format_user_prompts: bool = True,
            return_raw_output: bool = False) -> List[str]:
        
        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        formatted_prompts = self.format_prompts(user_prompts) if format_user_prompts else user_prompts
        outputs = self.model.chat(messages=formatted_prompts,
                                sampling_params=sampling_params,
                                use_tqdm=True)
        outputs = [it.outputs[0].text for it in outputs] if not return_raw_output else outputs
        if 'qwen' in self.model_id.lower():
            outputs = [it.split('</think>')[-1] for it in outputs]
        return outputs
    
    
class OpenAILM(LM):
    def __init__(self, model_id, sys_prompt_paths = None):
        super().__init__(model_id, sys_prompt_paths)
        
    def load_model(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it with: export OPENAI_API_KEY='your-key-here'"
            )
        model = openai.OpenAI(api_key=api_key)
        return model
    
    def get_batch_status(self, batch_id: str) -> Any:
        """
        Retrieves the current status object of a specific batch job.

        This function queries the OpenAI API to fetch the latest details
        for a given batch job ID. The returned object includes the job's
        status, such as 'validating', 'in_progress', 'completed', or 'failed'.

        Args:
            batch_id (str): The unique identifier for the batch job.

        Returns:
            Any: The batch object returned by the OpenAI API, containing the
                 job's status and other relevant metadata.
        """
        # Retrieve the batch information using the provided batch ID
        batch_info = self.model.batches.retrieve(batch_id)
        return batch_info
    
    def get_batch(self, batch_id):
        output_file_id = self.model.batches.retrieve(batch_id).output_file_id
        return self.model.files.content(output_file_id).text
    
    def convert_messages_format(self, all_messages):
        all_converted_messages = []
        for messages in all_messages:
            converted_messages = []
            for message in messages: 
                if message['role'] == 'assistant':
                    converted_message = []
                    if "content" in message:
                        converted_message.append({
                            "role": "assistant",
                            "content": message["content"]
                        })
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            converted_message.append({
                                "role": "assistant",
                                "content": json.dumps({
                                    "toolUseId": tool_call["id"],
                                    "name": tool_call["function"]["name"],
                                    "input": json_repair.loads(tool_call["function"]["arguments"])
                                    })
                            })
                    converted_messages.extend(converted_message)
                else:
                    converted_messages.append(deepcopy(message))
            all_converted_messages.append(converted_messages)
        return all_converted_messages
    
    def format_prompts(self, user_prompts: List, spotlighting: bool = False, add_sys_prompt: bool = True) -> List[str]:
        formatted_prompts = []
        for user_prompt in user_prompts:
            if not user_prompt:
                formatted_prompt = None
            elif isinstance(user_prompt, str):
                if spotlighting:
                    user_prompt = re.sub(r'\s', '\u02c6', user_prompt)
                if add_sys_prompt:
                    formatted_prompt = [{"role": "system", "content": self.sys_prompt},
                                        {"role": "user", "content": user_prompt}]
                else:
                    formatted_prompt = [{"role": "user", "content": user_prompt}]
            elif isinstance(user_prompt, list):
                formatted_prompt = deepcopy(user_prompt)
                for p in formatted_prompt:
                    if spotlighting and p['role'] == 'user' and 'content' in p and isinstance(p['content'], str):
                        p['content'] = re.sub(r'\s', '\u02c6', p['content'])
            
            formatted_prompts.append(formatted_prompt)           
        return formatted_prompts
    
    def batched_generate(self, 
                         user_prompts, 
                         tool_configs = None,
                         temperature = 1, 
                         top_p = 0.9, 
                         max_tokens = None,
                         wait_for_completion = False,  # CHANGED: Default to False for backward compatibility
                         return_matched_results = False):  # CHANGED: Clearer parameter name
        batch_requests = []
        for i, messages in enumerate(user_prompts):
            request_body = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p
            }
            if 'o4' not in self.model_id.lower():
                request_body['max_tokens'] = max_tokens
            else:
                request_body['max_completion_tokens'] = max_tokens
            
            if tool_configs:
                request_body['tools'] = tool_configs[i]

            batch_requests.append({
                "custom_id": f"request_{i}",  # CHANGED: Use 0-based indexing to match array indices
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            })

        # Create a temporary file to write the JSONL data
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".jsonl", encoding='utf-8') as tmp_file:
                for req in batch_requests:
                    tmp_file.write(json.dumps(req) + "\n")
                tmp_file_path = tmp_file.name

            # Upload the file to OpenAI for batch processing
            print(f"Uploading batch file: {tmp_file_path}")
            batch_input_file = self.model.files.create(
                file=open(tmp_file_path, "rb"),
                purpose="batch"
            )

            # Create the batch job
            print(f"Creating batch job with file ID: {batch_input_file.id}")
            batch = self.model.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"Batch generation for {len(user_prompts)} prompts."
                    # "description": f"{file_path_inputs}",
                }
            )
            print(f"Batch job created successfully! Batch ID: {batch.id}")
            
            # NEW: Handle different return modes
            if return_matched_results:
                # Return matched results list (with None for failed requests)
                return self.get_batch_results(batch.id)
            elif wait_for_completion:
                # Wait but return the batch object (for backward compatibility)
                self._wait_for_batch_completion(batch.id)
                return batch
            else:
                # UNCHANGED: Return batch object immediately (original behavior)
                return batch

        finally:
            # Ensure the temporary file is deleted
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                print(f"Cleaned up temporary file: {tmp_file_path}")

    def _wait_for_batch_completion(self, batch_id):
        """
        NEW METHOD: Wait for batch completion without returning results.
        Used when wait_for_completion=True but return_matched_results=False.
        """
        import time
        
        print(f"Waiting for batch {batch_id} to complete...")
        
        while True:
            batch = self.model.batches.retrieve(batch_id)
            print(f"Batch status: {batch.status}")
            
            if batch.status == "completed":
                print("Batch completed successfully!")
                break
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch job failed with status: {batch.status}")
            
            time.sleep(30)

    def get_batch_results(self, batch_id):
        """
        Retrieves the results of a completed batch job with input-output matching.
        
        SIMPLIFIED VERSION: Returns raw JSONL results matched to input order.

        Args:
            batch_id (str): The ID of the batch.

        Returns:
            A list where each element corresponds to the input at the same index.
            Each element is the raw json.loads(line) result, or None for missing outputs.
            Returns None if the batch is not complete or failed.
        """
        batch_info = self.model.batches.retrieve(batch_id)

        if batch_info.status != 'completed':
            print(f"Batch job '{batch_id}' is not yet completed. Current status: {batch_info.status}")
            return None

        if not batch_info.output_file_id:
            print(f"Batch job '{batch_id}' completed but has no output file.")
            return None

        # Get number of inputs from batch metadata
        num_inputs = None
        try:
            description = batch_info.metadata.get('description', '')
            if 'prompts' in description:
                import re
                match = re.search(r'(\d+) prompts', description)
                if match:
                    num_inputs = int(match.group(1))
        except:
            pass
        
        if num_inputs is None:
            print("Warning: Could not determine number of inputs from batch metadata. Returning unordered results.")
            # FALLBACK: Return raw results in file order if we can't determine input count
            output_file_content = self.model.files.content(batch_info.output_file_id).read()
            results = []
            for line in output_file_content.decode('utf-8').strip().split('\n'):
                if line:
                    results.append(json.loads(line))
            return results

        # Initialize results array with None values to match input order
        results = [None] * num_inputs
        
        # Retrieve the content of the output file
        output_file_content = self.model.files.content(batch_info.output_file_id).read()

        # Parse each line and match to input order
        successful_count = 0
        failed_count = 0
        
        for line in output_file_content.decode('utf-8').strip().split('\n'):
            if line:
                result_data = json.loads(line)
                custom_id = result_data['custom_id']
                
                # Extract index from custom_id (format: "request_{index}")
                try:
                    index = int(custom_id.split('_')[1])
                    
                    if 0 <= index < num_inputs:
                        # Store the raw result_data at the correct index
                        results[index] = result_data
                        successful_count += 1
                    else:
                        print(f"Warning: Received result with invalid index {index}")
                        failed_count += 1
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse custom_id '{custom_id}': {e}")
                    failed_count += 1

        # Count None values as failed
        none_count = results.count(None)
        print(f"Batch results: {successful_count} processed, {none_count} missing out of {num_inputs} total")
        return results
    
    def generate(self,
                 user_prompts: List[str],
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 max_tokens: Optional[int] = None,
                 tool_configs: List = None,
                 format_user_prompts: bool = True,
                 return_raw_output: bool = False,
                 spotlighting: bool = False,
                 role: str = None,
                 batch_size: Optional[int] = 1) -> Union[List[str], List[Dict]]:
        """
        Generates responses for a list of prompts using either synchronous or batch API calls.

        If the number of prompts is greater than or equal to `batch_size`, this function
        will use the OpenAI Batch API. It will submit the job, poll for completion,
        and then return the results.

        If the number of prompts is less than `batch_size`, it will make a standard,
        synchronous API call for each prompt and return the results directly.

        Args:
            user_prompts (List[str]): List of user prompts to process.
            temperature (float): Sampling temperature for generation.
            top_p (float): Nucleus sampling parameter.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            tool_configs (List, optional): A list of tool configurations, one for each prompt.
            format_user_prompts (bool): Whether to format the prompts with the system message.
            return_raw_output (bool): If True, returns the raw message object instead of the text content.
            role (str, optional): A descriptor for the role, used for logging.
            batch_size (Optional[int]): The threshold for using the Batch API. If the number of
                                        prompts is >= batch_size, batch processing is used.
                                        Defaults to 100.

        Returns:
            Union[List[str], List[Dict]]: A list of generated text content or, if return_raw_output
                                           is True, a list of raw message objects.
        """
        formatted_prompts = self.format_prompts(user_prompts, spotlighting) if format_user_prompts else user_prompts
        outputs = []

        logging.info(f"\n\nROLE: {role}")
        sys_prompt = {"role": "system", "content": self.sys_prompt}
        # logging.info(f"\n\n[SYSTEM PROMPT]: {sys_prompt}")
        
        if tool_configs:
            tool_configs = [json.loads(tool_config) for tool_config in tool_configs]
            
        # logging.info(f"\n\n[TOOL CONFIGS]: {tool_configs}")

        # === Decide between Batch or Synchronous API ===
        if batch_size is not None:
            # --- Use Batch API for large jobs ---
            print(f"Using Batch API.")
            try:
                logging.info(f"\n[USER PROMPTS] {formatted_prompts}")
                created_batch = self.batched_generate(
                    formatted_prompts,
                    tool_configs=tool_configs,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                if created_batch:
                    print(f"Created batch of {len(formatted_prompts)}")
                    print(f"\n--- Waiting for OpenAI batch job to complete (ID: {created_batch.id}) ---")
                    batch_id = created_batch.id
                    while True:
                        status = self.get_batch_status(batch_id)
                        print(f"Current batch status: {status.status}...")
                        if status.status == 'completed':
                            print("Batch completed!")
                            results = self.get_batch_results(batch_id)
                            if results:
                                for result in results:
                                    if not result:
                                        outputs.append(None)
                                    else:
                                        response_body = result.get('response', {}).get('body', {})
                                        message = response_body.get('choices', [{}])[0].get('message', {})
                                        if return_raw_output:
                                            outputs.append(message)
                                        else:
                                            content = json.dumps(message.get('tool_calls')) if message.get('tool_calls') else message.get('content', '')
                                            outputs.append(content)
                            break
                        elif status.status in ['failed', 'expired', 'cancelled']:
                            print(f"Batch job {status.status}. No results will be returned.")
                            raise Exception("Task failed.")
                        time.sleep(30)
            except (ValueError, openai.APIError) as e:
                print(f"An error occurred during the batch process: {e}")
                outputs = ([{}] if return_raw_output else [""]) * len(user_prompts)
        else:
            # --- Use Synchronous API for smaller jobs ---
            print(f"Using synchronous API.")
            for i, prompt in enumerate(tqdm(formatted_prompts, desc="Generating responses")):
                logging.info(f"\n\n[USER PROMPT {i+1}]: {prompt}")
                tool_config = tool_configs[i] if tool_configs and i < len(tool_configs) else None
                success = False
                while not success:
                    try:
                        if 'o4' in self.model_id or 'o3' in self.model_id:
                            completion = self.model.chat.completions.create(
                                model=self.model_id,
                                messages=prompt,
                                max_completion_tokens=max_tokens,
                                tools=tool_config
                            )
                        else:
                            completion = self.model.chat.completions.create(
                                model=self.model_id,
                                messages=prompt,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                tools=tool_config
                            )
                        message = completion.choices[0].message
                        if return_raw_output:
                            outputs.append(message)
                        else:
                            content = json.dumps(message.tool_calls) if message.tool_calls else message.content or ""
                            outputs.append(content)
                        success = True
                    except openai.APIError as e:
                        print(f"An API error occurred: {e}. Retrying in 60 seconds...")
                        time.sleep(60)

        logging.info(f"\n\n[MODEL OUTPUTS]: {outputs}")
        return outputs