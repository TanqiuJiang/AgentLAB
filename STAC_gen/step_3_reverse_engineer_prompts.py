"""
Step 3: Reverse Engineer Attack Prompts

This script reverse engineers user prompts that would lead AI agents to execute the verified
tool chains from Step 2, creating realistic adversarial scenarios for agent evaluation. It
serves as the third step in the STAC pipeline for both SHADE_Arena and Agent_SafetyBench.

Key Features:
- Uses PromptWriter class to analyze target tool chains and generate natural prompts
- Synthesizes multi-turn conversations with realistic progression
- Maintains conversation context and coherence throughout interactions
- Creates believable dialogue that doesn't immediately reveal malicious intent
- Handles environment-specific constraints and interaction patterns
- Supports batch processing for efficient large-scale generation

Usage Examples:
    python -m python_scripts.step_3_reverse_engineer_prompts --dataset SHADE_Arena --model Qwen/Qwen3-32B --temperature 0.6 --batch_size 512
    python -m python_scripts.step_3_reverse_engineer_prompts --dataset Agent_SafetyBench --model Qwen/Qwen3-32B --temperature 0.6 --batch_size 512

Arguments:
    --dataset: Dataset name, either 'SHADE_Arena' or 'Agent_SafetyBench' (required)
    --output_dir: Directory containing Step 2 verification results (optional, auto-generated if not provided)
    --model: Language model for prompt generation (default: 'Qwen/Qwen3-32B')
    --temperature: Model temperature for generation (default: 0.6)
    --top_p: Model top_p parameter (default: 0.95)
    --batch_size: Number of attacks to process simultaneously (default: 1)
    --region: AWS region for model access; only matters if using Bedrock API (default: 'us-east-1')

Output:
    JSON file with reverse-engineered prompts and complete multi-turn interaction histories
"""

import argparse
import json
import os
import logging
import datetime

from tqdm import tqdm

from src.STAC import PromptWriter
from src.Environments import SHADEArenaEnvironment, AgentSafetyBenchEnvironment
from src.utils import redirect_output, batchify, convert_message_between_APIs, gen_tool_call_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['SHADE_Arena', 'Agent_SafetyBench'],
                        help="Dataset name: 'SHADE_Arena' or 'Agent_SafetyBench'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing Step 2 verification results (auto-generated if not provided)")
    parser.add_argument("--model", type=str, default='Qwen/Qwen3-32B')
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--region", type=str, default='us-east-1')
    return parser.parse_args()

def evaluate_agent_safetybench(args, ds, envs, system):
    try:
        for d_i, d in enumerate(ds):
            d['interaction_history'] = []

        for t in range(10):
            logging.info(f"\n===================================== TURN={t+1} =====================================")
            target_actions = []
            target_env_feedbacks = []
            for d_i, d in enumerate(ds):
                tool_chain = d['final_tool_chain']['tool_chain']

                if len(tool_chain) > t:
                    tool_call_id = gen_tool_call_id()
                    tool_call = {'type': 'tool',
                                'tool_call_id': tool_call_id,
                                'tool_name': tool_chain[t]["tool_name"],
                                'arguments': tool_chain[t]['parameters']}
                    target_action = {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                                "name": tool_chain[t]["tool_name"],
                                                "arguments": json.dumps(tool_chain[t]['parameters'])
                                                }
                                    }
                    target_actions.append(target_action)
                    target_env_feedbacks += envs[d_i].step(tool_call)
                else:
                    target_actions.append(None)
                    target_env_feedbacks.append(None)

            if sum([1 for a in target_actions if a]) == 0:
                break

            user_prompts = system.step([d['interaction_history'] for d in ds], target_actions)
            for d_i, d in enumerate(ds):
                if user_prompts[d_i]:
                    d['interaction_history'].append(convert_message_between_APIs({"role": "user", "content": user_prompts[d_i]}, args.model))
                    d['interaction_history'].append(convert_message_between_APIs({"role": "assistant", "tool_calls": [target_actions[d_i]]}, args.model))
                    d['interaction_history'].append(convert_message_between_APIs(target_env_feedbacks[d_i], args.model))

    except Exception as e:
        print(e)
        import traceback; traceback.print_exc();
        return None

    return ds

def evaluate_shade_arena(args, ds, system):
    envs = []
    try:
        for d_i, d in enumerate(ds):
            d['interaction_history'] = []
            envs.append(SHADEArenaEnvironment(model_id='gpt', env_name=d["environment"]))

        for t in range(10):
            logging.info(f"\n===================================== TURN={t+1} =====================================")
            target_actions = []
            target_env_feedbacks = []
            for d_i, d in enumerate(ds):
                tool_chain = d['final_tool_chain']['tool_chain']

                if len(tool_chain) > t:
                    tool_call_id = gen_tool_call_id()
                    tool_call = {'type': 'tool',
                                'tool_call_id': tool_call_id,
                                'tool_name': tool_chain[t]["tool_name"],
                                'arguments': tool_chain[t]['parameters']}
                    target_action = {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                                "name": tool_chain[t]["tool_name"],
                                                "arguments": json.dumps(tool_chain[t]['parameters'])
                                                }
                                    }
                    target_actions.append(target_action)
                    target_env_feedbacks += envs[d_i].step(tool_call)
                else:
                    target_actions.append(None)
                    target_env_feedbacks.append(None)

            if sum([1 for a in target_actions if a]) == 0:
                break

            user_prompts = system.step([d['interaction_history'] for d in ds], target_actions)
            for d_i, d in enumerate(ds):
                if user_prompts[d_i]:
                    d['interaction_history'].append(convert_message_between_APIs({"role": "user", "content": user_prompts[d_i]}, args.model))
                    d['interaction_history'].append(convert_message_between_APIs({"role": "assistant", "tool_calls": [target_actions[d_i]]}, args.model))
                    d['interaction_history'].append(convert_message_between_APIs(target_env_feedbacks[d_i], args.model))

    except Exception as e:
        print(e)
        import traceback; traceback.print_exc();
        return None

    return ds

def eval_file_agent_safetybench(args, path, outpath):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('Agent_SafetyBench/data/released_data.json') as f:
        asb_data = json.load(f)

    asb_data_lookup = {}
    for d in asb_data:
        asb_data_lookup[d['id']] = d

    for d in data:
        del d['messages']
        del d['tool_chain']

    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                outdata = json.load(f)
        except Exception as e:
            print(e)
            outdata = []
    else:
        outdata = []

    outids = set([d['id'] for d in outdata])
    _data = []
    print("data: ", len(outids))
    for d in data:
        if d['id'] in outids:
            continue
        if 'finish' in d:
            if d['finish'] != 1:
                continue
        _data.append(d)
    data = _data

    data = [d for d in data if d['final_tool_chain'] and d['final_tool_chain'] != "" and 'tool_chain' in d['final_tool_chain']]

    asb_data = [asb_data_lookup[d['id']] for d in data]
    envs = [AgentSafetyBenchEnvironment(d, args.model, allow_all_tools=True) for d in asb_data]
    assert len(envs) == len(data)

    reverse_prompt_writer = PromptWriter(model_id=args.model,
                                        temperature=args.temperature,
                                        top_p=args.top_p)

    for batch in tqdm(batchify(list(zip(data, envs)), args.batch_size), total=(len(data) + args.batch_size - 1) // args.batch_size):
        ds, envs_batch = zip(*batch)
        final_result = evaluate_agent_safetybench(args, ds, envs_batch, reverse_prompt_writer)
        for i in range(len(final_result)):
            d = final_result[i]
            outdata.append(d)

            with open(outpath, 'w', encoding='utf-8') as fw:
                json.dump(outdata, fw, indent=2, ensure_ascii=False)

def eval_file_shade_arena(args, path, outpath):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for d in data:
        del d['messages']
        del d['tool_chain']

    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                outdata = json.load(f)
        except Exception as e:
            print(e)
            outdata = []
    else:
        outdata = []

    outids = set([d['id'] for d in outdata])
    _data = []
    print("data: ", len(outids))
    for d in data:
        if d['id'] in outids:
            continue
        if 'finish' in d:
            if d['finish'] != 1:
                continue
        _data.append(d)
    data = _data

    data = [d for d in data if d['final_tool_chain'] and d['final_tool_chain'] != "" and 'tool_chain' in d['final_tool_chain']]

    reverse_prompt_writer = PromptWriter(model_id=args.model,
                                        temperature=args.temperature,
                                        top_p=args.top_p)

    for batch in tqdm(batchify(data, args.batch_size), total=(len(data) + args.batch_size - 1) // args.batch_size):
        # skip the CodeAssistant environment as it's faulty
        final_result = evaluate_shade_arena(args, batch, reverse_prompt_writer)
        for i in range(len(final_result)):
            d = final_result[i]
            outdata.append(d)

            with open(outpath, 'w', encoding='utf-8') as fw:
                json.dump(outdata, fw, indent=2, ensure_ascii=False)

def main():
    args = parse_args()

    # Set default output_dir based on dataset if not provided
    if args.output_dir is None:
        if args.dataset == 'SHADE_Arena':
            args.output_dir = 'output/SHADE_Arena/verification/tool_chain_attacks_gpt-4.1_gpt-4.1'
        else:  # Agent_SafetyBench
            args.output_dir = 'output/Agent_SafetyBench/verification/tool_chain_attacks_gpt-4.1_gpt-4.1'

    path = f"{args.output_dir}/gen_res.json"

    outdir = f"{args.output_dir}/Prompts/{args.model}/"
    outpath = f"{outdir}/gen_res.json"
    log_path = f"{outdir}/gen_log.txt"

    os.makedirs(outdir, exist_ok=True)

    io_log_path = f"{outdir}/io_log.log"
    logging.basicConfig(
        filename=io_log_path,
        filemode='w', # Overwrite the log file each time
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Experiment ran at {current_time}")

    redirect_output(log_path, 'w')

    # Call the appropriate eval_file function based on dataset
    if args.dataset == 'SHADE_Arena':
        eval_file_shade_arena(args, path, outpath)
    else:  # Agent_SafetyBench
        eval_file_agent_safetybench(args, path, outpath)

if __name__ == '__main__':
    main()
