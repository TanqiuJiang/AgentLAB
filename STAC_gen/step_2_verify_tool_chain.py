"""
Step 2: Verify Tool Chains

This script verifies and validates the tool chains generated in Step 1, ensuring
they are executable and meaningful within both SHADE_Arena and Agent_SafetyBench environments.
It serves as the second step in the STAC pipeline.

Key Features:
- Uses Verifier class to simulate tool chain execution in both SHADE_Arena and Agent_SafetyBench
- Supports batch processing with configurable batch sizes
- Implements robust error handling with retry mechanisms (up to 20 attempts)
- Provides comprehensive logging and progress tracking
- Incremental processing with resumable execution
- Maintains data integrity even when individual verifications fail

Usage Examples:
    python -m python_scripts.step_2_verify_tool_chain --dataset SHADE_Arena --model gpt-4.1
    python -m python_scripts.step_2_verify_tool_chain --dataset Agent_SafetyBench --model gpt-4.1

Arguments:
    --dataset: Dataset name, either 'SHADE_Arena' or 'Agent_SafetyBench' (required)
    --model: Language model for verification (default: 'gpt-4.1')
    --input_path: Path to Step 1 output JSON (optional, auto-generated if not provided)
    --temperature: Model temperature (default: 0.6)
    --top_p: Model top_p parameter (default: 0.95)
    --max_tokens: Maximum tokens per verification (default: 8192)
    --env: Filter for specific environment (optional)
    --batch_size: Number of attacks to verify simultaneously (default: 512)
    --region: AWS region for model access; only matters when using the Bedrock API (default: 'us-east-1')

Output:
    JSON file with verified attacks including complete interaction histories and execution traces
"""

import argparse
import json
import os
import logging
import datetime
from tqdm import tqdm
from copy import deepcopy

from src.utils import redirect_output, batchify, get_failure_mode
from src.STAC import Verifier
from src.Environments import SHADEArenaEnvironment, AgentSafetyBenchEnvironment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['SHADE_Arena', 'Agent_SafetyBench'],
                        help="Dataset name: 'SHADE_Arena' or 'Agent_SafetyBench'")
    parser.add_argument("--system_design", type=str)
    parser.add_argument("--model", type=str, default='gpt-4.1')
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to Step 1 output JSON (auto-generated if not provided)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=float, default=8192)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--region", type=str, default='us-east-1')
    return parser.parse_args()

def evaluate_shade_arena(args, data, verifier, outdata, outpath):
    for ds in tqdm(batchify(data, args.batch_size), total=(len(data) + args.batch_size - 1) // args.batch_size):

        def save_data(outdata, outpath, ds, messages, final_tool_chains):
            data_to_save = deepcopy(outdata)
            for d_i, d in enumerate(ds):
                if messages[d_i]:
                    d['final_tool_chain'] = final_tool_chains[d_i]
                    d['messages'] = [s for s in messages[d_i] if s['role'] != 'system']
                    data_to_save.append(d)

                    with open(outpath, 'w', encoding='utf-8') as fw:
                        json.dump(data_to_save, fw, indent=2, ensure_ascii=False)

        envs, failure_modes, tool_chains = [], [], []
        try:
            for d in ds:
                logging.info(f"\n===================================== ID={d['id']} =====================================")
                envs.append(SHADEArenaEnvironment(model_id=args.model, env_name=d["environment"]))
                failure_modes.append(d['failure_mode'])
                tool_chains.append(d["tool_chain"])

            verifier.reset(envs, tool_chains, failure_modes)
            success = False
            n_tries = 0
            while not success:
                n_tries += 1
                success = verifier.step(batch_size=args.batch_size)
                save_data(outdata, outpath, ds, verifier.messages, verifier.final_tool_chains)
                if n_tries >= 20:
                    break
        except Exception as e:
            print(e)
            import traceback; traceback.print_exc();

        break

def evaluate_agent_safetybench(args, data, all_envs, verifier, outdata, outpath):
    for batch in tqdm(batchify(list(zip(data, all_envs)), args.batch_size), total=(len(data) + args.batch_size - 1) // args.batch_size):
        ds, envs = zip(*batch)
        def save_data(outdata, outpath, ds, messages, final_tool_chains):
            data_to_save = deepcopy(outdata)
            for d_i, d in enumerate(ds):
                if messages[d_i]:
                    d['final_tool_chain'] = final_tool_chains[d_i]
                    d['messages'] = [s for s in messages[d_i] if s['role'] != 'system']
                    data_to_save.append(d)

                    with open(outpath, 'w', encoding='utf-8') as fw:
                        json.dump(data_to_save, fw, indent=2, ensure_ascii=False)

        failure_modes, tool_chains = [], []
        try:
            for d in ds:
                logging.info(f"\n===================================== ID={d['id']} =====================================")
                failure_modes.append(d['failure_mode'])
                tool_chains.append(d["tool_chain"])

            verifier.reset(envs, tool_chains, failure_modes)
            success = False
            n_tries = 0
            while not success:
                n_tries += 1
                success = verifier.step(batch_size=args.batch_size)
                save_data(outdata, outpath, ds, verifier.messages, verifier.final_tool_chains)
                if n_tries >= 20:
                    break
        except Exception as e:
            print(e)
            import traceback; traceback.print_exc();

        break

def eval_file_shade_arena(args, outpath):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    id = -1
    for env_name in data:
        for failure_mode in data[env_name]:
            for d in data[env_name][failure_mode]["tool_chain_attacks"]:
                id += 1
                if args.env and args.env not in env_name:
                    continue
                this_data = {"id": id,
                             "failure_mode_id": failure_mode,
                             "failure_mode": get_failure_mode(int(failure_mode)),
                             "environment": env_name,
                             "attack_goal": d['objective'],
                             "tool_chain": json.dumps(d)}
                new_data.append(this_data)
    data = new_data

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
        _data.append(d)
    data = _data

    verifier = Verifier(model_id=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    evaluate_shade_arena(args, data, verifier, outdata, outpath)

def eval_file_agent_safetybench(args, outpath):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('Agent_SafetyBench/data/released_data.json') as f:
        asb_data = json.load(f)

    asb_data_lookup = {}
    for d in asb_data:
        asb_data_lookup[d['id']] = d

    new_data = []
    count = -1
    for env_name in data:
        for failure_mode in data[env_name]:
            for d in data[env_name][failure_mode]:
                count += 1
                if args.env and args.env not in env_name:
                    continue
                this_data = {"id": d['original_data_id'],
                             "failure_mode_id": failure_mode,
                             "failure_mode": get_failure_mode(int(failure_mode)),
                             "environment": env_name,
                             "attack_goal": d['tool_chain_attacks'][0]['objective'],
                             "tool_chain": json.dumps(d['tool_chain_attacks'][0])}
                new_data.append(this_data)
    data = new_data

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
    for d_i, d in enumerate(data):
        if d['id'] in outids:
            continue
        _data.append(d)
    data = _data

    asb_data = [asb_data_lookup[d['id']] for d in data]
    envs = [AgentSafetyBenchEnvironment(d, args.model, allow_all_tools=True) for d in asb_data]
    assert len(envs) == len(data)

    verifier = Verifier(model_id=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    evaluate_agent_safetybench(args, data, envs, verifier, outdata, outpath)

def main():
    args = parse_args()

    # Set default input_path based on dataset if not provided
    if args.input_path is None:
        if args.dataset == 'SHADE_Arena':
            args.input_path = 'output/SHADE_Arena/tool_chain_attacks_gpt-4.1.json'
        else:  # Agent_SafetyBench
            args.input_path = 'output/Agent_SafetyBench/tool_chain_attacks_gpt-4.1.json'

    outdir = "/".join(args.input_path.split("/")[:-1]) + f"/verification/{args.input_path.split('/')[-1].replace('.json', '')}_{args.model}"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/gen_res.json"
    log_path = f"{outdir}/gen_log.txt"
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
        eval_file_shade_arena(args, outpath)
    else:  # Agent_SafetyBench
        eval_file_agent_safetybench(args, outpath)

if __name__ == '__main__':
    main()
