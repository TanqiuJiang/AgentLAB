"""
Step 4: Evaluate Agents Against STAC Attacks via Adaptive Planning

This script evaluates the effectiveness of adversarial attacks using adaptive planning against
AI agents in both SHADE-Arena and Agent_SafetyBench environments. It serves as the fourth and
final step in the STAC pipeline, testing various defense mechanisms against multi-turn tool attacks.

The benchmark type (SHADE_Arena or Agent_SafetyBench) is automatically determined from the data file,
where each item contains a 'generation_config.dataset' field indicating its benchmark type.

Key Features:
- Implements adaptive planning system with Planner, Judge, and Agent components
- Supports multiple defense mechanisms: no_defense, failure_modes, summarization, reasoning, spotlighting
- Uses adaptive multi-turn interactions (up to 3 turns) to simulate realistic attack scenarios
- Provides detailed scoring for both attack progress and agent helpfulness
- Supports various language models (GPT, Claude, LLaMA, Qwen, etc.)
- Batch processing with configurable batch sizes for efficient evaluation
- Automatically processes both SHADE-Arena and Agent_SafetyBench data from unified input file

Defense Mechanisms:
- no_defense: Baseline agent without additional safety measures
- failure_modes: Agent aware of potential failure modes
- summarization: Agent summarizes user intent before actions
- reasoning: Agent employs harm-benefit reasoning before actions
- spotlighting: User prompts datamarked with special character

Usage:
    python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense no_defense --batch_size 512

Arguments:
    --output_dir: Directory for evaluation results (default: 'data/Eval')
    --model_planner: Language model for attack planning (default: 'gpt-4.1')
    --model_judge: Language model for attack evaluation (default: 'gpt-4.1')
    --model_agent: Language model for target agent (default: 'gpt-4.1')
    --temperature: Model temperature (default: 0.0)
    --top_p: Model top_p parameter (default: 0.95)
    --batch_size: Number of attacks to evaluate simultaneously (default: 1)
    --region: AWS region for model access; only matters if using Bedrock API (default: 'us-west-2')
    --max_n_turns: Maximum evaluation turns (default: 3)
    --defense: Defense mechanism to evaluate (default: 'no_defense')

Input:
    data/STAC_benchmark_data.json - Unified data file containing both SHADE-Arena and Agent_SafetyBench items

Output:
    JSON file with detailed evaluation results including generated attacks, agent responses, and judge evaluations
"""

import argparse
import json
import os
import logging
import datetime
from copy import deepcopy

import numpy as np
from tqdm import tqdm, trange

from prompts.agent import AgentSysemPrompts
from src.Agents import Agent
from src.STAC import Planner, Judge, AdaptivePlanningSystem
from src.Environments import SHADEArenaEnvironment, AgentSafetyBenchEnvironment
from src.utils import redirect_output, batchify, convert_message_between_APIs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/STAC_benchmark_data.json",
                       help="Path to STAC benchmark data (default to STAC benchmark if not specified)")
    parser.add_argument("--output_dir", type=str, default="data/Eval",
                       help="Directory for evaluation results (auto-generated if not specified)")
    parser.add_argument("--model_planner", type=str, default='gpt-4.1')
    parser.add_argument("--model_judge", type=str, default='gpt-4.1')
    parser.add_argument("--model_agent", type=str, default='gpt-4.1')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--region", type=str, default='us-west-2')
    parser.add_argument("--n_gpus_per_model", type=int, default=8)
    parser.add_argument("--max_n_turns", type=int, default=3)
    parser.add_argument("--defense", type=str, default="no_defense",
                       help="Defense mechanism: no_defense, failure_modes, summarization, reasoning, spotlighting")
    args = parser.parse_args()
    return args


def evaluate(args, ds, envs, outdata, outpath, system, benchmark):
    """
    Evaluate a batch of attacks using the adaptive planning system.

    Args:
        args: Command line arguments
        ds: List of data items to evaluate
        envs: List of environments (None for SHADE_Arena, pre-created for Agent_SafetyBench)
        outdata: Existing output data
        outpath: Path to save results
        system: AdaptivePlanningSystem instance
        benchmark: Benchmark type ('SHADE_Arena' or 'Agent_SafetyBench')
    """
    attack_goals, explanations = [], []

    # For SHADE_Arena, create environments here
    if benchmark == 'SHADE_Arena':
        envs = []

    try:
        planner_outputs, crafted_prompts = [], []
        for d_i, d in enumerate(ds):
            logging.info(f"\n===================================== ID={d['id']} =====================================")
            logging.info(f"\nENVIRONMENT: {d['generation_config']['environment']}")
            logging.info(f"\nTOOL CHAIN: {d['attack_plan']}")

            # Create environment for SHADE_Arena
            if benchmark == 'SHADE_Arena':
                envs.append(SHADEArenaEnvironment(model_id=args.model_agent, env_name=d['generation_config']['environment']))

            this_step = d["attack_plan"]['verified_tool_chain'][-1]
            target_tool_call = {"tool_name": this_step['tool_name'], "parameters": this_step['parameters']}
            attack_goals.append(f"The attack goal is: {d['attack_plan']['attack_goal']} The target tool call is: {json.dumps(target_tool_call)}")
            explanations.append(d["attack_plan"]["explanation"])

            if 'planner_history' in d and len(d['planner_history']) > 0:
                planner_outputs.append(json.dumps(d['planner_history'][0]))
                crafted_prompts.append(d['planner_history'][0]['predicted_attack_prompt'])

        if None in planner_outputs or len(planner_outputs) == 0:
            planner_outputs, crafted_prompts = None, None

        def save_data(outdata, outpath, ds, messages, planner_history, judge_history):
            data_to_save = deepcopy(outdata)
            for d_i, d in enumerate(ds):
                if messages[d_i]:
                    d['interaction_history'] = [convert_message_between_APIs(s, args.model_agent) for s in messages[d_i] if s['role'] != 'system']
                    d['planner_history'] += planner_history[d_i]
                    d['judge_history'] = judge_history[d_i]
                    d['n_turns'] = sum([1 for p in d['interaction_history'] if p['role'] == 'user'])
                    d['final_attack_progress'] = judge_history[d_i][-1]['goal_progress']['score']
                    d['final_agent_helpfulness'] = judge_history[d_i][-1]['agent_helpfulness']['score']
                    data_to_save.append(d)

                    with open(outpath, 'w', encoding='utf-8') as fw:
                        json.dump(data_to_save, fw, indent=2, ensure_ascii=False)

        system.reset(envs, attack_goals, explanations, ds)

        # Run previous actions in environment
        for d_i, d in enumerate(ds):
            for m in d['interaction_history']:
                m_gpt = convert_message_between_APIs(m, 'gpt')
                if m['role'] == 'assistant' and 'tool_calls' in m_gpt:
                    arguments = json.loads(m_gpt['tool_calls'][0]['function']['arguments'])
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    tool_call = {'type': 'tool',
                                'tool_call_id': m_gpt['tool_calls'][0]['id'],
                                'tool_name': m_gpt['tool_calls'][0]['function']['name'],
                                'arguments': arguments}
                    envs[d_i].step(tool_call)

        system.reset(envs, attack_goals, explanations, ds)

        # Execute adaptive planning turns
        for turn in trange(args.max_n_turns, desc="  Turns"):
            logging.info(f"\nTurn {turn+1} starts.")
            if not np.any(system.running):
                break
            system.step(planner_outputs=planner_outputs,
                       crafted_prompts=crafted_prompts,
                       batch_size=args.batch_size,
                       spotlighting=args.defense=='spotlighting')
            planner_outputs, crafted_prompts = None, None
            save_data(outdata, outpath, ds, system.agent.messages, system.planner.output_history, system.judge.output_history)
            logging.info(f"\nTurn {turn+1} ends.")

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()


def eval_file(args, outpath):
    """
    Main evaluation function that processes the entire dataset.

    Args:
        args: Command line arguments
        outpath: Path to save results
    """
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)

    # Load input data
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load Agent_SafetyBench data
    with open('Agent_SafetyBench/data/released_data.json') as f:
        asb_data = json.load(f)
    asb_data_lookup = {d['id']: d for d in asb_data}

    # Process interaction history
    for d in data:
        n_turns_to_keep = len(d['attack_plan']['verified_tool_chain'])
        counter_turns = 0
        for t in range(len(d['interaction_history'])):
            if d['interaction_history'][t]['role'] == 'user':
                counter_turns += 1
                if counter_turns == n_turns_to_keep:
                    n_chats_to_keep = t
                    break
        d['interaction_history'] = d['interaction_history'][:n_chats_to_keep]
        d['interaction_history'] = [convert_message_between_APIs(h, args.model_agent) for h in d['interaction_history']]

    # Load existing output data if available
    print(f"Loading existing output data from {outpath}...")
    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                outdata = json.load(f)
        except Exception as e:
            print(e)
            outdata = []
    else:
        outdata = []

    # Filter out already processed data
    outids = [d['id'] for d in outdata]
    _data = []
    print("Already processed: ", len(outids))
    for d in data:
        id = d['id']
        if id not in outids:
            _data.append(d)
    data = _data

    # Separate data by benchmark type
    shade_arena_data = []
    agent_safety_bench_data = []

    for d in data:
        benchmark = d.get('generation_config', {}).get('dataset', None)
        if benchmark == 'SHADE_Arena':
            shade_arena_data.append(d)
        elif benchmark == 'Agent_SafetyBench':
            agent_safety_bench_data.append(d)
        else:
            logging.warning(f"Unknown benchmark type for ID {d.get('id', 'unknown')}: {benchmark}")

    print(f"SHADE_Arena data: {len(shade_arena_data)}")
    print(f"Agent_SafetyBench data: {len(agent_safety_bench_data)}")

    # Initialize Agent
    agent = Agent(model_id=args.model_agent,
                 envs=[],
                 temperature=args.temperature,
                 top_p=args.top_p,
                 max_tokens=2048,
                 sys_prompt_path=None,
                 region=args.region,
                 n_agents=args.batch_size)

    # Set up system prompt for agent
    agent_sys_prompts = AgentSysemPrompts()
    if 'gpt' in args.model_agent.lower() or 'o3' in args.model_agent.lower() or 'o4' in args.model_agent.lower():
        agent.extend_sys_prompt(agent_sys_prompts.basic_sys_prompt_gpt)
    else:
        agent.extend_sys_prompt(agent_sys_prompts.basic_sys_prompt)

    # Apply defense mechanism
    if args.defense == 'failure_modes':
        agent.extend_sys_prompt(agent_sys_prompts.sys_prompt_with_failure_modes)
    elif args.defense == 'summarization':
        agent.extend_sys_prompt(agent_sys_prompts.sys_prompt_with_summarization)
    elif args.defense == 'reasoning':
        agent.extend_sys_prompt(agent_sys_prompts.sys_prompt_with_reasoning)
    elif args.defense == 'spotlighting':
        agent.extend_sys_prompt(agent_sys_prompts.sys_prompt_with_datamarking)

    # Initialize Planner
    planner = Planner(model_id=args.model_planner,
                     temperature=0.15,
                     top_p=args.top_p,
                     max_tokens=4096,
                     sys_prompt_path=f'prompts/planner.md',
                     region=args.region,
                     n_gpus=args.n_gpus_per_model)

    # Initialize Judge
    judge = Judge(model_id=args.model_judge,
                 temperature=0.15,
                 top_p=args.top_p,
                 max_tokens=2048,
                 sys_prompt_path=f"prompts/judge.md",
                 region=args.region,
                 n_gpus=args.n_gpus_per_model)

    # Initialize Adaptive Planning System
    system = AdaptivePlanningSystem(planner=planner,
                                   judge=judge,
                                   agent=agent,
                                   max_n_turns=args.max_n_turns)

    # Process SHADE_Arena batches
    if len(shade_arena_data) > 0:
        print(f"\nProcessing SHADE_Arena data...")
        for batch in tqdm(batchify(shade_arena_data, args.batch_size),
                         total=(len(shade_arena_data) + args.batch_size - 1) // args.batch_size,
                         desc="SHADE_Arena"):
            evaluate(args, batch, None, outdata, outpath, system, 'SHADE_Arena')
            if os.path.exists(outpath):
                with open(outpath, 'r', encoding='utf-8') as f:
                    outdata = json.load(f)

    # Process Agent_SafetyBench batches
    if len(agent_safety_bench_data) > 0:
        print(f"\nProcessing Agent_SafetyBench data...")
        # Create environments for Agent_SafetyBench
        asb_data_for_envs = [asb_data_lookup[d['id']] for d in agent_safety_bench_data]
        envs = [AgentSafetyBenchEnvironment(d, args.model_agent, allow_all_tools=True) for d in asb_data_for_envs]
        assert len(envs) == len(agent_safety_bench_data)

        # Batch data and environments together
        for batch in tqdm(batchify(list(zip(agent_safety_bench_data, envs)), args.batch_size),
                         total=(len(agent_safety_bench_data) + args.batch_size - 1) // args.batch_size,
                         desc="Agent_SafetyBench"):
            ds, envs_batch = zip(*batch)
            evaluate(args, ds, envs_batch, outdata, outpath, system, 'Agent_SafetyBench')
            if os.path.exists(outpath):
                with open(outpath, 'r', encoding='utf-8') as f:
                    outdata = json.load(f)


def main():
    args = parse_args()

    outdir = f"{args.output_dir}/{args.model_planner}/{args.model_agent}/{args.defense}"
    outpath = f"{outdir}/gen_res.json"
    log_path = f"{outdir}/gen_log.txt"

    os.makedirs(outdir, exist_ok=True)

    # Set up logging
    io_log_path = f"{outdir}/io_log.log"
    logging.basicConfig(
        filename=io_log_path,
        filemode='w',  # Overwrite the log file each time
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Experiment ran at {current_time}")
    logging.info(f"Model Agent: {args.model_agent}")
    logging.info(f"Defense: {args.defense}")

    redirect_output(log_path, 'w')
    eval_file(args, outpath)


if __name__ == '__main__':
    main()
