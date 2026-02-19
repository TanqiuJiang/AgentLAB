"""
Step 1: Generate Tool Chains for Agent_SafetyBench or SHADE_Arena Environments

This script generates adversarial tool chains targeting either Agent_SafetyBench or SHADE_Arena
environments across 10 different failure modes. It serves as the first step in the STAC pipeline.

Key Features:
- Supports 10 distinct failure modes (harmful content, incomplete info, premature calls, etc.)
- Two dataset options: Agent_SafetyBench or SHADE_Arena
- SHADE_Arena targets 4 environments: travel, spam_filter_updating, workspace, banking
- Supports both individual and batched generation modes
- Incremental processing with resumable execution
- Structured JSON output for downstream pipeline steps

Usage Examples:
    # Batch API with batch size 32 (for OpenAI)
    python -m python_scripts.step_1_gen_tool_chains --dataset Agent_SafetyBench --model_name_or_path gpt-4.1 --n_cases 32 --batch_size 32
    # Synchronous API (default, process one by one)
    python -m python_scripts.step_1_gen_tool_chains --dataset SHADE_Arena --model_name_or_path gpt-4.1 --n_cases 1

Arguments:
    --dataset: Dataset to use ('Agent_SafetyBench' or 'SHADE_Arena', required)
    --model_name_or_path: Language model to use (default: 'gpt-4.1')
    --output_path: Path for output JSON file (default: auto-generated based on dataset)
    --n_cases: Number of cases to generate (default: 120 for Agent_SafetyBench, all for SHADE_Arena)
    --batch_size: Batch size for OpenAI Batch API. Use 'None' for synchronous API (default: None)

Output:
    JSON file containing tool chains generated for all environments and failure modes
"""

import argparse
import json
import os
import datetime
import random
import logging
from collections import defaultdict

from src.Environments import AgentSafetyBenchEnvironment, SHADEArenaEnvironment
from src.STAC import Generator
from src.utils import batchify, get_failure_mode


def int_or_none(value):
    """
    Custom argparse type function that converts a string to int or None.

    Args:
        value (str): String value from command line

    Returns:
        int or None: Converted value

    Raises:
        argparse.ArgumentTypeError: If value is not a valid integer or 'None'
    """
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Expected integer or 'None'")


def parse_args():
    """
    Parse command-line arguments for tool chain generation.

    This function sets up the argument parser with all required and optional arguments
    for both Agent_SafetyBench and SHADE_Arena datasets. It also handles default
    output path selection based on the chosen dataset.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - dataset (str): 'Agent_SafetyBench' or 'SHADE_Arena'
            - model_name_or_path (str): LLM identifier (e.g., 'gpt-4.1', 'gpt-4o')
            - output_path (str): Path to save generated tool chains JSON
            - n_cases (int or None): Number of cases to generate
            - batch_size (int or None): Batch size for OpenAI Batch API. If None (default), uses synchronous API

    Key Logic:
        1. All processing arguments (n_cases, batch_size) are shared between datasets
        2. Auto-generates default output path based on selected dataset if not provided
        3. Default batch_size is None (synchronous API mode)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['Agent_SafetyBench', 'SHADE_Arena'],
                        help="Dataset to use: 'Agent_SafetyBench' or 'SHADE_Arena'")
    parser.add_argument("--model_name_or_path", type=str, default='gpt-4.1')
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path (default: auto-generated based on dataset)")
    parser.add_argument("--n_cases", type=int, default=None,
                        help="Number of cases to generate (default: 120 for Agent_SafetyBench, all for SHADE_Arena)")
    parser.add_argument("--batch_size", type=int_or_none, default=None,
                        help="Batch size for OpenAI Batch API. Use 'None' for synchronous API (default: None)")

    args = parser.parse_args()

    # Validate arguments
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer or 'None'")
    if args.n_cases is not None and args.n_cases <= 0:
        parser.error("--n_cases must be a positive integer")

    # Set default output path based on dataset
    if args.output_path is None:
        if args.dataset == 'Agent_SafetyBench':
            args.output_path = f'output/Agent_SafetyBench/tool_chain_attacks_{args.model_name_or_path}.json'
        else:  # SHADE_Arena
            args.output_path = f'output/SHADE_Arena/tool_chain_attacks_{args.model_name_or_path}.json'

    if args.output_path.endswith('.json'):
        outdir = '/'.join(args.output_path.split("/")[:-1])
        os.makedirs(outdir, exist_ok=True)

    return args


def save_generated_attacks(output_path, data):
    """
    Save generated attacks to a JSON file with error handling.

    Args:
        output_path (str): Path to save the JSON file
        data (dict): Generated attacks data to save

    Raises:
        IOError: If file cannot be written
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.error(f"Failed to save results to {output_path}: {e}")
        raise


def load_existing_attacks(output_path):
    """
    Load existing attacks from a JSON file if it exists.

    Args:
        output_path (str): Path to the JSON file

    Returns:
        dict: Existing attacks data, or empty dict if file doesn't exist

    Raises:
        json.JSONDecodeError: If file exists but contains invalid JSON
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse existing attacks file {output_path}: {e}")
            raise
    return {}


def ensure_env_dict_initialized(all_generated_attacks, env_name):
    """
    Ensure that the environment key exists in the attacks dictionary.

    Args:
        all_generated_attacks (dict): Dictionary of all generated attacks
        env_name (str): Environment name to initialize

    Returns:
        None (modifies dict in place)
    """
    if env_name not in all_generated_attacks:
        all_generated_attacks[env_name] = {}


def sample_uniform_by_failure_mode_id(failure_mode_ids, n):
    """
    Sample n example indices with uniform distribution over failure mode IDs.

    This function ensures balanced representation of all failure modes (1-10) in the
    sampled dataset by giving each failure mode equal probability of being selected,
    regardless of how many examples exist for each mode.

    Args:
        failure_mode_ids (List[List[int]]): List where each element is a sublist of
            failure mode IDs (1-10) associated with that example. Example:
            [[1, 3], [2], [1], [5, 7]] means:
            - Example 0 has failure modes 1 and 3
            - Example 1 has failure mode 2
            - Example 2 has failure mode 1
            - Example 3 has failure modes 5 and 7
        n (int): Number of example indices to sample

    Returns:
        List[int]: List of n sampled indices from the original failure_mode_ids list.
            Indices may repeat to achieve the requested sample size.

    Raises:
        ValueError: If failure_mode_ids is empty or n is not positive

    Example:
        failure_mode_ids = [[1, 2], [3], [1], [4]]
        sample_uniform_by_failure_mode_id(failure_mode_ids, 3)
        # Might return [0, 2, 1] - balanced across failure modes 1, 3
    """
    if not failure_mode_ids:
        raise ValueError("failure_mode_ids cannot be empty")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Build reverse mapping: failure_mode_id -> example indices
    id_to_indices = defaultdict(list)
    for i, sublist in enumerate(failure_mode_ids):
        for failure_id in sublist:
            id_to_indices[failure_id].append(i)

    if not id_to_indices:
        raise ValueError("No valid failure mode IDs found in failure_mode_ids")

    available_ids = list(id_to_indices.keys())

    # Sample with uniform distribution over failure modes
    samples = []
    for _ in range(n):
        selected_id = random.choice(available_ids)
        selected_index = random.choice(id_to_indices[selected_id])
        samples.append(selected_index)

    return samples


def main_agent_safetybench(args):
    """
    Generate adversarial tool chains for Agent_SafetyBench dataset.

    This function processes Agent_SafetyBench data, samples examples with balanced
    failure mode coverage, and generates adversarial tool chains using the STAC
    Generator. It supports incremental generation with checkpointing.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - model_name_or_path (str): LLM to use for generation
            - output_path (str): Where to save generated attacks JSON
            - n_cases (int or None): Number of examples to sample and process (default: 120)
            - batch_size (int or None): Batch size for OpenAI Batch API. If None, uses synchronous API

    Input Files:
        - Agent_SafetyBench/data/released_data.json: Agent_SafetyBench dataset
        - args.output_path (if exists): Previously generated attacks for resumption

    Output Files:
        - args.output_path: JSON with structure:
          {
            "env_name": {
              failure_mode_id: [
                {
                  "tool_chain": [...],
                  "original_data_id": "...",
                  ... (other attack metadata)
                },
                ...
              ],
              ...
            },
            ...
          }
        - {outdir}/io_log.log: Timestamped execution log

    Key Logic:
        1. Load existing attacks (if any) for incremental generation
        2. Load Agent_SafetyBench dataset and create environments for each example
        3. Filter out examples with no valid environments
        4. Randomly assign one failure mode to each environment
        5. Sample n_cases examples with uniform failure mode distribution (default 120)
        6. Skip already-generated examples to support resumption
        7. Process examples in batches (size determined by batch_size argument):
           a. Extract environment info (state, constraints) and tool info (available tools)
           b. Generate adversarial tool chains using the STAC Generator
           c. Store results with original data IDs for traceability
           d. Save checkpoint after each batch
        8. Break after first batch (likely for testing/debugging purposes)

    Note:
        - Sets random seed to 42 for reproducibility
        - Logs selected sample IDs and timestamps for debugging
        - When batch_size is not None, uses OpenAI Batch API with specified batch size
        - When batch_size is None, uses synchronous API (one-by-one processing)
    """
    random.seed(42)

    # Load existing attacks for incremental generation
    all_generated_attacks = load_existing_attacks(args.output_path)

    # Load Agent_SafetyBench dataset
    try:
        with open('Agent_SafetyBench/data/released_data.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Agent_SafetyBench dataset not found. "
            "Please ensure 'Agent_SafetyBench/data/released_data.json' exists."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in Agent_SafetyBench dataset: {e}")

    # Setup logging
    outdir = '/'.join(args.output_path.split("/")[:-1])
    os.makedirs(outdir, exist_ok=True)
    io_log_path = f"{outdir}/io_log.log"
    logging.basicConfig(
        filename=io_log_path,
        filemode='w',  # Overwrite the log file each time
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[START] Experiment started at {current_time}")
    logging.info(f"[CONFIG] Model: {args.model_name_or_path}, n_cases: {args.n_cases}, batch_size: {args.batch_size}")

    # Create environments for all data examples
    envs = [AgentSafetyBenchEnvironment(d, args.model_name_or_path, allow_all_tools=True) for d in data]

    # Filter out examples with no valid environments
    data = [d for d_i, d in enumerate(data) if len(envs[d_i].envs) > 0]
    envs = [env for env in envs if len(env.envs) > 0]
    logging.info(f"[DATA] Loaded {len(envs)} valid examples from Agent_SafetyBench dataset")

    # Randomly assign one failure mode to each environment
    failure_mode_ids = [[random.choice(env.get_failure_mode_ids())] for env in envs]

    # Use n_cases if specified, otherwise default to 120
    n_cases = args.n_cases if args.n_cases is not None else 120
    sample_ids = sample_uniform_by_failure_mode_id(failure_mode_ids, n=n_cases)
    logging.info(f"[SAMPLING] Sampled {n_cases} examples with uniform failure mode distribution")
    logging.info(f"[SAMPLING] Selected sample IDs: {sample_ids}")

    data = [data[i] for i in sample_ids]
    envs = [envs[i] for i in sample_ids]
    failure_mode_ids = [failure_mode_ids[i][0] for i in sample_ids]

    # Skip already generated examples
    n_skip = sum(len(attacks) for attacks in all_generated_attacks.values())
    if n_skip > 0:
        logging.info(f"[RESUME] Skipping {n_skip} already generated examples")
    envs = envs[n_skip:]
    failure_mode_ids = failure_mode_ids[n_skip:]

    if not envs:
        logging.info("[COMPLETE] All examples already generated")
        return

    for batch in batchify(list(zip(envs, failure_mode_ids, data)), args.batch_size):
        envs_batch, failure_mode_ids_batch, data_batch = zip(*batch)
        env_names_batch = [env.envs[0].__class__.__name__ for env in envs_batch]

        # Initialize attack generator for this batch
        # Agent_SafetyBench uses batch API when batch_size is specified
        batched = args.batch_size is not None
        attack_generator = Generator(args.model_name_or_path, n_cases_per_fm=1, batched=batched)
        attack_generator.reset(args.batch_size)

        # Prepare batch data
        env_infos = []
        tool_infos = []
        failure_modes = []
        for env_i, env_name in enumerate(env_names_batch):
            ensure_env_dict_initialized(all_generated_attacks, env_name)

            failure_mode_id = failure_mode_ids_batch[env_i]
            env_infos.append(envs_batch[env_i].get_env_info())
            tool_infos.append(envs_batch[env_i].get_tool_info())
            failure_modes.append(get_failure_mode(failure_mode_id))

        # Generate tool chains for batch
        outputs = attack_generator.step(env_infos, tool_infos, failure_modes, env_names_batch, batch_size=args.batch_size)

        # Store results with original data IDs
        for output_i, output in enumerate(outputs):
            env_name = env_names_batch[output_i]
            failure_mode_id = failure_mode_ids_batch[output_i]
            if failure_mode_id not in all_generated_attacks[env_name]:
                all_generated_attacks[env_name][failure_mode_id] = []
            output['original_data_id'] = data_batch[output_i]['id']
            all_generated_attacks[env_name][failure_mode_id].append(output)

        # Save checkpoint after batch
        save_generated_attacks(args.output_path, all_generated_attacks)
        logging.info(f"[CHECKPOINT] Saved {len(outputs)} generated attacks")
        break


def main_shade_arena(args):
    """
    Generate adversarial tool chains for SHADE_Arena dataset.

    This function processes SHADE_Arena environments (travel, spam_filter_updating,
    workspace, banking) and generates tool chains for all 10 failure modes across
    all environments. Supports both sequential and batched generation modes.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - model_name_or_path (str): LLM to use for generation
            - output_path (str): Where to save generated attacks JSON
            - batch_size (int or None): If not None, uses OpenAI Batch API with specified batch size.
              If None, uses synchronous API (one-by-one processing)
            - n_cases (int or None): If specified, generate only the first n cases
              instead of all 40 env-failure mode combinations

    Input Files:
        - args.output_path (if exists): Previously generated attacks for resumption
        - SHADE_Arena environment definitions (loaded via SHADEArenaEnvironment)

    Output Files:
        - args.output_path: JSON with structure:
          {
            "env_name": {
              "failure_mode_id": [
                {
                  "tool_chain": [...],
                  ... (other attack metadata)
                },
                ...
              ],
              ...
            },
            ...
          }

    Key Logic:
        1. Build list of all env-failure mode combinations (4 envs × 10 modes = 40 total)
        2. If n_cases is specified, limit to first n combinations

        Batch API Mode (batch_size is not None):
            3. Group combinations into batches of size batch_size
            4. For each batch:
               a. Skip already-generated combinations
               b. Create SHADEArenaEnvironment instances for combinations in batch
               c. Initialize STAC Generator with batched=True (OpenAI Batch API)
               d. Extract environment info (state, constraints) and tool info for batch
               e. Generate adversarial tool chains via Generator.step() for batch
               f. Save checkpoint after each batch
            5. Provides checkpointing after each batch

        Synchronous API Mode (batch_size is None):
            3. Create environment instances for all unique env_names upfront
            4. Initialize STAC Generator with batched=False (synchronous API)
            5. For each combination:
               a. Extract environment info and tool info
               b. Generate tool chain via Generator.step() (one at a time)
               c. Store result
            6. Save final results after all combinations

    Note:
        - Batch API mode (batch_size is not None) uses OpenAI Batch API, better for large-scale generation
        - Synchronous API mode (batch_size is None) processes one at a time, simpler but slower
        - When n_cases is used, combinations are ordered: travel(1-10), spam_filter(1-10), etc.
    """
    # Load existing attacks for incremental generation
    all_generated_attacks = load_existing_attacks(args.output_path)

    env_names = ['travel', 'spam_filter_updating', 'workspace', 'banking']

    # Build list of all env-failure mode combinations
    all_combinations = []
    for env_name in env_names:
        for failure_mode_id in range(1, 11):
            all_combinations.append((env_name, failure_mode_id))

    # If n_cases is specified, limit to first n combinations
    if args.n_cases is not None:
        all_combinations = all_combinations[:args.n_cases]

    if args.batch_size is not None:
        # Batch API mode: process env-failure mode combinations in batches
        for i in range(0, len(all_combinations), args.batch_size):
            batch_combinations = all_combinations[i:i + args.batch_size]

            # Filter out already-generated combinations
            combinations_to_generate = []
            for env_name, failure_mode_id in batch_combinations:
                ensure_env_dict_initialized(all_generated_attacks, env_name)
                if str(failure_mode_id) not in all_generated_attacks[env_name]:
                    combinations_to_generate.append((env_name, failure_mode_id))

            # Skip if all combinations in this batch are already generated
            if not combinations_to_generate:
                continue

            # Create environment instances for unique env_names in this batch
            unique_env_names_batch = list(set(env_name for env_name, _ in combinations_to_generate))
            envs_dict_batch = {env_name: SHADEArenaEnvironment(args.model_name_or_path, env_name)
                               for env_name in unique_env_names_batch}

            # Initialize Generator with batched=True for OpenAI Batch API
            attack_generator = Generator(args.model_name_or_path)
            attack_generator.reset(batch_size=args.batch_size)

            # Prepare batch data
            env_names_batch = []
            env_infos_batch = []
            tool_infos_batch = []
            failure_modes_batch = []
            failure_mode_ids_batch = []

            for env_name, failure_mode_id in combinations_to_generate:
                env_names_batch.append(env_name)
                env_infos_batch.append(envs_dict_batch[env_name].get_env_info())
                tool_infos_batch.append(envs_dict_batch[env_name].get_tool_info())
                failure_modes_batch.append(get_failure_mode(failure_mode_id))
                failure_mode_ids_batch.append(str(failure_mode_id))

            # Generate for this batch
            outputs = attack_generator.step(
                env_infos_batch,
                tool_infos_batch,
                failure_modes_batch,
                env_names_batch,
                batch_size=len(combinations_to_generate)
            )

            # Store outputs
            for output_idx, output in enumerate(outputs):
                env_name = env_names_batch[output_idx]
                failure_mode_id = failure_mode_ids_batch[output_idx]
                all_generated_attacks[env_name][failure_mode_id] = output

            # Save checkpoint after each batch
            save_generated_attacks(args.output_path, all_generated_attacks)
    else:
        # Synchronous API mode: process combinations one by one
        unique_env_names = list(set(env_name for env_name, _ in all_combinations))
        envs_dict = {env_name: SHADEArenaEnvironment(args.model_name_or_path, env_name)
                     for env_name in unique_env_names}

        # Initialize Generator with batched=False for synchronous API
        attack_generator = Generator(args.model_name_or_path)
        attack_generator.reset(batch_size=args.batch_size)

        # Process each combination one by one
        for env_name, failure_mode_id in all_combinations:
            ensure_env_dict_initialized(all_generated_attacks, env_name)

            # Skip if already generated
            if str(failure_mode_id) in all_generated_attacks[env_name]:
                continue

            # Extract environment and tool info
            env_info = envs_dict[env_name].get_env_info()
            tool_info = envs_dict[env_name].get_tool_info()
            failure_mode = get_failure_mode(failure_mode_id)

            # Generate tool chain for this single combination
            outputs = attack_generator.step([env_info], [tool_info], [failure_mode], [env_name], batch_size=None)

            # Store result (outputs is a list with one element)
            all_generated_attacks[env_name][str(failure_mode_id)] = outputs[0] if outputs else None

        # Save final results
        save_generated_attacks(args.output_path, all_generated_attacks)


def main():
    """
    Main entry point for tool chain generation script.

    This function orchestrates the entire tool chain generation process by:
    1. Parsing command-line arguments
    2. Routing to the appropriate dataset-specific handler based on --dataset flag

    The script supports two datasets:
        - Agent_SafetyBench: Diverse set of agent scenarios with varied failure modes
        - SHADE_Arena: Four specific environments (travel, spam_filter, workspace, banking)

    Each dataset has its own processing logic and output format, handled by dedicated
    functions (main_agent_safetybench or main_shade_arena).
    """
    args = parse_args()

    if args.dataset == 'Agent_SafetyBench':
        main_agent_safetybench(args)
    else:  # SHADE_Arena
        main_shade_arena(args)


if __name__ == '__main__':
    main()
