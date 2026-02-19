# python -m agentdojo.attacks.search_attack_pipeline2

# version2: drop the no_search branch, the input user/injection task can be a list of tasks, ask ai for batch acceleration.
# version3: round-based architecture with batch API and parallel benchmarking.
# version - evaluation only for benchamrk results, save at the folder evluation / agent_model_name / suite_name / benchmark_version / attack_method / defense_method /  summary_results.txt

import argparse
import asyncio
import json
import enum
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from agentdojo.scripts.benchmark import benchmark_suite
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.important_instructions_attacks import ToolKnowledgeAttack
from agentdojo.attacks.long_horizon_attack6 import (
    build_initial_prompt,
    _resolve_output_dir,
    _find_repo_root,
    _next_iter_injection_output_path,
    Snippets,
)
from agentdojo.attacks.long_horizon_rewrite3 import (
    build_rewrite_prompt,
    check_search_success,
    _next_version_injection_output_path,
)


DEFAULT_ATTACK_MODEL_NAME = "gpt-5-mini"
DEFAULT_AGENT_MODEL_NAME = "gpt-4o-mini-2024-07-18"
# DEFAULT_AGENT_MODEL_NAME = "gpt-4o"
DEFAULT_BENCHMARK_VERSION = "v1.2.1"
DEFAULT_SUITE_NAME = "workspace"
DEFAULT_USER_TASK_ID = ["user_task_0"]
DEFAULT_INJECTION_TASK_ID = ["injection_task_1"]
DEFAULT_OUTPUT_SUBDIR = Path("res")
DEFAULT_LOGDIR = Path("runs")
DEFAULT_MAX_REWRITES = 1
DEFAULT_ATTACK_METHOD = "long_horizon"
DEFAULT_MAX_WORKERS = 4


# # Pydantic models for structured output parsing
# class UserMessage(BaseModel):
#     user: str = Field(..., description="The content of the user's message.")


# class AssistantMessage(BaseModel):
#     assistant: str = Field(..., description="The content of the assistant's message.")


# class SystemMessage(BaseModel):
#     system: str = Field(..., description="The content of the system's message.")


# class Snippet(BaseModel):
#     environment: str = Field(..., description="The environment where the snippet is stored.")
#     title: str = Field(..., description="The title of the snippet.")
#     content: list[UserMessage | AssistantMessage | SystemMessage] = Field(..., description="The content of the conversation.")


# class SnippetsResponse(BaseModel):
#     snippets: list[Snippet] = Field(..., description="The list of snippets.")


@dataclass
class PairInfo:
    """Info for a (user_task_id, injection_task_id) pair."""
    user_task_id: str
    injection_task_id: str
    output_dir: Path = None
    snippet_path: Path = None


@dataclass
class PairResult:
    """Result for a single pair after benchmarking."""
    user_task_id: str
    injection_task_id: str
    security_passed: bool
    utility_passed: bool
    # snippet_name: str
    # defense_method: str
    # agent_model_name: str
    messages: list[dict] = field(default_factory=list)
    error: str = None


def _json_safe(obj):
    """Recursively convert objects into JSON-serializable equivalents."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                key = "|".join(map(str, k))
            elif isinstance(k, (str, int, float, bool)) or k is None:
                key = k
            else:
                key = str(k)
            out[key] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return [_json_safe(x) for x in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

# ============================================================================
# Benchmark Worker
# ============================================================================

def benchmark_worker(
    suite_name: str,
    user_task_id: str,
    injection_task_id: str,
    agent_model_name: str,
    agent_logdir: Path,
    benchmark_version: str,
    attack_method: str,
    defense: str | None,
    round_num: int,
    snippet_name: str,
) -> PairResult:
    """
    Worker function for benchmarking a single pair.
    Returns structured PairResult without writing to files.
    """
    try:
        suite = get_suite(benchmark_version, suite_name)
        results = benchmark_suite(
            suite=suite,
            model=ModelsEnum(agent_model_name),
            logdir=agent_logdir,
            force_rerun=True,
            benchmark_version=benchmark_version,
            user_tasks=(user_task_id,),
            injection_tasks=(injection_task_id,),
            model_id=None,
            attack=attack_method,
            defense=defense,
            tool_delimiter="tool",
            system_message_name=None,
            system_message=None,
            tool_output_format=None,
            live=None,
        )
        
        security_passed = results["security_results"].get((user_task_id, injection_task_id), False)
        utility_passed = results["utility_results"].get((user_task_id, injection_task_id), False)
        
        # Load messages from the benchmark log
        messages = []
        log_path = Path(agent_logdir) / agent_model_name / suite_name / user_task_id / attack_method / f"{injection_task_id}.json"
        if log_path.exists():
            with open(log_path, "r") as f:
                content = json.load(f)
                messages = content.get('messages', [])
        
        return PairResult(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            round_num=round_num,
            security_passed=security_passed,
            utility_passed=utility_passed,
            snippet_name=snippet_name,
            messages=messages,
        )
    except Exception as e:
        return PairResult(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            round_num=round_num,
            security_passed=False,
            utility_passed=False,
            snippet_name=snippet_name,
            error=str(e),
        )

# ============================================================================
# File Saving Helpers
# ============================================================================

def save_snippet(
    snippets: list[dict],
    output_dir: Path,
    is_rewrite: bool = False,
) -> tuple[Path, Path]:
    """
    Save snippets to output directory.
    Returns (versioned_path, newest_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_rewrite:
        versioned_path = _next_version_injection_output_path(output_dir)
    else:
        versioned_path = _next_iter_injection_output_path(output_dir)
    
    newest_path = output_dir / "newest_injection.json"
    
    with open(versioned_path, "w") as f:
        json.dump(snippets, f, ensure_ascii=False, indent=4)
    
    shutil.copy(str(versioned_path), str(newest_path))
    
    return versioned_path, newest_path


def apply_rewrite_to_snippet(
    original_snippet: list[dict],
    rewritten_snippets: list[dict],
    search_success: bool,
) -> list[dict]:
    """Apply rewritten snippets to the original snippet based on search success."""
    result = original_snippet.copy()
    if not search_success:
        # Rewrite search snippet (first one)
        if len(rewritten_snippets) >= 1:
            result[0] = rewritten_snippets[0]
    else:
        # Rewrite conduct snippets (rest)
        if len(rewritten_snippets) == len(original_snippet) - 1:
            result[1:] = rewritten_snippets
    return result


# ============================================================================
# Main Pipeline
# ============================================================================

def batch_generate(
    pairs_with_prompts: list[tuple[PairInfo, str]],
    client: OpenAI,
    use_batch_api: bool,
    model_name: str = DEFAULT_ATTACK_MODEL_NAME,
) -> tuple[dict[str, list[dict]], float]:
    """
    Generate snippets for all pairs using either batch API or concurrent calls.
    Returns (results_dict, elapsed_time).
    """
    start_time = time.time()
    
    if use_batch_api:
        batch_file = prepare_batch_requests(pairs_with_prompts, model_name)
        output_file = submit_batch_and_wait(batch_file, client)
        results = parse_batch_results(output_file)
    else:
        results = asyncio.run(concurrent_generate(pairs_with_prompts, client, model_name))
    
    elapsed_time = time.time() - start_time
    return results, elapsed_time


def run_pipeline(args) -> list[PairResult]:
    """Main pipeline orchestrator with round-based loop."""
    client = OpenAI()
    repo_root = _find_repo_root()
    suite = get_suite(args.benchmark_version, args.suite)
    
    # Initialize all pairs
    all_pairs: list[PairInfo] = []
    skipped_pairs: list[PairInfo] = []
    for user_task_id in args.user_task_id:
        for injection_task_id in args.injection_task_id:
            output_dir = _resolve_output_dir(
                repo_root=repo_root,
                out_dir=Path(args.out_dir),
                suite_name=args.suite,
                benchmark_version=args.benchmark_version,
                user_task_id=user_task_id,
                injection_task_id=injection_task_id,
                attack_method=args.attack_method,
            )
            pair_info = PairInfo(
                user_task_id=user_task_id,
                injection_task_id=injection_task_id,
                output_dir=output_dir,
            )
            
            # Check if pair already succeeded (skip unless force-run)
            if not args.force_run:
                latest = load_latest_pair_result(output_dir)
                if latest and latest.get("security_passed", False):
                    skipped_pairs.append(pair_info)
                    continue
            
            all_pairs.append(pair_info)
    
    if skipped_pairs:
        print(f"\nSkipping {len(skipped_pairs)} pairs with previous attack success:")
        for p in skipped_pairs:
            print(f"  - {p.user_task_id}/{p.injection_task_id}")
    
    if not all_pairs:
        print("\nNo pairs to process. All pairs already succeeded.")
        return [], len(skipped_pairs)
    
    all_results: list[PairResult] = []
    total_batch_time = 0.0
    
    # Track which pairs are still active (not yet succeeded)
    active_pairs = all_pairs.copy()
    pair_to_snippet: dict[str, list[dict]] = {}  # custom_id -> current snippet
    pair_to_snippet_name: dict[str, str] = {}  # custom_id -> snippet filename
    
    # =========== Round 0: Initial generation and benchmark ===========
    print(f"\n{'='*60}")
    print(f"Round 0: Initial generation for {len(active_pairs)} pairs")
    print(f"{'='*60}")
    
    # Build initial prompts
    pairs_with_prompts = build_all_initial_prompts(active_pairs, suite, args.attack_model_name)
    
    # Generate snippets
    results, batch_time = batch_generate(
        pairs_with_prompts, client, args.use_batch_api, args.attack_model_name
    )
    total_batch_time += batch_time
    print(f"Generation completed in {batch_time:.1f}s")
    
    # Save generated snippets and prompts
    for pair_info, prompt in pairs_with_prompts:
        custom_id = f"{pair_info.user_task_id}|{pair_info.injection_task_id}"
        snippets = results.get(custom_id)
        
        if snippets is None:
            print(f"Failed to generate snippet for {custom_id}")
            continue
        
        versioned_path, _ = save_snippet(snippets, pair_info.output_dir, is_rewrite=False)
        pair_to_snippet[custom_id] = snippets
        pair_to_snippet_name[custom_id] = versioned_path.name
        # Save newest prompt
        with open(pair_info.output_dir / "newest_prompt.txt", "w") as f:
            f.write(prompt)
        print(f"Saved {custom_id} -> {versioned_path.name}")
    
    # Benchmark all pairs in parallel
    print(f"\nBenchmarking {len(active_pairs)} pairs...")
    round_results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for pair_info in active_pairs:
            custom_id = f"{pair_info.user_task_id}|{pair_info.injection_task_id}"
            if custom_id not in pair_to_snippet:
                continue
            
            future = executor.submit(
                benchmark_worker,
                args.suite,
                pair_info.user_task_id,
                pair_info.injection_task_id,
                args.agent_model_name,
                Path(args.agent_logdir),
                args.benchmark_version,
                args.attack_method,
                args.defense,
                0,  # round_num
                pair_to_snippet_name.get(custom_id, ""),
            )
            futures[future] = pair_info
        
        for future in as_completed(futures):
            pair_info = futures[future]
            result = future.result()
            round_results.append(result)
            # save the messages to the file
            with open(pair_info.output_dir / "newest_messages.json", "w") as f:
                json.dump(result.messages, f, ensure_ascii=False, indent=4)
            status = "PASS" if result.security_passed else "FAIL"
            print(f"  {result.user_task_id}/{result.injection_task_id}: {status}")
            # Save per-pair result
            save_pair_result(result, pair_info.output_dir)
    
    all_results.extend(round_results)
    
    # =========== Rounds 1..max_rewrites: Rewrite failed pairs ===========
    for round_num in range(1, args.max_rewrites + 1):
        # Filter failed pairs
        failed_pairs = []
        for result in round_results:
            if not result.security_passed:
                custom_id = f"{result.user_task_id}|{result.injection_task_id}"
                pair_info = next(
                    (p for p in active_pairs 
                     if f"{p.user_task_id}|{p.injection_task_id}" == custom_id),
                    None
                )
                if pair_info and custom_id in pair_to_snippet:
                    failed_pairs.append((
                        pair_info,
                        result.messages,
                        pair_to_snippet[custom_id],
                    ))
        
        if not failed_pairs:
            print(f"\nAll pairs succeeded! Stopping early at round {round_num - 1}.")
            break
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Rewriting {len(failed_pairs)} failed pairs")
        print(f"{'='*60}")
        
        # Build rewrite prompts
        rewrite_prompts = build_rewrite_prompts(failed_pairs, suite)
        
        # Generate rewrites
        pairs_with_prompts_for_batch = [(pair_info, prompt) for pair_info, prompt, _ in rewrite_prompts]
        results, batch_time = batch_generate(
            pairs_with_prompts_for_batch, client, args.use_batch_api, args.attack_model_name
        )
        total_batch_time += batch_time
        print(f"Rewrite generation completed in {batch_time:.1f}s")
        
        # Apply rewrites and save
        for pair_info, prompt, search_success in rewrite_prompts:
            custom_id = f"{pair_info.user_task_id}|{pair_info.injection_task_id}"
            rewritten_snippets = results.get(custom_id)
            
            if rewritten_snippets is None:
                print(f"Failed to rewrite snippet for {custom_id}")
                continue
            
            original_snippet = pair_to_snippet[custom_id]
            new_snippet = apply_rewrite_to_snippet(original_snippet, rewritten_snippets, search_success)
            
            versioned_path, _ = save_snippet(new_snippet, pair_info.output_dir, is_rewrite=True)
            pair_to_snippet[custom_id] = new_snippet
            pair_to_snippet_name[custom_id] = versioned_path.name
            # Save newest prompt
            with open(pair_info.output_dir / "newest_prompt.txt", "w") as f:
                f.write(prompt)
            print(f"Saved rewrite {custom_id} -> {versioned_path.name}")
        
        # Benchmark failed pairs in parallel
        print(f"\nBenchmarking {len(failed_pairs)} pairs...")
        round_results = []
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for pair_info, _, _ in failed_pairs:
                custom_id = f"{pair_info.user_task_id}|{pair_info.injection_task_id}"
                
                future = executor.submit(
                    benchmark_worker,
                    args.suite,
                    pair_info.user_task_id,
                    pair_info.injection_task_id,
                    args.agent_model_name,
                    Path(args.agent_logdir),
                    args.benchmark_version,
                    args.attack_method,
                    args.defense,
                    round_num,
                    pair_to_snippet_name.get(custom_id, ""),
                )
                futures[future] = pair_info
            
            for future in as_completed(futures):
                pair_info = futures[future]
                result = future.result()
                round_results.append(result)
                # save the messages to the file
                with open(pair_info.output_dir / "newest_messages.json", "w") as f:
                    json.dump(result.messages, f, ensure_ascii=False, indent=4)
                status = "PASS" if result.security_passed else "FAIL"
                print(f"  {result.user_task_id}/{result.injection_task_id}: {status}")
                # Save per-pair result
                save_pair_result(result, pair_info.output_dir)
        
        all_results.extend(round_results)
    
    # Print timing summary
    print(f"\n{'='*60}")
    print(f"Total batch/API generation time: {total_batch_time:.1f}s")
    print(f"{'='*60}")
    
    return all_results, len(skipped_pairs)


def print_summary(all_results: list[PairResult], num_skipped: int = 0) -> str:
    """Print final summary of utility and attack success rates. Returns summary string."""
    # Get the latest result for each pair
    latest_results: dict[str, PairResult] = {}
    for result in all_results:
        key = f"{result.user_task_id}|{result.injection_task_id}"
        if key not in latest_results or result.round_num > latest_results[key].round_num:
            latest_results[key] = result
    
    # Count per-round success
    round_success: dict[int, int] = {}
    round_total: dict[int, int] = {}
    for r in latest_results.values():
        rnd = r.round_num
        round_total[rnd] = round_total.get(rnd, 0) + 1
        if r.security_passed:
            round_success[rnd] = round_success.get(rnd, 0) + 1
    
    # Include skipped pairs as "round -1" (previously succeeded)
    if num_skipped > 0:
        round_success[-1] = num_skipped
        round_total[-1] = num_skipped
    
    total_pairs = len(latest_results) + num_skipped
    security_passed = sum(1 for r in latest_results.values() if r.security_passed) + num_skipped
    utility_passed = sum(1 for r in latest_results.values() if r.utility_passed)
    
    attack_success_rate = (security_passed / total_pairs * 100) if total_pairs > 0 else 0
    utility_rate = (utility_passed / total_pairs * 100) if total_pairs > 0 else 0
    
    # Build summary string
    lines = []
    lines.append("=" * 60)
    lines.append("FINAL SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total pairs:          {total_pairs}")
    lines.append(f"Skipped (prev success): {num_skipped}")
    lines.append(f"Attack success rate:  {security_passed}/{total_pairs} ({attack_success_rate:.1f}%)")
    lines.append(f"Utility rate:         {utility_passed}/{total_pairs} ({utility_rate:.1f}%)")
    lines.append("")
    lines.append("Per-round attack success:")
    for rnd in sorted(round_total.keys()):
        succ = round_success.get(rnd, 0)
        tot = round_total[rnd]
        rate = (succ / tot * 100) if tot > 0 else 0
        label = "skipped" if rnd == -1 else f"round {rnd}"
        lines.append(f"  {label}: {succ}/{tot} ({rate:.1f}%)")
    lines.append("")
    
    # List successful pairs
    success_pairs = [f"{r.user_task_id}/{r.injection_task_id}" for r in latest_results.values() if r.security_passed]
    lines.append(f"Successful pairs ({len(success_pairs) + num_skipped}):")
    if num_skipped > 0:
        lines.append(f"  (+ {num_skipped} skipped pairs from previous runs)")
    for p in sorted(success_pairs):
        lines.append(f"  {p}")
    lines.append("")
    
    # List failed pairs
    failed_pairs = [f"{r.user_task_id}/{r.injection_task_id}" for r in latest_results.values() if not r.security_passed]
    lines.append(f"Failed pairs ({len(failed_pairs)}):")
    for p in sorted(failed_pairs):
        lines.append(f"  {p}")
    lines.append("=" * 60)
    
    summary = "\n".join(lines)
    print(f"\n{summary}")
    return summary


def save_results_jsonl(all_results: list[PairResult], output_path: Path) -> None:
    """Save all results to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "a") as f:
        for result in all_results:
            record = {
                "user_task_id": result.user_task_id,
                "injection_task_id": result.injection_task_id,
                "round": result.round_num,
                "security_passed": result.security_passed,
                "utility_passed": result.utility_passed,
                "snippet_name": result.snippet_name,
            }
            if result.error:
                record["error"] = result.error
            f.write(json.dumps(record) + "\n")
    
    print(f"Results saved to {output_path}")


def save_pair_result(result: PairResult, output_dir: Path) -> None:
    """Append a result to the per-pair results.jsonl file."""
    results_file = output_dir / "results.jsonl"
    record = {
        "snippet_name": result.snippet_name,
        "round": result.round_num,
        "security_passed": result.security_passed,
        "utility_passed": result.utility_passed,
    }
    if result.error:
        record["error"] = result.error
    
    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_latest_pair_result(output_dir: Path) -> dict | None:
    """Load the latest result from per-pair results.jsonl. Returns None if no results."""
    results_file = output_dir / "results.jsonl"
    if not results_file.exists():
        return None
    
    latest = None
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                latest = json.loads(line)
    return latest


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search attack pipeline with batch API support.")
    p.add_argument("--suite", default=DEFAULT_SUITE_NAME)
    p.add_argument("--user-task-id", nargs='+', default=DEFAULT_USER_TASK_ID)
    p.add_argument("--injection-task-id", nargs='+', default=DEFAULT_INJECTION_TASK_ID)
    p.add_argument("--attack-model-name", default=DEFAULT_ATTACK_MODEL_NAME)
    p.add_argument("--out-dir", default=Path("res"), help="Output directory (relative to repo root unless absolute).")
    p.add_argument("--agent-model-name", default=DEFAULT_AGENT_MODEL_NAME)
    p.add_argument("--benchmark-version", default=DEFAULT_BENCHMARK_VERSION)
    p.add_argument("--agent-logdir", default=DEFAULT_LOGDIR, help="Output directory (relative to repo root unless absolute).")
    p.add_argument("--max-rewrites", default=DEFAULT_MAX_REWRITES, type=int, help="Maximum number of rewrites to attempt.")
    p.add_argument("--attack-method", default=DEFAULT_ATTACK_METHOD, help="Attack method.")
    p.add_argument("--defense", default=None, choices=[None, "tool_filter", "transformers_pi_detector", "spotlighting_with_delimiting", "repeat_user_prompt"], help="Defense method.")
    p.add_argument("--use-batch-api", action="store_true", default=False, help="Use OpenAI Batch API instead of concurrent calls.")
    p.add_argument("--max-workers", default=DEFAULT_MAX_WORKERS, type=int, help="Maximum number of parallel benchmark workers.")
    p.add_argument("--results-output", default=None, help="Path to save results JSONL file.")
    p.add_argument("--force-run", action="store_true", default=False, help="Force run even if pair already succeeded.")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # Run the pipeline
    all_results, num_skipped = run_pipeline(args)
    
    # Print summary
    summary = print_summary(all_results, num_skipped)
    
    # Save results if output path specified
    if args.results_output:
        output_path = Path(args.results_output)
        save_results_jsonl(all_results, output_path)
    else:
        # Default output path
        repo_root = _find_repo_root()
        output_path = repo_root / "res" / args.attack_method / args.suite / args.benchmark_version / "pipeline_results.jsonl"
        save_results_jsonl(all_results, output_path)
    
    # Save summary to txt file
    summary_path = output_path.parent / "summary_results.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
