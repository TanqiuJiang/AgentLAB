#!/usr/bin/env python3
"""
WebShop Bridge Process
=======================
Reads JSON commands from stdin, executes them against the WebShop env,
writes JSON responses to stdout. One line in, one line out.

This script is spawned automatically by WebShopSubprocessClient.
It should NEVER be run manually.
"""

import sys
import json
import re
import random
import os

# Critical: web_agent_site is not pip-installed. It must be importable
# from the directory containing this script (envs/webshop/).
BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

# Redirect any print statements from WebShop/gym to stderr
# so they don't corrupt our JSON protocol on stdout
import io
_real_stdout = sys.stdout
sys.stdout = sys.stderr

try:
    import gym
    from web_agent_site.envs import WebAgentTextEnv
    print("✓ WebShop loaded", file=sys.stderr)
except ImportError as e:
    # Write error as JSON to real stdout so the client can read it
    _real_stdout.write(json.dumps({"error": f"Failed to import WebShop: {e}"}) + "\n")
    _real_stdout.flush()
    sys.exit(1)


def get_page_type(obs):
    obs_lower = obs.lower()
    if "buy now" in obs_lower and "price:" in obs_lower:
        return "item"
    elif "page 1" in obs_lower or "total results" in obs_lower:
        return "results"
    return "search"


def extract_instruction(obs):
    match = re.search(r'Instruction:?\s*(?:\[SEP\])?\s*(.+?)(?:\[SEP\]|$)', obs, re.I)
    return match.group(1).strip() if match else ""


def extract_price(obs):
    match = re.search(r'Price:\s*\$(\d+\.?\d*)', obs)
    return float(match.group(1)) if match else 0.0


def extract_product_name(obs):
    match = re.search(r'\[SEP\]\s*([^[\]]{10,80}?)\s*\[SEP\]\s*Price:', obs)
    return match.group(1).strip() if match else ""


def extract_all_prices(obs):
    return [float(p) for p in re.findall(r'\$(\d+\.?\d*)', obs) if float(p) > 0]


def do_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated
    elif len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done
    else:
        raise ValueError(f"Unexpected step result length: {len(result)}")


def main():
    # Read init config from first line
    init_line = sys.stdin.readline().strip()
    if not init_line:
        _real_stdout.write(json.dumps({"error": "No init config received"}) + "\n")
        _real_stdout.flush()
        return

    try:
        init_config = json.loads(init_line)
    except json.JSONDecodeError as e:
        _real_stdout.write(json.dumps({"error": f"Invalid init JSON: {e}"}) + "\n")
        _real_stdout.flush()
        return

    # Create environment
    num_products = init_config.get("num_products", None)
    print(f"Creating WebShop env (num_products={num_products})...", file=sys.stderr)

    try:
        if num_products:
            env = gym.make('WebAgentTextEnv-v0', observation_mode='text',
                           num_products=num_products)
        else:
            env = gym.make('WebAgentTextEnv-v0', observation_mode='text')
    except Exception as e:
        _real_stdout.write(json.dumps({"error": f"Failed to create env: {e}"}) + "\n")
        _real_stdout.flush()
        return

    # Determine num_tasks
    num_tasks = 12087
    try:
        if hasattr(env, 'num_goals'):
            num_tasks = env.num_goals
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'num_goals'):
            num_tasks = env.unwrapped.num_goals
    except:
        pass

    # Send ready signal
    _real_stdout.write(json.dumps({"status": "ready", "num_tasks": num_tasks}) + "\n")
    _real_stdout.flush()
    print(f"Bridge ready. num_tasks={num_tasks}", file=sys.stderr)

    # Main loop: read commands, execute, respond
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            _real_stdout.write(json.dumps({"error": f"Invalid JSON: {line[:100]}"}) + "\n")
            _real_stdout.flush()
            continue

        command = cmd.get("command", "")

        try:
            if command == "reset":
                task_idx = cmd.get("task_idx", None)
                if task_idx is not None:
                    random.seed(task_idx % num_tasks)

                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result

                _real_stdout.write(json.dumps({
                    "observation": obs,
                    "task_instruction": extract_instruction(obs),
                    "page_type": get_page_type(obs),
                    "all_prices_on_page": extract_all_prices(obs),
                }) + "\n")

            elif command == "step":
                action = cmd.get("action", "")
                obs, reward, done = do_step(env, action)

                _real_stdout.write(json.dumps({
                    "observation": obs,
                    "reward": reward,
                    "done": done,
                    "page_type": get_page_type(obs),
                    "current_product_price": extract_price(obs),
                    "current_product_name": extract_product_name(obs),
                    "all_prices_on_page": extract_all_prices(obs),
                }) + "\n")

            elif command == "ping":
                _real_stdout.write(json.dumps({"status": "alive"}) + "\n")

            elif command == "quit":
                _real_stdout.write(json.dumps({"status": "bye"}) + "\n")
                _real_stdout.flush()
                break

            else:
                _real_stdout.write(json.dumps({"error": f"Unknown command: {command}"}) + "\n")

        except Exception as e:
            _real_stdout.write(json.dumps({"error": str(e)}) + "\n")

        _real_stdout.flush()

    print("Bridge shutting down.", file=sys.stderr)
    env.close()


if __name__ == "__main__":
    main()