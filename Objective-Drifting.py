#!/usr/bin/env python3
"""
Objective Drifting Attack — Top-Level Runner
==============================================
Dispatches to the appropriate attack script (v12 aggressive or v13 subtle)
and handles WebShop environment setup automatically.

Usage:
    # v12 aggressive attacks (one victim at a time)
    python Objective-Drifting.py --mode aggressive --victim gpt4o --num_tasks 100 --verbose
    python Objective-Drifting.py --mode aggressive --victim llama --num_tasks 100
    python Objective-Drifting.py --mode aggressive --victim qwen --num_tasks 100

    # v13 subtle attacks (multi-victim, for stronger models)
    python Objective-Drifting.py --mode subtle --victims claude gemini --num_tasks 50
    python Objective-Drifting.py --mode subtle --victims claude gemini gpt --num_tasks 50

    # Run all API-accessible victims (no vLLM required)
    python Objective-Drifting.py --mode all-api --num_tasks 100

    # Run everything including local vLLM models
    python Objective-Drifting.py --mode all --num_tasks 100

    # Environment management
    python Objective-Drifting.py --setup-only
    python Objective-Drifting.py --check-env
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
WEBSHOP_ENV_DIR = ROOT_DIR / "envs" / "webshop"
SETUP_SCRIPT = WEBSHOP_ENV_DIR / "setup_webshop.sh"
V12_SCRIPT = ROOT_DIR / "attacks" / "goal_drift" / "goal_drift_v12.py"
V13_SCRIPT = ROOT_DIR / "attacks" / "goal_drift" / "goal_drift_v13_subtle.py"
CONDA_ENV_NAME = "webshop"

# ============================================================================
# VICTIM PRESETS
# ============================================================================

# v12 aggressive attack victims (one at a time via unified v12 script)
V12_VICTIMS = {
    "gpt4o": {
        "display": "GPT-4o",
        "args": ["--victim_model", "gpt-4o"],
    },
    "llama": {
        "display": "Llama 3.1 8B Instruct (vLLM)",
        "args": ["--victim_model", "meta-llama/Llama-3.1-8B-Instruct",
                 "--victim_base_url", "http://localhost:8000/v1",
                 "--victim_api_key", "EMPTY"],
        "requires_vllm": True,
    },
    "qwen": {
        "display": "Qwen3-30B-A3B (vLLM)",
        "args": ["--victim_model", "Qwen/Qwen3-30B-A3B",
                 "--victim_base_url", "http://localhost:8001/v1",
                 "--victim_api_key", "EMPTY"],
        "requires_vllm": True,
    },
}

# v13 subtle attack victims (passed as --victims to multi-victim runner)
V13_VICTIMS = ["claude", "gemini", "gpt"]

# Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'


# ============================================================================
# WEBSHOP ENVIRONMENT MANAGEMENT
# ============================================================================

def check_conda() -> bool:
    return shutil.which("conda") is not None


def conda_env_exists() -> bool:
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return any(CONDA_ENV_NAME == os.path.basename(p)
                       for p in data.get("envs", []))
    except:
        pass
    return False


def verify_env() -> bool:
    if not SETUP_SCRIPT.exists():
        return False
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--check"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def setup_env() -> bool:
    if not SETUP_SCRIPT.exists():
        print(f"{RED}Setup script not found: {SETUP_SCRIPT}{NC}")
        return False
    print(f"\n{CYAN}Running WebShop environment setup...{NC}")
    print(f"This downloads product data and builds search indexes.")
    print(f"One-time process, may take 10-30 minutes.\n")
    result = subprocess.run(["bash", str(SETUP_SCRIPT)], timeout=3600)
    return result.returncode == 0


def ensure_env_ready() -> bool:
    if not check_conda():
        print(f"{RED}Error: conda is required but not found.{NC}")
        print(f"Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        return False
    if conda_env_exists() and verify_env():
        print(f"{GREEN}✓ WebShop environment is ready.{NC}")
        return True
    if conda_env_exists():
        print(f"{YELLOW}WebShop env exists but verification failed. Re-running setup...{NC}")
    else:
        print(f"{YELLOW}WebShop environment not found. Setting up...{NC}")
    return setup_env()


# ============================================================================
# ATTACK DISPATCHERS
# ============================================================================

def run_v12_attack(victim_key: str, extra_args: list):
    """Run a v12 aggressive attack against one victim."""
    if victim_key not in V12_VICTIMS:
        print(f"{RED}Unknown victim: {victim_key}{NC}")
        print(f"Available: {', '.join(V12_VICTIMS.keys())}")
        sys.exit(1)

    preset = V12_VICTIMS[victim_key]

    if preset.get("requires_vllm"):
        print(f"{YELLOW}Note: {preset['display']} requires a running vLLM server.{NC}")

    if not V12_SCRIPT.exists():
        print(f"{RED}Attack script not found: {V12_SCRIPT}{NC}")
        sys.exit(1)

    cmd = [sys.executable, str(V12_SCRIPT)] + preset["args"] + extra_args

    print(f"\n{CYAN}{'='*60}")
    print(f" Aggressive Attack (v12) → {preset['display']}")
    print(f"{'='*60}{NC}\n")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def run_v13_attack(victim_keys: list, extra_args: list):
    """Run a v13 subtle attack against one or more victims."""
    if not V13_SCRIPT.exists():
        print(f"{RED}Attack script not found: {V13_SCRIPT}{NC}")
        sys.exit(1)

    cmd = [sys.executable, str(V13_SCRIPT),
           "--victims"] + victim_keys + extra_args

    print(f"\n{CYAN}{'='*60}")
    print(f" Subtle Persuasion Attack (v13) → {', '.join(victim_keys)}")
    print(f"{'='*60}{NC}\n")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def run_all_api(extra_args: list):
    """Run all API-accessible victims (no vLLM required)."""
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f" Running ALL API victims")
    print(f"{'='*60}{NC}\n")

    results = {}

    # v12 aggressive on GPT-4o
    print(f"\n{BOLD}[1/2] v12 Aggressive → GPT-4o{NC}")
    rc = run_v12_attack("gpt4o", extra_args)
    results["v12_gpt4o"] = rc

    # v13 subtle on Claude + Gemini + GPT
    print(f"\n{BOLD}[2/2] v13 Subtle → Claude, Gemini, GPT{NC}")
    rc = run_v13_attack(["claude", "gemini", "gpt"], extra_args)
    results["v13_subtle"] = rc

    # Summary
    print(f"\n{BOLD}{'='*60}")
    print(f" Summary")
    print(f"{'='*60}{NC}")
    for name, rc in results.items():
        status = f"{GREEN}✓{NC}" if rc == 0 else f"{RED}✗ (exit {rc}){NC}"
        print(f"  {name}: {status}")

    return 0 if all(rc == 0 for rc in results.values()) else 1


def run_all(extra_args: list):
    """Run everything including vLLM models."""
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f" Running ALL victims (API + vLLM)")
    print(f"{'='*60}{NC}\n")

    results = {}
    step = 1
    total = len(V12_VICTIMS) + 1  # all v12 victims + one v13 run

    # v12 aggressive on each victim
    for vk, preset in V12_VICTIMS.items():
        print(f"\n{BOLD}[{step}/{total}] v12 Aggressive → {preset['display']}{NC}")
        if preset.get("requires_vllm"):
            print(f"{YELLOW}  Requires vLLM. Skipping if not available...{NC}")
        rc = run_v12_attack(vk, extra_args)
        results[f"v12_{vk}"] = rc
        step += 1

    # v13 subtle on Claude + Gemini + GPT
    print(f"\n{BOLD}[{step}/{total}] v13 Subtle → Claude, Gemini, GPT{NC}")
    rc = run_v13_attack(["claude", "gemini", "gpt"], extra_args)
    results["v13_subtle"] = rc

    # Summary
    print(f"\n{BOLD}{'='*60}")
    print(f" Summary")
    print(f"{'='*60}{NC}")
    for name, rc in results.items():
        status = f"{GREEN}✓{NC}" if rc == 0 else f"{RED}✗ (exit {rc}){NC}"
        print(f"  {name}: {status}")

    return 0 if all(rc == 0 for rc in results.values()) else 1


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Objective Drifting Attack — dispatches to v12/v13 attack scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  aggressive   Run v12 aggressive attack on a single victim
  subtle       Run v13 subtle persuasion on one or more victims
  all-api      Run all API-accessible victims (GPT-4o, Claude, Gemini)
  all          Run everything including local vLLM models

v12 Victims (--victim):
  gpt4o        GPT-4o via OpenAI API
  llama        Llama 3.1 8B Instruct via vLLM (port 8000)
  qwen         Qwen3-30B-A3B via vLLM (port 8001)

v13 Victims (--victims):
  claude       Claude 4.5 Sonnet via Anthropic API
  gemini       Gemini 3 Flash via Google API
  gpt          GPT-4o via OpenAI API

Examples:
  python Objective-Drifting.py --mode aggressive --victim gpt4o --num_tasks 100 --verbose
  python Objective-Drifting.py --mode subtle --victims claude gemini --num_tasks 50
  python Objective-Drifting.py --mode all-api --num_tasks 100
  python Objective-Drifting.py --setup-only
        """,
    )

    # Our flags
    parser.add_argument("--mode", type=str, default=None,
                        choices=["aggressive", "subtle", "all-api", "all"],
                        help="Attack mode")
    parser.add_argument("--victim", type=str, default=None,
                        choices=list(V12_VICTIMS.keys()),
                        help="Victim for v12 aggressive mode")
    parser.add_argument("--victims", nargs="+", default=None,
                        choices=V13_VICTIMS,
                        help="Victims for v13 subtle mode")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only set up WebShop environment")
    parser.add_argument("--check-env", action="store_true",
                        help="Check if WebShop environment is ready")
    parser.add_argument("--list", action="store_true",
                        help="List available victims and modes")

    # Everything else passes through to the attack script
    args, extra_args = parser.parse_known_args()

    # --list
    if args.list:
        print(f"\n{BOLD}Available Modes:{NC}")
        print(f"  aggressive   Single-victim aggressive goal override (v12)")
        print(f"  subtle       Multi-victim subtle persuasion (v13)")
        print(f"  all-api      All API victims")
        print(f"  all          All victims including vLLM\n")
        print(f"{BOLD}v12 Victims (--victim):{NC}")
        for k, v in V12_VICTIMS.items():
            vllm = " [requires vLLM]" if v.get("requires_vllm") else ""
            print(f"  {k:<12} {v['display']}{vllm}")
        print(f"\n{BOLD}v13 Victims (--victims):{NC}")
        for v in V13_VICTIMS:
            print(f"  {v}")
        print(f"\n{BOLD}Pass-through flags:{NC}")
        print(f"  --num_tasks N    Number of tasks")
        print(f"  --num_seeds N    Seeds per task")
        print(f"  --verbose        Verbose output")
        print(f"  --strategy X     Attack strategy")
        print(f"  (any other flags are forwarded to the attack script)")
        return

    # --check-env
    if args.check_env:
        if not check_conda():
            print(f"{RED}✗ conda not found{NC}"); sys.exit(1)
        if not conda_env_exists():
            print(f"{RED}✗ Conda env '{CONDA_ENV_NAME}' not found.{NC}")
            print(f"  Run: python Objective-Drifting.py --setup-only")
            sys.exit(1)
        if verify_env():
            print(f"{GREEN}✓ WebShop environment is ready.{NC}")
        else:
            print(f"{RED}✗ Verification failed.{NC}")
            sys.exit(1)
        return

    # --setup-only
    if args.setup_only:
        if not check_conda():
            print(f"{RED}conda required.{NC}"); sys.exit(1)
        if conda_env_exists() and verify_env():
            print(f"{GREEN}✓ Already set up and working.{NC}")
        else:
            if not setup_env(): sys.exit(1)
        return

    # Must have a mode
    if args.mode is None:
        parser.print_help()
        print(f"\n{YELLOW}Tip: Use --list to see all options.{NC}")
        return

    # Ensure WebShop env is ready before any attack
    print(f"{CYAN}{'='*60}")
    print(f" Objective Drifting Attack")
    print(f"{'='*60}{NC}\n")

    if not ensure_env_ready():
        print(f"\n{RED}Cannot proceed without WebShop environment.{NC}")
        print(f"Try: bash {SETUP_SCRIPT}")
        sys.exit(1)

    # Dispatch based on mode
    if args.mode == "aggressive":
        if not args.victim:
            print(f"{RED}--victim required for aggressive mode.{NC}")
            print(f"Options: {', '.join(V12_VICTIMS.keys())}")
            sys.exit(1)
        rc = run_v12_attack(args.victim, extra_args)
        sys.exit(rc)

    elif args.mode == "subtle":
        victims = args.victims or ["claude", "gemini"]
        rc = run_v13_attack(victims, extra_args)
        sys.exit(rc)

    elif args.mode == "all-api":
        rc = run_all_api(extra_args)
        sys.exit(rc)

    elif args.mode == "all":
        rc = run_all(extra_args)
        sys.exit(rc)


if __name__ == "__main__":
    main()