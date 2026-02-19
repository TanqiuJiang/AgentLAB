#!/bin/bash
# =============================================================================
# setup_and_test.sh — Task-Injection Track Setup & Verification
# 
# This script creates an isolated conda environment, installs the modified
# AgentDojo package, and runs verification tests to ensure everything works.
#
# Usage:
#   cd /path/to/AgentLAB/Task-Injection
#   bash setup_and_test.sh              # Setup + import tests only
#   bash setup_and_test.sh --run-test   # Setup + run a minimal live test
# =============================================================================

set -e  # Exit on any error

# ── Configuration ──
ENV_NAME="agentdojo"
PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTDOJO_DIR="${SCRIPT_DIR}/agentdojo"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step()  { echo -e "\n${BLUE}[STEP]${NC} $1"; }
print_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_fail()  { echo -e "${RED}[FAIL]${NC} $1"; }

# ── Parse arguments ──
RUN_LIVE_TEST=false
for arg in "$@"; do
    case $arg in
        --run-test) RUN_LIVE_TEST=true ;;
        *) echo "Unknown argument: $arg"; echo "Usage: bash setup_and_test.sh [--run-test]"; exit 1 ;;
    esac
done

# =============================================================================
# PHASE 1: Environment Setup
# =============================================================================
echo -e "\n${BLUE}=============================================${NC}"
echo -e "${BLUE}  Task-Injection Track: Setup & Verification ${NC}"
echo -e "${BLUE}=============================================${NC}"

# ── 1.1 Check that agentdojo directory exists ──
print_step "Checking agentdojo directory..."
if [ ! -d "$AGENTDOJO_DIR" ]; then
    print_fail "agentdojo directory not found at: $AGENTDOJO_DIR"
    echo "  Make sure this script is inside the Task-Injection/ directory"
    echo "  and the agentdojo/ subdirectory exists."
    exit 1
fi

if [ ! -f "$AGENTDOJO_DIR/pyproject.toml" ]; then
    print_fail "pyproject.toml not found in $AGENTDOJO_DIR"
    exit 1
fi
print_ok "agentdojo directory found with pyproject.toml"

# ── 1.2 Check conda is available ──
print_step "Checking conda..."
if ! command -v conda &> /dev/null; then
    print_fail "conda not found. Please install miniconda or anaconda first."
    exit 1
fi
print_ok "conda found: $(conda --version)"

# ── 1.3 Create or reuse conda environment ──
print_step "Setting up conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."

# Initialize conda for script usage
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME} "; then
    print_warn "Environment '${ENV_NAME}' already exists."
    echo "  To recreate from scratch, run: conda env remove -n ${ENV_NAME}"
    echo "  Activating existing environment..."
    conda activate "${ENV_NAME}"

    # Verify Python version
    CURRENT_PY=$(python --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ "$CURRENT_PY" != "$PYTHON_VERSION" ]]; then
        print_warn "Existing env uses Python ${CURRENT_PY}, expected ${PYTHON_VERSION}."
        echo "  Proceeding anyway — should be fine if >= 3.10."
    fi
else
    echo "  Creating new environment..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y -q
    conda activate "${ENV_NAME}"
    print_ok "Environment created and activated"
fi

echo "  Python: $(python --version)"
echo "  Location: $(which python)"

# ── 1.4 Install agentdojo as editable package ──
print_step "Installing agentdojo (editable) with transformers extra..."
cd "$AGENTDOJO_DIR"
pip install -e ".[transformers]" -q 2>&1 | tail -5
print_ok "agentdojo installed"

# =============================================================================
# PHASE 2: Verification Tests
# =============================================================================
echo -e "\n${BLUE}=============================================${NC}"
echo -e "${BLUE}  Running Verification Tests                  ${NC}"
echo -e "${BLUE}=============================================${NC}"

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    print_step "Test: ${test_name}"
    if eval "$test_cmd" 2>&1; then
        print_ok "PASSED: ${test_name}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_fail "FAILED: ${test_name}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# ── 2.1 Package import test ──
run_test "Import agentdojo core" \
    "python -c 'import agentdojo; print(f\"  agentdojo version: {agentdojo.__version__}\")'"

# ── 2.2 Import custom attack modules (with dummy API key) ──
run_test "Import search_attack_pipeline" \
    "OPENAI_API_KEY=sk-test python -c 'from agentdojo.attacks.search_attack_pipeline import main; print(\"  search_attack_pipeline: importable\")'"

run_test "Import long_horizon_attack" \
    "OPENAI_API_KEY=sk-test python -c 'from agentdojo.attacks.long_horizon_attack import build_initial_prompt; print(\"  long_horizon_attack: importable\")'"

run_test "Import long_horizon_rewrite" \
    "OPENAI_API_KEY=sk-test python -c 'from agentdojo.attacks.long_horizon_rewrite import build_rewrite_prompt; print(\"  long_horizon_rewrite: importable\")'"

# ── 2.3 CLI help test ──
run_test "search_attack_pipeline --help" \
    "OPENAI_API_KEY=sk-test python -m agentdojo.attacks.search_attack_pipeline --help > /dev/null"

run_test "benchmark --help" \
    "OPENAI_API_KEY=sk-test python -m agentdojo.scripts.benchmark --help > /dev/null"

# ── 2.4 Attack registry test ──
run_test "long_horizon attack registered" \
    "OPENAI_API_KEY=sk-test python -c '
from agentdojo.attacks.attack_registry import ATTACKS
assert \"long_horizon\" in ATTACKS, f\"long_horizon not in registry: {list(ATTACKS.keys())}\"
print(f\"  Registered attacks: {list(ATTACKS.keys())}\")
'"

# ── 2.5 Model enum test ──
run_test "LOCAL model enum exists" \
    "python -c '
from agentdojo.models import ModelsEnum
assert hasattr(ModelsEnum, \"LOCAL\"), \"LOCAL not in ModelsEnum\"
assert hasattr(ModelsEnum, \"VLLM_PARSED\"), \"VLLM_PARSED not in ModelsEnum\"
print(f\"  ModelsEnum.LOCAL = {ModelsEnum.LOCAL.value}\")
print(f\"  ModelsEnum.VLLM_PARSED = {ModelsEnum.VLLM_PARSED.value}\")
'"

# ── 2.6 Task suite loading test ──
run_test "Load banking task suite" \
    "python -c '
from agentdojo.task_suite.load_suites import get_suite
suite = get_suite(\"banking\")
user_tasks = list(suite.user_tasks.keys())
injection_tasks = list(suite.injection_tasks.keys())
print(f\"  Banking suite: {len(user_tasks)} user tasks, {len(injection_tasks)} injection tasks\")
'"

# ── 2.7 Pipeline config test ──
run_test "AgentPipeline.from_config with gpt-4o" \
    "OPENAI_API_KEY=sk-test python -c '
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
config = PipelineConfig(llm=\"gpt-4o-2024-05-13\")
pipeline = AgentPipeline.from_config(config)
print(f\"  Pipeline created: {type(pipeline).__name__}\")
'"

# ── 2.8 Defense availability test ──
run_test "Defense options available" \
    "OPENAI_API_KEY=sk-test python -c '
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
print(f\"  Available defenses: {list(DEFENSES.keys())}\")
assert \"repeat_user_prompt\" in DEFENSES, \"repeat_user_prompt defense missing\"
'"

# ── 2.9 Check for hardcoded paths ──
print_step "Test: No hardcoded personal paths in source"
HARDCODED=$(grep -rn "/data3/yuhui\|/home/tanjiang" src/agentdojo/attacks/*.py 2>/dev/null || true)
if [ -z "$HARDCODED" ]; then
    print_ok "PASSED: No hardcoded paths found"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    print_fail "FAILED: Hardcoded paths found:"
    echo "$HARDCODED"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# ── 2.10 Check no __pycache__ in source ──
print_step "Test: No __pycache__ directories in source"
PYCACHE_COUNT=$(find src/ -type d -name "__pycache__" | wc -l)
if [ "$PYCACHE_COUNT" -eq 0 ]; then
    print_ok "PASSED: No __pycache__ directories"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    print_fail "FAILED: Found ${PYCACHE_COUNT} __pycache__ directories"
    find src/ -type d -name "__pycache__"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# =============================================================================
# PHASE 3: Live Test (optional, requires API keys)
# =============================================================================
if [ "$RUN_LIVE_TEST" = true ]; then
    echo -e "\n${BLUE}=============================================${NC}"
    echo -e "${BLUE}  Live Test (requires API keys)               ${NC}"
    echo -e "${BLUE}=============================================${NC}"

    # Load .env if it exists
    if [ -f "$AGENTDOJO_DIR/.env" ]; then
        print_step "Loading .env file..."
        set -a
        source "$AGENTDOJO_DIR/.env"
        set +a
        print_ok ".env loaded"
    fi

    # Check for API key
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-test" ] || [ "$OPENAI_API_KEY" = "your-real-key-here" ]; then
        print_fail "Valid OPENAI_API_KEY not found."
        echo "  Set it in $AGENTDOJO_DIR/.env or export it before running."
        echo "  Skipping live test."
    else
        print_ok "OPENAI_API_KEY found"

        print_step "Running minimal live test (1 user task × 1 injection task, banking suite)..."
        echo "  This will make real API calls and may take a few minutes."
        echo ""

        python -m agentdojo.attacks.search_attack_pipeline \
            --suite banking \
            --user-task-id user_task_0 \
            --injection-task-id injection_task_0 \
            --max-workers 1 \
            --max-rewrites 1 \
            --agent-model-name gpt-4o \
            --attack-model-name gpt-4o

        if [ $? -eq 0 ]; then
            print_ok "PASSED: Live test completed successfully"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            print_fail "FAILED: Live test encountered errors"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    fi
fi

# =============================================================================
# PHASE 4: Summary
# =============================================================================
echo -e "\n${BLUE}=============================================${NC}"
echo -e "${BLUE}  Test Summary                                ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "  Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "  Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}All tests passed! Task-Injection track is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Add your real API keys to: ${AGENTDOJO_DIR}/.env"
    echo "  2. Run a live test:  bash setup_and_test.sh --run-test"
    echo "  3. Run full experiments using: ${AGENTDOJO_DIR}/scripts/run.sh"
    echo ""
    echo "Quick start:"
    echo "  conda activate ${ENV_NAME}"
    echo "  cd ${AGENTDOJO_DIR}"
    echo "  bash scripts/run.sh"
    exit 0
else
    echo -e "${RED}Some tests failed. Please fix the issues above before publishing.${NC}"
    exit 1
fi