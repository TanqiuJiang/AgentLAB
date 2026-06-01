#!/usr/bin/env bash
# envs/webshop/setup_webshop.sh
# ---------------------------------------------------------------------------
# Builds the isolated `webshop` conda env for Objective-Drifting.py by
# REPRODUCING the working environment (Python 3.10) captured in
# webshop_requirements.txt, which is a pip freeze of that env.
#
# Why --no-deps: that env holds both the WebShop stack and the AgentLab runner
# stack in one place and pins pydantic 1.9 next to packages whose metadata wants
# pydantic 2 (google-genai; Gemini is offloaded to a separate gemini_env). A
# normal `pip install -r` fails the resolver on that conflict. --no-deps installs
# the exact pinned set, exactly as the working env has it.
#
# Contract (do not change):
#   bash setup_webshop.sh            full setup; exit 0 on success
#   bash setup_webshop.sh --check    exit 0 IFF the env is ready
#   conda env name:                  webshop
#
# Data: WEBSHOP_DATA=small (default, 1,000 products) or =all. For the full
# 12,087-instruction set you must ALSO edit web_agent_site/utils.py (upstream
# README step 6) to point at the full items_shuffle.json / items_ins_v2.json.

set -o pipefail

ENV_NAME="webshop"
PY_VERSION="${WEBSHOP_PY:-3.10}"        # your env was 3.10.19; pin exactly with WEBSHOP_PY=3.10.19
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/webshop_requirements.txt"
DATA_DIR="$SCRIPT_DIR/data"
SEARCH_DIR="$SCRIPT_DIR/search_engine"
DATA_SPLIT="${WEBSHOP_DATA:-small}"
SPACY_MODEL="en_core_web_sm"            # this env uses sm 3.4.0 (pinned in requirements)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
say()  { echo -e "${CYAN}[webshop-setup]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[fail]${NC} $*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
eval "$(conda shell.bash hook)" || die "could not initialize conda shell hook"
env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

# ------------------------------- --check -----------------------------------
if [[ "${1:-}" == "--check" ]]; then
    env_exists || die "conda env '$ENV_NAME' does not exist (run without --check first)"
    conda activate "$ENV_NAME" || die "could not activate '$ENV_NAME'"
    python -c "import gym, web_agent_site" 2>/dev/null              || die "web_agent_site not importable in '$ENV_NAME'"
    python -c "import pyserini, spacy, transformers" 2>/dev/null    || die "core deps missing in '$ENV_NAME'"
    python -c "import spacy; spacy.load('$SPACY_MODEL')" 2>/dev/null || die "$SPACY_MODEL not installed/loadable"
    ls "$DATA_DIR"/items_*.json >/dev/null 2>&1                     || die "product/instruction data missing in $DATA_DIR"
    find "$SEARCH_DIR/indexes" -mindepth 1 >/dev/null 2>&1           || die "search index not built in $SEARCH_DIR/indexes"
    ok "webshop environment is ready"
    exit 0
fi

# ------------------------------- full setup --------------------------------
[[ -f "$REQ_FILE" ]] || die "requirements file not found: $REQ_FILE"

if env_exists; then
    ok "conda env '$ENV_NAME' already exists"
else
    say "creating conda env '$ENV_NAME' (python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" || die "env creation failed"
fi
conda activate "$ENV_NAME" || die "could not activate '$ENV_NAME'"

# 1. conda-only packages (NOT in the pip requirements)
say "installing faiss-cpu + JDK 11 via conda"
conda install -y -c pytorch faiss-cpu || warn "faiss-cpu via conda failed; sparse search may still work"
conda install -y -c conda-forge openjdk=11 || die "openjdk install failed (pyserini needs a JVM)"

# 2. python deps, reproduced exactly with --no-deps
say "installing python requirements with --no-deps (reproducing the frozen env)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install Cython==3.2.2 pybind11==3.0.1     # build deps, in case nmslib compiles
python -m pip install --no-deps -r "$REQ_FILE" || die "pip install failed (see above).
    If nmslib failed to build, ensure a C++ compiler is present (e.g. apt install build-essential),
    then: pip install --no-build-isolation --no-deps nmslib==2.1.2"

# 3. data (gdown IDs from your setup.sh; skip if present)
mkdir -p "$DATA_DIR"
if ls "$DATA_DIR"/items_shuffle*.json >/dev/null 2>&1; then
    ok "data already present in $DATA_DIR; skipping download"
else
    say "downloading WebShop data (split=$DATA_SPLIT) into $DATA_DIR"
    (
        cd "$DATA_DIR" || exit 1
        if [[ "$DATA_SPLIT" == "small" ]]; then
            gdown 'https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib'  # items_shuffle_1000
            gdown 'https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu'  # items_ins_v2_1000
        elif [[ "$DATA_SPLIT" == "all" ]]; then
            gdown 'https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB'  # items_shuffle (full)
            gdown 'https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi'  # items_ins_v2  (full)
        else
            die "WEBSHOP_DATA must be 'small' or 'all' (got '$DATA_SPLIT')"
        fi
        gdown 'https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O'      # items_human_ins (goals)
    ) || die "data download failed"
fi

# 4. search index (your pipeline: convert_product_file_format.py -> run_indexing.sh)
#    convert writes documents.jsonl into resources*/ (which must exist first),
#    run_indexing.sh builds indexes/ indexes_100/ indexes_1k/ indexes_100k/.
if find "$SEARCH_DIR/indexes" -mindepth 1 >/dev/null 2>&1; then
    ok "search index already present; skipping build"
else
    [[ -f "$SEARCH_DIR/convert_product_file_format.py" ]] || die "missing convert_product_file_format.py"
    [[ -f "$SEARCH_DIR/run_indexing.sh" ]] || die "missing run_indexing.sh"
    say "building search index (slow step)"
    (
        cd "$SEARCH_DIR" || exit 1
        mkdir -p resources resources_100 resources_1k resources_100k
        python convert_product_file_format.py || exit 1
        bash run_indexing.sh || exit 1
    ) || die "index build failed"
fi

ok "setup complete. verify with:  bash \"$SCRIPT_DIR/setup_webshop.sh\" --check"