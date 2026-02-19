#!/bin/bash

# screen -dmS vllm bash -c "bash /data3/yuhui/9/agentdojo/scripts/vllm_llama.sh"
# --- Lists for Loops ---
# Add your model and short name pairs here, separated by a comma.
# --- Server Parameters ---
HOST="localhost"
PORT="8000"

# --- Hardware Configuration ---
export CUDA_VISIBLE_DEVICES="0,1,2,3"
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
DTYPE=bfloat16

#==============================================================================
#                             HELPER FUNCTIONS
#==============================================================================

# --- Function to start the VLLM server ---
# Takes the model path and an optional served model name as arguments.
start_vllm_server() {
    local model_path=$1
    local served_name_arg=""
    if [ -n "$2" ]; then
        served_name_arg="--served-model-name $2"
    fi

    echo "Starting VLLM OpenAI API server in the background for model: $model_path"
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      $served_name_arg \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --port $PORT \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --disable-log-requests \
      --disable-log-stats \
      --dtype $DTYPE \
      --enable-auto-tool-choice \
      --tool-call-parser pythonic
}


start_vllm_server "meta-llama/Llama-3.3-70B-Instruct" >> /data3/yuhui/9/agentdojo/log/vllm_$(date +%Y%m%d_%H%M).log 2>&1
