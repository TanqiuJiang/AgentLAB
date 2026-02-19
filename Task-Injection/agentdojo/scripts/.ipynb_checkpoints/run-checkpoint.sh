

# example1: gpt-4o no defense

for i in {1..5}; do
  python -m agentdojo.attacks.search_attack_pipeline \
    --suite banking \
    --user-task-id user_task_0 user_task_1 user_task_2 user_task_3 user_task_4 user_task_5 user_task_6 user_task_7 user_task_8 user_task_9 user_task_10 user_task_11 user_task_12 user_task_13 user_task_14 user_task_15 \
    --injection-task-id injection_task_0 injection_task_1 injection_task_2 injection_task_3 injection_task_4 injection_task_5 injection_task_6 injection_task_7 injection_task_8 \
    --max-workers 2 \
    --max-rewrites 1 \
    --agent-model-name gpt-4o \
    --attack-model-name gpt-5.1 \
    --use-batch-api 
done

# example2: gpt-4o with defense

for i in {1..5}; do
  python -m agentdojo.attacks.search_attack_pipeline \
    --suite banking \
    --user-task-id user_task_0 user_task_1 user_task_2 user_task_3 user_task_4 user_task_5 user_task_6 user_task_7 user_task_8 user_task_9 user_task_10 user_task_11 user_task_12 user_task_13 user_task_14 user_task_15 \
    --injection-task-id injection_task_0 injection_task_1 injection_task_2 injection_task_3 injection_task_4 injection_task_5 injection_task_6 injection_task_7 injection_task_8 \
    --max-workers 2 \
    --max-rewrites 1 \
    --agent-model-name gpt-4o \
    --attack-model-name gpt-5.1 \
    --defense repeat_user_prompt \
    --use-batch-api 
done


# example3: local vllm servered model

for i in {1..5}; do
  python -m agentdojo.attacks.search_attack_pipeline \
    --suite banking \
    --user-task-id user_task_0 user_task_1 user_task_2 user_task_3 user_task_4 user_task_5 user_task_6 user_task_7 user_task_8 user_task_9 user_task_10 user_task_11 user_task_12 user_task_13 user_task_14 user_task_15 \
    --injection-task-id injection_task_0 injection_task_1 injection_task_2 injection_task_3 injection_task_4 injection_task_5 injection_task_6 injection_task_7 injection_task_8 \
    --max-workers 1 \
    --max-rewrites 1 \
    --agent-model-name local \
    --attack-model-name gpt-5-mini \
    --use-batch-api
done

# example4: reproduce the agent run results

python -m agentdojo.scripts.benchmark -s banking -ut user_task_0 -ut user_task_1 -ut user_task_2 -ut user_task_3 -ut user_task_4 -ut user_task_5 -ut user_task_6 -ut user_task_7 -ut user_task_8 -ut user_task_9 -ut user_task_10 -ut user_task_11 -ut user_task_12 -ut user_task_13 -ut user_task_14 -ut user_task_15 -it injection_task_0 -it injection_task_1 -it injection_task_2 -it injection_task_3 -it injection_task_4 -it injection_task_5 -it injection_task_6 -it injection_task_7 -it injection_task_8 --model LOCAL --model-id Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --attack long_horizon --defense repeat_user_prompt