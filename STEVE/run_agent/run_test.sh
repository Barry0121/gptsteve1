#!/bin/bash
echo "Running " $1 "for " $2 "times. " "$3 frames each."
task_dir=$(pwd)/data/"$4"
mkdir -p "$task_dir"

# Change custom_text_prompt to whatever text prompt you want to generate a video
for ((i = 1; i <= $2; i++));
do
    test_dir="$task_dir/test$i"
    mkdir -p "$test_dir"
    echo "Test " $i
    python "$(pwd)/steve1/run_agent/run_agent.py" \
        --in_model "$(pwd)/data/weights/vpt/2x.model" \
        --in_weights "$(pwd)/data/weights/steve1/steve1.weights" \
        --prior_weights "$(pwd)/data/weights/steve1/steve1_prior.pt" \
        --text_cond_scale 6.0 \
        --visual_cond_scale 7.0 \
        --gameplay_length "$3" \
        --save_dirpath "$test_dir" \
        --task "$1" \
        --start_template_pth "$(pwd)/steve1/data/prompt/start_prompt.json" \
        --next_template_pth "$(pwd)/steve1/data/prompt/next_prompt.json" \
        --help_template_pth "$(pwd)/steve1/data/prompt/help_prompt.json" \
        --stuckpoint_pth "$(pwd)/steve1/data/prompt" \
        --help_gap 2000
        # --custom_text_prompt_pth /home/user/ltl_steve1/STEVE/steve1/data/prompts.txt \
        # --custom_conditions_pth /home/user/ltl_steve1/STEVE/steve1/data/conditions.txt; \
    echo "Sleep to prevent GPT rate limiter kicking in..."
    sleep 5
done
