# Change custom_text_prompt to whatever text prompt you want to generate a video
for i in {1..5}; do \
xvfb-run python steve1/run_agent/run_agent.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/weights/steve1/steve1.weights \
--prior_weights data/weights/steve1/steve1_prior.pt \
--text_cond_scale 6.0 \
--visual_cond_scale 7.0 \
--gameplay_length 10000 \
--save_dirpath data/generated_videos/custom_text_prompt \
--task collect_stone \
--start_template_pth /home/user/ltl_steve1/STEVE/steve1/data/prompt/start_prompt.json \
--next_template_pth /home/user/ltl_steve1/STEVE/steve1/data/prompt/next_prompt.json \
--help_template_pth /home/user/ltl_steve1/STEVE/steve1/data/prompt/help_prompt.json \
--stuckpoint_pth /home/user/ltl_steve1/STEVE/steve1/data/prompt \
--help_gap 2000
# --custom_text_prompt_pth /home/user/ltl_steve1/STEVE/steve1/data/prompts.txt \
# --custom_conditions_pth /home/user/ltl_steve1/STEVE/steve1/data/conditions.txt; \
done
