# Change custom_text_prompt to whatever text prompt you want to generate a video for
xvfb-run python steve1/run_agent/run_agent.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/weights/steve1/steve1.weights \
--prior_weights data/weights/steve1/steve1_prior.pt \
--text_cond_scale 6.0 \
--visual_cond_scale 7.0 \
--gameplay_length 6000 \
--save_dirpath data/generated_videos/custom_text_prompt \
--custom_text_prompt_pth /home/user/ltl_steve1/STEVE-1/steve1/data/prompts.txt \
--custom_conditions_pth /home/user/ltl_steve1/STEVE-1/steve1/data/conditions.txt