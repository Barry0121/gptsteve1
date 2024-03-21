import os
import sys
from datetime import datetime

import cv2
import torch
from tqdm import tqdm
import argparse
from collections import OrderedDict

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent.paper_prompts import load_text_prompt_embeds, load_visual_prompt_embeds
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video

FPS = 30

def check_inventory(inventory: OrderedDict, obj: str, num: int) -> bool:
    if obj == "air":
        return True
    if '*' in obj:
        obj = obj.strip('*')
        candidate_num_list = [inventory.get(candidate) for candidate in inventory.keys() if obj in candidate]
        if num == 0 and obj != "air":
            return max(candidate_num_list) == num
        return max(candidate_num_list) >= num
    if num == 0 and obj != "air":
        return inventory.get(obj) == num
    return inventory.get(obj) >= num

# modify the prompt_embed into list, then set conditions to change prompt
def run_agent(prompt_embed_list: list, gameplay_length, save_video_filepath,
              in_model, in_weights, seed, cond_scale, condition_list):
    assert cond_scale is not None
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)

    # Make sure seed is set if specified
    obs = env.reset()
    if seed is not None:
        env.seed(seed)

    # Setup
    gameplay_frames = []
    prog_evaluator = ProgrammaticEvaluator(obs)
    # print(obs)

    prompt_idx = 0
    prompt_embed = prompt_embed_list[prompt_idx]
    obj, num = condition_list[prompt_idx]
    count = 0
    pick_up_state = False

    # Run agent in MineRL env
    # todo: act according to the prompt sequence, set termination for each prompt
    # condition format: (obj, num)
    for frame_idx in tqdm(range(gameplay_length)):
        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)

        obs, _, _, _ = env.step(minerl_action)
        count += 1

        if pick_up_state:
            if count == 150:
                print(f"Pick up finished! Return to task...")
                prompt_embed = prompt_embed_list[prompt_idx]
                obj, num = condition_list[prompt_idx]
                count = 0
                pick_up_state = False
        else:
            # if count == 1000:
            #     pick_up_state = True
            #     print("Try to pick up things....")
            #     mineclip = load_mineclip_wconfig()
            #     prior = load_vae_model(PRIOR_INFO)
            #     prompt_embed = get_prior_embed("pick up drops", mineclip, prior, DEVICE)
            #     count = 0

            if obj != "air" and check_inventory(obs['inventory'], obj, num):
                print(f"Condition {obj}, {num} satisfied! Next task...")
                prompt_idx += 1
                prompt_embed = prompt_embed_list[prompt_idx]
                obj, num = condition_list[prompt_idx]
                count = 0
    

        frame = obs['pov']
        # frame = cv2.resize(frame, (128, 128))
        gameplay_frames.append(frame)

        prog_evaluator.update(obs)

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)

    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()


def generate_text_prompt_videos(custom_prompt_embeds, condition_list, in_model, in_weights, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in custom_prompt_embeds.items():
        print(f'\nGenerating video for text prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'{datetime.now()}Text_Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, None, cond_scale, condition_list)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


def generate_visual_prompt_videos(prompt_embeds, in_model, in_weights, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in prompt_embeds.items():
        print(f'\nGenerating video for visual prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'{name} - Visual Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, None, cond_scale)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--text_cond_scale', type=float, default=6.0)
    parser.add_argument('--visual_cond_scale', type=float, default=7.0)
    parser.add_argument('--gameplay_length', type=int, default=2000)
    parser.add_argument('--save_dirpath', type=str, default='data/generated_videos/')
    parser.add_argument('--custom_text_prompt_pth', type=str, default=None)
    parser.add_argument('--custom_conditions_pth', type=str, default=None)
    args = parser.parse_args()

    if args.custom_text_prompt_pth is not None:
        # Generate a video for the text prompt
        prompt_embed_list = []
        condition_list = []
        with open(args.custom_text_prompt_pth) as prompt_f:
            for p in prompt_f.readlines():
                p = p.strip('\n')
                mineclip = load_mineclip_wconfig()
                prior = load_vae_model(PRIOR_INFO)
                prompt_embed_list.append(get_prior_embed(p, mineclip, prior, DEVICE))

        with open(args.custom_conditions_pth) as condition_f:
            for c in condition_f.readlines():
                c = c.strip('\n').split(' ')[:2]
                c[1] = int(c[1])
                condition_list.append(tuple(c))

        custom_prompt_embeds = {args.custom_text_prompt_pth: prompt_embed_list}
        generate_text_prompt_videos(custom_prompt_embeds, condition_list, args.in_model, args.in_weights, args.text_cond_scale,
                                    args.gameplay_length, args.save_dirpath)
    else:
        raise NotImplementedError
        # Generate videos for the text and visual prompts used in the paper
        text_prompt_embeds = load_text_prompt_embeds()
        visual_prompt_embeds = load_visual_prompt_embeds()
        generate_text_prompt_videos(text_prompt_embeds, args.in_model, args.in_weights, args.text_cond_scale,
                                    args.gameplay_length, args.save_dirpath)
        generate_visual_prompt_videos(visual_prompt_embeds, args.in_model, args.in_weights, args.visual_cond_scale,
                                      args.gameplay_length, args.save_dirpath)
        sys.exit(0)
