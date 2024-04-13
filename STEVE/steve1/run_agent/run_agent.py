import os
import sys
import time
from dotenv import load_dotenv
from datetime import datetime

import cv2
import json
import torch
from tqdm import tqdm
import argparse
from collections import OrderedDict
import base64
import requests

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent.paper_prompts import load_text_prompt_embeds, load_visual_prompt_embeds
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video
import openai

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Dialog:
    def __init__(self):
        self.log = []
    def add_log(self, role, content):
        self.log.append({"role": role, "content": content})
    def save_log(self, path):
        with open(path, "w+") as f:
            json.dump(self.log, f)


class ChatApp:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
            }
        self.messages = [
            {"role": "system", "content": [{"type":"text", "text": "You are an Minecraft expert and will help me solve tasks in Minecraft."}]},
        ]

    def chat(self, message):
        message = [{"type":"text", "text": message}]
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=self.messages,
            headers = self.headers
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]["content"]

    def process_response(self, response):
        task, condition = None, None
        try:
            task, condition = tuple(response.split('\n'))
            task = task.strip('\n').split(':')[1]
            condition = condition.strip('\n').split(':')[1].strip(' ')
            condition = condition.split(' ')[:2]
            condition[1] = int(condition[1])
            condition = tuple(condition)
        except:
            print(f"Invalid instruction: {response}")
            task = "explore as far as possible"
            condition = ("air", 0)

        print(f"Get task:{task}, stop at {condition}")
        logger.add_log("assistant", response)
        return task, condition

    def create_task(self, args, obs = None, mode = "start", base64_image = None):
        if mode == "start":
            with open(args.start_template_pth) as tpl_f:
                template = json.load(tpl_f)
        elif mode == "next":
            with open(args.next_template_pth) as tpl_f:
                template = json.load(tpl_f)
        if mode == "help":
            with open(args.help_template_pth) as tpl_f:
                template = json.load(tpl_f)
                return self.help_task(template, base64_image, obs)

        prompt = template["content"]
        prompt = prompt.replace("*TASK*", args.task)
        if not obs:
            prompt = prompt.replace("*INVENTORY*", "nothing")
        else:
            inventory = obs["inventory"]
            item_num_list = [(inventory.get(candidate), candidate) for candidate in inventory.keys() if inventory.get(candidate)]
            if item_num_list:
                item_str = ','.join([str(t[0]) + ' ' + str(t[1]) for t in item_num_list])
                prompt = prompt.replace("*INVENTORY*", item_str)
            else:
                prompt = prompt.replace("*INVENTORY*", "nothing")
        response = self.chat(prompt)
        return self.process_response(response)

    def help_task(self, template, base64_image, obs):
        time.sleep(60) # Sleep for about a minute to avoid GPT4 RateLimit
        prompt = template["content"]
        prompt = prompt.replace("*TASK*", args.task)
        inventory = obs["inventory"]
        item_num_list = [(inventory.get(candidate), candidate) for candidate in inventory.keys() if inventory.get(candidate)]
        if item_num_list:
            item_str = ','.join([str(t[0]) + ' ' + str(t[1]) for t in item_num_list])
            prompt = prompt.replace("*INVENTORY*", item_str)
        else:
            prompt = prompt.replace("*INVENTORY*", "nothing")

        # print(prompt)
        content_txt = {
                "type": "text",
                "text": prompt
            }
        content_img = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        self.messages.append({"role": "user", "content": [content_img]})
        self.messages.append({"role": "user", "content": [content_txt]})
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=self.messages,
            headers = self.headers
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        response = response["choices"][0]["message"]["content"]

        return self.process_response(response)


FPS = 30

def check_inventory(inventory: OrderedDict, obj: str, num: int) -> bool:
    if obj == "air":
        return True

    if obj not in inventory.keys():
        obj = '*' + obj

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
def run_agent(prompt_embed, gameplay_length, save_video_filepath,
              in_model, in_weights, seed, cond_scale, condition, bot: ChatApp, args, stv1):
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

    count = 0
    # prompt_embed = prompt_embed_list[prompt_idx]
    obj, num = condition
    mineclip, prior = stv1

    for frame_idx in tqdm(range(gameplay_length)):

        count += 1
        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)

        obs, _, _, _ = env.step(minerl_action)


        # satisfy inventory condition
        if obj != "air" and check_inventory(obs['inventory'], obj, num):
            print(f"Condition {obj}, {num} satisfied! Next task...")
            logger.add_log("agent", f"Condition {obj}, {num} satisfied! Next task...")
            task, condition = bot.create_task(args, obs, "next")
            prompt_embed = get_prior_embed(task, mineclip, prior, DEVICE)
            obj, num = condition
            count = 0

        frame = obs['pov']

        if count == args.help_gap:
            print("Got stuck! Asking for help...")
            logger.add_log("agent", "Got stuck! Asking for help...")
            pic_dir = os.path.join(args.stuckpoint_pth, "current.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(pic_dir, frame)
            base64_image = encode_image(pic_dir)
            task, condition = bot.create_task(args, obs, "help", base64_image)
            prompt_embed = get_prior_embed(task, mineclip, prior, DEVICE)
            obj, num = condition
            count = 0

        # frame = cv2.resize(frame, (128, 128))
        gameplay_frames.append(frame)

        prog_evaluator.update(obs)

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)
    inventory = obs["inventory"]
    item_num_list = [(inventory.get(candidate), candidate) for candidate in inventory.keys() if inventory.get(candidate)]
    with open(os.path.join(args.save_dirpath, f"{datetime.now()}result.txt"), "w+") as fp:
        fp.writelines([str(item_num_list)])
    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()
    logger.save_log(os.path.join(args.save_dirpath, f"{datetime.now()}dialog.json"))


def generate_text_prompt_videos(custom_prompt_embeds, condition_list, in_model, in_weights, cond_scale, gameplay_length, save_dirpath, bot, args, stv1):
    for name, prompt_embed in custom_prompt_embeds.items():
        print(f'\nGenerating video for text prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'{datetime.now()}Text_Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, None, cond_scale, condition_list, bot, args, stv1)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


def generate_visual_prompt_videos(prompt_embeds, in_model, in_weights, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in prompt_embeds.items():
        print(f'\nGenerating video for visual prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'{name} - Visual Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, None, cond_scale, stv1)
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
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--start_template_pth', type=str, default=None)
    parser.add_argument('--next_template_pth', type=str, default=None)
    parser.add_argument('--help_template_pth', type=str, default=None)
    parser.add_argument('--stuckpoint_pth', type=str, default=None)
    parser.add_argument('--help_gap', type=int, default=2000)
    args = parser.parse_args()

    bot = ChatApp()
    logger = Dialog()
    task, condition = bot.create_task(args)

    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(PRIOR_INFO)

    stv1 = (mineclip, prior)
    prompt_embed = get_prior_embed(task, mineclip, prior, DEVICE)


    # prompt_embed_list = []
    # condition_list = []
    # with open(args.custom_text_prompt_pth) as prompt_f:
    #     for p in prompt_f.readlines():
    #         p = p.strip('\n')
    #         mineclip = load_mineclip_wconfig()
    #         prior = load_vae_model(PRIOR_INFO)
    #         prompt_embed_list.append(get_prior_embed(p, mineclip, prior, DEVICE))

    # with open(args.custom_conditions_pth) as condition_f:
    #     try:
    #         for c in condition_f.readlines():
    #             c = c.strip('\n').split(' ')[:2]
    #             c[1] = int(c[1])
    #             condition_list.append(tuple(c))
    #     except:
    #         condition_list

    custom_prompt_embeds = {args.start_template_pth: prompt_embed}
    generate_text_prompt_videos(custom_prompt_embeds, condition, args.in_model, args.in_weights, args.text_cond_scale,
                                args.gameplay_length, args.save_dirpath, bot, args, stv1)
