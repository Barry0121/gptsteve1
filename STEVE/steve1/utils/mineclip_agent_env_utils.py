import pickle

import gym
import torch

from steve1.MineRLConditionalAgent import MineRLConditionalAgent
from steve1.VPT.agent import ENV_KWARGS
from steve1.config import MINECLIP_CONFIG, DEVICE
from steve1.mineclip_code.load_mineclip import load
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from minerl.herobraine.hero.handlers.translation import TranslationHandler

from minerl.herobraine.hero import handlers as H, mc
from minerl.herobraine.hero.mc import SIMPLE_KEYBOARD_ACTION

from typing import List

KEYMAP = {
    '17': 'forward',
    '30': 'left',
    '31': 'back',
    '32': 'right',
    '57': 'jump',
    '18': 'inventory',
    '21': 'drop',
    '42': 'sneak',
    '29': 'sprint',
    '-100': 'attack',  # BUTTON0 Left Click
    '-99': 'use',  # BUTTON1 Right Click
    '-98': 'pickItem',  # BUTTON2 Middle Click
    # '20': 'chat',  # This and following not currently in use
    # '33': 'swapHands',
    # '15': 'playerlist',  # Show player list gui
    # '53': 'command',  # Start typing server cmd
    # '60': 'screenshot',
    # '63': 'togglePerspective',
    # '87': 'fullscreen',
    # '46': 'saveToolbarActivator',
    # '45': 'loadToolbarActivator',
    # '38': 'advancements',
}

KEYMAP.update({str(x + 1): str(x) for x in range(1, 10)})

# TODO: add all other keys.
INVERSE_KEYMAP = {
    KEYMAP[key]: key for key in KEYMAP
}

class HumanSurvival_nodrop(HumanSurvival):
    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return [
            H.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items()
            if k in SIMPLE_KEYBOARD_ACTION
        ] + [
            H.CameraAction()
        ]

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_mineclip_wconfig():
    print('Loading MineClip...')
    return load(MINECLIP_CONFIG, device=DEVICE)


def make_env(seed):
    print('Loading MineRL...')
    env = HumanSurvival(**ENV_KWARGS).make()
    print('Starting new env...')
    env.reset()
    if seed is not None:
        print(f'Setting seed to {seed}...')
        env.seed(seed)
    return env


def make_agent(in_model, in_weights, cond_scale):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(env, device='cuda', policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent


def load_mineclip_agent_env(in_model, in_weights, seed, cond_scale):
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed)
    return agent, mineclip, env
