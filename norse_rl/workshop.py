# Setup environment
from norse_rl import simulate
import gym
import torch

# Import neuron simulator
import norse.torch as norse
from norse_rl.util import Linear

def run(environment:str , model: torch.nn.Module, **kwargs):
    env_args = {}
    if 'image_scale' in kwargs:
        env_args['image_scale'] = kwargs['image_scale']
    env = gym.make(environment, **env_args)

    sim_args = {}
    if 'show_fps' in kwargs:
        sim_args['show_fps'] = kwargs['show_fps']
    simulation = simulate.Simulation(env, **sim_args)
    return simulation.run(model)