import random

import numpy as np
import pygad
import torch
import torch.nn as nn
import torch.optim as optim

from snake import Snake
from solve_snake import featurize_state
from torch_logic import Agent

#enviroment parames
height = 10
width = 10
lifespan = 5000

# GA hyperparams
pop = 50
generations = 50
mutation_rate = 0.10
mutation_scale = 0.4

#Torch params
episodes_per_life = 10 #***
learn_rate = 0.01
learn_rate_floor = 0.01
learn_rate_decay = (learn_rate_floor/learn_rate)**(1/episodes_per_life)
explore_rate = 0.2
explore_rate_floor = 0.001
explore_rate_decay_generation = (explore_rate_floor/explore_rate)**(1/generations)
explore_rate_decay_episode = (explore_rate_floor/explore_rate)**(1/episodes_per_life)
discount = 0.95

seed_value = 1234
rng = np.random.default_rng(seed_value)#global rng


agent = Agent()
env = Snake(
        width=width,
        height=height,
        lifespan=lifespan,
        state_includes_location=False,
        state_includes_sensory=True,
        render_mode="pygame",
        seed=int(rng.integers(1,1000000)),
        )

state = env.reset()
total_reward = 0.0

#test function that you ctrl c out of, infinitely trains a snake agent with 254 hidden layers
while True:
    while True:
        features = featurize_state(state)
        action = agent.get_action(features)

        new_state, reward, done, info = env.step(action)
        total_reward += float(reward)

        new_features = featurize_state(new_state)
        agent.train(old_state=state, action=action, reward=reward, new_state=new_state, done=done)

        env.render()

        state = new_state
        if done:
            break
    state = env.reset()
    
