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
height = 30
width = 30
lifespan = 5000
frame_rate = 500

# GA hyperparams
pop = 50
generations = 50
mutation_rate = 0.10
mutation_scale = 0.4

#Torch params


seed_value = 1234
rng = np.random.default_rng(seed_value)#global rng


agent = Agent(episodes= 500)
env = Snake(
        width=width,
        height=height,
        lifespan=lifespan,
        state_includes_location=False,
        state_includes_sensory=True,
        render_mode="pygame",
        seed=int(rng.integers(1,1000000)),
        frame_rate=frame_rate,
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

        agent.train(old_state=state, action=action, reward=reward, new_state=new_state, done=done)

        env.render()

        state = new_state
        if done:
            break
    state = env.reset()
    
