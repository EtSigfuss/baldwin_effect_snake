import random

import numpy as np
import pygad
import torch
import torch.nn as nn
import torch.optim as optim

from agent_interface import featurize_state
from constants import ACTION_STRING, ACTIONS, SNAKE_FEATURE_COLS
from snake import Snake
from torch_logic import Agent

#enviroment parames
height = 10
width = 10
lifespan = 5000
frame_rate = 200


#Torch params
input_size = len(SNAKE_FEATURE_COLS)
hidden_size = 2**3
output_size = len(ACTIONS)
starting_epsilon = 1


seed_value = 1234
rng = np.random.default_rng(seed_value)#global rng


agent = Agent(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size = output_size,
                        decay_actions= 30000,
                        epsilon=starting_epsilon,)
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
    state = env.reset()
    while True:
        features = featurize_state(state)
        action = agent.get_action(features)

        new_state, reward, done, info = env.step(action)
        total_reward += float(reward)

        agent.train(old_state=state, action=action, reward=reward, new_state=new_state, done=done)

        env.render(epsilon=agent.epsilon)

        state = new_state
        if done:
            break
