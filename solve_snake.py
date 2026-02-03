import numpy as np

from constants import SNAKE_FEATURE_COLS
from torch_logic import Agent


def featurize_state(state):
    """
    gives an ordered array of floats of the desired features
    from a state dict
    """
    return np.array(
        [state.get(key, 0.0) for key in SNAKE_FEATURE_COLS]
    )


def decode_chromosome(chrom):
    """unfolds chromosome into lists of weights and bias indexed by actions"""
    n_features = len(SNAKE_FEATURE_COLS)
    W = chrom[: 4 * n_features].reshape(4, n_features)
    b = chrom[4 * n_features: 4 * n_features + 4]
    return W, b









def run_episode(agent:Agent, env, learn=False, render=False,):
    """
    
    """

    state = env.reset()
    total_reward = 0.0

    while True:
        features = featurize_state(state)
        action = agent.get_action(features)

        new_state, reward, done, info = env.step(action)
        total_reward += float(reward)

        if learn:
            agent.train(old_state=state, action=action, reward=reward, new_state=new_state, done=done)
        if render:
            env.render()

        state = new_state
        if done:
            break
    return int(env.score)
