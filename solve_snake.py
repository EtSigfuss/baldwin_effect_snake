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

def get_q_values(W, b, features):
    """
    returns array of all actions specifically their q values for a given state and genome
    """
    return W @ features + b


def select_action(features, W, b, explore_rate=0.2):
    """
    if explore_rate is bigger than a random number between 0 and 1
    do a random choice.
    else do action with highest q value
    """
    #potentially have random choice and explor
    if np.random.rand() < explore_rate:
        return int(np.random.choice(ACTIONS))

    #get best perceived val
    q_values = get_q_values(W, b, features)

    return np.argmax(q_values)

def train_linear_q(W,b, features, action, reward, new_features, done, learn_rate, discount):
    """
    modify weight and bias by error between:
    - the reward gained + g*max Q of resultant state
    - diff between acted on q val of current state
    I'm told this is temporal difference which makes sense as its the difference between
    what we actually got plus the dicount of what we can now get, and what we thought we'd get
    """
    q_current = get_q_values(W,b,features)[action]

    if done:
        target = reward
    else:
        q_next = get_q_values(W,b,new_features)
        target = reward + discount*np.max(q_next)
    
    temp_diff_error = target - q_current

    W[action] += learn_rate * temp_diff_error*features
    b[action] += learn_rate * temp_diff_error


def run_episode(env, solution, explore_rate=0, discount=0, learn_rate = 0,learn=False, render=False, agent:Agent = None):
    """
    runs a genome through an episode(a single game of snake)
    if learning is enabled, weighn and bias are modified via temporal diference
    and RL params are used
    rendering can be toggled here
    """
    W, b= decode_chromosome(solution)
    state = env.reset()
    total_reward = 0.0
    if not learn:
        explore_rate = 0

    while True:
        features = featurize_state(state)
        action = agent.get_action(features)

        new_state, reward, done, info = env.step(action)
        total_reward += float(reward)

        if learn and agent:
            new_features = featurize_state(new_state)
            agent.train(old_state=state, action=action, reward=reward, new_features=new_state, done=done)
        if render:
            env.render()

        state = new_state
        if done:
            break
    return total_reward, int(env.score)
