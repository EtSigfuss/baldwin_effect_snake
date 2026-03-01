import os
import warnings

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import sys

import numpy as np
import pygad

from agent_interface import featurize_state, run_episode
from constants import ACTION_STRING, ACTIONS, SNAKE_FEATURE_COLS
from snake import Snake
from torch_logic import Agent

seed_value = 1234
rng = np.random.default_rng(seed_value)#global rng

#enviroment parames
height = 10
width = 10
lifespan = 20000

#NN params
episodes_per_life = 10
input_size = len(SNAKE_FEATURE_COLS)
hidden_size = 2**4
output_size = len(ACTIONS)

# GA hyperparams
pop = 50
generations = 50
mutation_rate = 0.10
mutation_scale = 0.10
gene_size = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size

#observation settings
fitness_eval_count = 0
generation_high_score = 0
learned_weights_store = {}

#trial and arch runs
arch_runs = 5
trial_amt = 10


def fitness_func_learning(ga_instance, solution, solution_idx):
    """
    An attempt at using deep q learning via torch for running snake

    """
    global learned_weights_store
    current_gen = ga_instance.generations_completed


    # print(f"generation = {current_gen} solution = {solution_idx}                ", end="\r")

    render_mode_life = None
    render_life = False


    agent = Agent(solution=solution,
                  input_size=input_size,
                  hidden_size=hidden_size,
                  output_size = output_size,
                  decay_actions= 100,
                  )

    total_score = 0

    env = Snake(
        width=width,
        height=height,
        lifespan=lifespan,
        state_includes_sensory=True,
        render_mode=render_mode_life,
        seed=int(rng.integers(1,1000000)),
        frame_rate=60
        )
    for _ in range(episodes_per_life):
        score = run_episode(env=env,
                                learn=True,
                                render=render_life,
                                agent=agent
                                )
        total_score += score
        env.reset()
    env.close()


    #capture all of first gen and in dict with total score agent and genes

    learned_weights_store[current_gen,solution_idx] = (score,agent,solution)
        
    

    return total_score

def extract_best_agent_from_gen(gen):
    gen_data = {k: v for k, v in learned_weights_store.items() if k[0] == gen}
    best_gen_key = max(gen_data, key = lambda k: gen_data[k][0])
    return  gen_data[best_gen_key]


if __name__ == "__main__":
    print("seed_value: ", seed_value)
    print("height: ", height)
    print("width: ",width)
    print()
    print("episodes_per_life: ", episodes_per_life)
    print("input_size: ", input_size)
    print("hidden size: ", hidden_size)
    print("output_size: ",output_size)
    print()
    print("pop: ", pop)
    print("generations: ", generations)
    print("mutation rate: ",mutation_rate)
    print("mutation scale: ", mutation_scale)
    print("gene size: ",gene_size)


    bald_ga = pygad.GA(num_generations=generations,
                sol_per_pop= pop,
                num_genes= gene_size,
                keep_elitism=3,
                num_parents_mating=5,
                mutation_percent_genes= mutation_scale,
                mutation_probability= mutation_rate,
                mutation_type='random',
                fitness_func=fitness_func_learning,
                # parallel_processing=['process', 4]
                )

    bald_total = 0




    bald_ga.run()

    bald_solution, solution_fitness, solution_idx = bald_ga.best_solution()

    print("\ndone bald")
    


    test_states = [
    # 1. Food medium distance ahead, clear path
    {
        "food_front/back_norm": 0.5, 
        "food_left/right_norm": 0.0, 
        "obstacle_front": 0.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 0.0
    },
    # 2. Food to the right, blocked by wall/body in front
    {
        "food_front/back_norm": 0.0, 
        "food_left/right_norm": 0.2, 
        "obstacle_front": 1.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 0.0
    },
    # 3. Trapped Front and Right, must turn Left to survive
    {
        "food_front/back_norm": -0.3, 
        "food_left/right_norm": -0.1, 
        "obstacle_front": 1.0, 
        "obstacle_right": 1.0, 
        "obstacle_left": 0.0
    },
    # 4. Food is far behind and to the left 
    {
        "food_front/back_norm": -0.75, 
        "food_left/right_norm": -0.5, 
        "obstacle_front": 0.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 0.0
    },
    # 5. Food is close ahead, obstacle immediately to the left
    {
        "food_front/back_norm": 0.05, 
        "food_left/right_norm": 0.0, 
        "obstacle_front": 0.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 1.0
    }
]


    for gen in range(generations):
        score, agent, genes = extract_best_agent_from_gen(gen)
        agent_instinct = Agent(solution=genes,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size = output_size,
                        decay_actions= trial_amt)
        
        agent_actions = []
        agent_instinct_actions = []
        for state in test_states:
            agent_actions.append(ACTION_STRING[agent.get_action(featurize_state(state),False)])
            agent_instinct_actions.append(ACTION_STRING[agent_instinct.get_action(featurize_state(state),False)])
        
        print(f"Generation {gen}")
        print(f"intinct actions {agent_instinct_actions}")
        print(f"learned actions {agent_actions}")
        print(f"learned score = {score}")
        