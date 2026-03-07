import os
import warnings

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import statistics
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
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
num_parents_mating = int(pop*.30)
generations = 2
mutation_probability = 0.05
gene_size = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size

#observation settings
fitness_eval_count = 0
generation_high_score = 0
starting_and_learned_weights_store = defaultdict(list)



def fitness_func_learning(ga_instance, solution, solution_idx):
    """

    """
    global starting_and_learned_weights_store
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


    #capture all agents and starting weights in a gen index dict 

    starting_and_learned_weights_store[current_gen].append((score,agent,solution))
        
    

    return total_score

def extract_best_agent_from_gen(gen):
    gen_data = {k: v for k, v in starting_and_learned_weights_store.items() if k[0] == gen}
    best_gen_key = max(gen_data, key = lambda k: gen_data[k][0])
    return  gen_data[best_gen_key]

def get_median_gen_learned_n_instinct_action(gen: int, state: dict[str, float])  :
    """
    gets the median actions for a game state
    """
    global starting_and_learned_weights_store

    #list of actions of all agents of a generation for a state
    agent_final_actions = []
    agent_instinct_actions = []
    #iterate through all instinct and final agents of a generation and get the median actions
    scores = []
    for score, final_agent, solution in starting_and_learned_weights_store[gen]:

        agent_instinct = Agent(solution=solution,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size = output_size,
                        )
        
        
        
        agent_final_actions.append(final_agent.get_action(featurize_state(state),False))
        agent_instinct_actions.append(agent_instinct.get_action(featurize_state(state),False))
        scores.append(score)
        
    instinct_median_action = statistics.median(agent_instinct_actions)
    final_median_action = statistics.median(agent_final_actions)
    mean_score = statistics.mean(scores)

    return mean_score, instinct_median_action, final_median_action



if __name__ == "__main__":
    results = []

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
    print("parents mating: ",num_parents_mating)
    print("generations: ", generations)
    print("mutation probability: ",mutation_probability)
    print("gene size: ",gene_size)


    bald_ga = pygad.GA(num_generations=generations,
                sol_per_pop= pop,
                num_genes= gene_size,
                keep_elitism=3,
                num_parents_mating=num_parents_mating,
                mutation_probability= mutation_probability,
                mutation_type='random',
                fitness_func=fitness_func_learning,
                # parallel_processing=['process', 4]
                crossover_type="uniform"
                )

    bald_total = 0




    bald_ga.run()

    bald_solution, solution_fitness, solution_idx = bald_ga.best_solution()

    print("\ndone bald")
    


    test_states = [
    # 1. Food medium distance ahead, clear path
    {
        "food_front/back_norm": 0.5, 
        "food_right/left_norm": 0.0, 
        "obstacle_front": 0.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 0.0
    },
    # 2. Food to the right, blocked in front
    {
        "food_front/back_norm": 0.0,
        "food_right/left_norm": 0.2,
        "obstacle_front": 1.0,
        "obstacle_right": 0.0,
        "obstacle_left": 0.0
    },
    # 3. Trapped Front and Right, must turn Left to survive, food is behind and right
    {
        "food_front/back_norm": -0.3,
        "food_right/left_norm": -0.1,
        "obstacle_front": 1.0,
        "obstacle_right": 1.0,
        "obstacle_left": 0.0
    },
    # 4. Food is far behind and to the left 
    {
        "food_front/back_norm": -0.75, 
        "food_right/left_norm": -0.5, 
        "obstacle_front": 0.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 0.0
    },
    # 5. Food is close ahead, obstacle immediately to the left
    {
        "food_front/back_norm": 0.05, 
        "food_right/left_norm": 0.0, 
        "obstacle_front": 0.0, 
        "obstacle_right": 0.0, 
        "obstacle_left": 1.0
    }
]       
    for state in test_states:
        print(state)


    for gen in range(generations):
        print(f"Generation {gen}")
        state_actions_instinct = []
        state_actions_final = []

        for state in test_states:
            mean_score, instinct_median_action, final_median_action =get_median_gen_learned_n_instinct_action(gen, state,)
            state_actions_instinct.append(ACTION_STRING[int(instinct_median_action)])
            state_actions_final.append(ACTION_STRING[int(final_median_action)])


        results.append({
            'gen': gen,
            'median intinct actions': state_actions_instinct,
            'median learned actions':state_actions_final,
            'average learned score': ("%.2f" %mean_score)
        })
        print(f"median intinct actions {state_actions_instinct}")
        print(f"median learned actions {state_actions_final}")
        print(f"average learned score = {mean_score}")
    
    results_df = pd.DataFrame(results)
    
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    results_df.to_csv(f'baldwin_explore_results/results{ts}.csv', index=False, header=True)