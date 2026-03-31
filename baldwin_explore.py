import itertools
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import json
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
lifespan = 1000

#NN params
episodes_per_life = 10
input_size = len(SNAKE_FEATURE_COLS)
hidden_size = 2**4
output_size = len(ACTIONS)

# GA hyperparams
pop = 50
num_parents_mating = int(pop*.15)
generations = 50
mutation_probability = 0.13
gene_size = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size

#observation settings
fitness_eval_count = 0
generation_high_score = 0
starting_and_learned_weights_store_by_gen = defaultdict(list)
starting_and_learned_weights_store_by_gen_and_indiv = {}

def log_params(ts = None):
    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    log_path = f'baldwin_explore_results/params_{ts}.json'

    params = {
        "timestamp": ts,
        "environment": {
            "seed_value": seed_value,
            "height": height,
            "width": width,
            "lifespan": lifespan
        },
        "neural_network": {
            "episodes_per_life": episodes_per_life,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size
        },
        "ga_hyperparams": {
            "population": pop,
            "num_parents_mating": num_parents_mating,
            "generations": generations,
            "mutation_probability": mutation_probability,
            "gene_size": gene_size
        }
    }

    with open(log_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"Parameters successfully logged to {log_path}")

def fitness_func_learning(ga_instance, solution, solution_idx):
    """

    """
    global starting_and_learned_weights_store_by_gen
    global starting_and_learned_weights_store_by_gen_and_indiv

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
    for ep in range(episodes_per_life):
        score = run_episode(env=env,
                                learn=True,
                                render=render_life,
                                agent=agent
                                )
        
        ## to minimize random action effect
        if ep > episodes_per_life/2:
            total_score += score
        env.reset()
    env.close()


    #capture all agents and starting weights in a gen index dict 

    starting_and_learned_weights_store_by_gen[current_gen].append((score,agent,solution))
    starting_and_learned_weights_store_by_gen_and_indiv[(current_gen, solution_idx)] = (score, agent, solution)
    

    return total_score

def extract_best_agent_from_gen(gen):
    """
    finds the best performing individual in a population returns the score, trained agent and solution
    """
    #filter for only specified gen
    gen_data = {k: v for k, v in starting_and_learned_weights_store_by_gen_and_indiv.items() if k[0] == gen}
    
    #find where score is hightst
    best_gen_key = max(gen_data, key = lambda k: gen_data[k][0])
    return  gen_data[best_gen_key]

def get_mode_gen_learned_n_instinct_action(gen: int, state: dict[str, float])  :
    """
    gets the mode actions for a game state
    """
    global starting_and_learned_weights_store_by_gen

    #list of actions of all agents of a generation for a state
    agent_learned_actions = []
    agent_instinct_actions = []
    #iterate through all instinct and learned agents of a generation and get the mode actions
    scores = []
    for score, learned_agent, solution in starting_and_learned_weights_store_by_gen[gen]:

        agent_instinct = Agent(solution=solution,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size = output_size,
                        )
        
        
        
        agent_learned_actions.append(learned_agent.get_action(featurize_state(state),False))
        agent_instinct_actions.append(agent_instinct.get_action(featurize_state(state),False))
        scores.append(score)
        
    instinct_mode_action = statistics.mode(agent_instinct_actions)
    learned_mode_action = statistics.mode(agent_learned_actions)
    mean_score = statistics.mean(scores)

    return mean_score, instinct_mode_action, learned_mode_action


def mode_actions_of_generations(test_states, ts = None):
    """gets actions of mode agents learned behavior and instincs across test states"""
    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    results_instinct = []
    results_learned = []

    for gen in range(generations):
        
        
        instinct_row = {'gen': gen, 'avg_score': 0.0}
        learned_row = {'gen': gen, 'avg_score': 0.0}
        current_gen_avg_score = 0


        for index, state in enumerate(test_states):

            mean_score, instinct_mode_action, learn_mode_action = get_mode_gen_learned_n_instinct_action(gen,state)
        
            instinct_row[f'{index}'] = ACTION_STRING[int(instinct_mode_action)]
            learned_row[f'{index}'] = ACTION_STRING[int(learn_mode_action)]

            current_gen_avg_score = mean_score

        instinct_row['avg_score'] = float(current_gen_avg_score)
        learned_row['avg_score'] = float(current_gen_avg_score)

        results_instinct.append(instinct_row)
        results_learned.append(learned_row)



    
    results_instinct_df = pd.DataFrame(results_instinct)
    results_learned_df = pd.DataFrame(results_learned)



    results_instinct_df.to_csv(f'baldwin_explore_results/mode_instinct_{ts}.csv', index=False)
    results_learned_df.to_csv(f'baldwin_explore_results/mode_learned_{ts}.csv', index=False)

    states_only_i = results_instinct_df.drop(columns=['gen', 'avg_score'])
    states_only_l = results_learned_df.drop(columns=['gen', 'avg_score'])
    #-----------------------------------------------------------------------
    #heat map
    assimilation_map = (states_only_i == states_only_l).astype(int)

    plt.figure(figsize=(16, 8))
    sns.heatmap(assimilation_map, 
                cmap="RdYlGn", 
                cbar_kws={'label': '0: learned | 1: innate'},
                linewidths=0.05, 
                linecolor='gray')
    
    plt.title(f"where learned behavior differs from instictual(mode actions of whole gen)", fontsize=16)
    plt.xlabel("state (0-71)", fontsize=12)
    plt.ylabel("generation", fontsize=12)

    plt.tight_layout()
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig(f"baldwin_explore_results/mode_learn_inst_match_map_{ts}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    #create line chart of aggreeement per generation
    #-----------------------------------------------------------------------
    agreement_per_gen = assimilation_map.mean(axis=1) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(results_instinct_df['gen'], agreement_per_gen,
            marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=4)

    plt.title("Action Agreement Between Mode Instinct and Learned Behavior", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Agreement Percentage (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 105)  


    plt.tight_layout()
    plt.savefig(f"baldwin_explore_results/agreement_trend_mode_{ts}.png", dpi=300)
    # plt.show()
    plt.close()

def elite_actions_of_generations(test_states, ts = None): 
    """gets actions of best agents learned behavior and instincs across test states"""
    results_instinct = []
    results_learned = []

    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    for gen in range(generations):
    
        # get best agent
        best_score, best_learned_agent, best_solution = extract_best_agent_from_gen(gen)

        #starting version of Agent
        best_instinct_agent = Agent(
            solution=best_solution,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        )

        instinct_state_actions = {'gen': gen, 'best_score': float(best_score)}
        learned_state_actions = {'gen': gen, 'best_score': float(best_score)}

        for index, state in enumerate(test_states):
            feat_state = featurize_state(state)

            #get action of both instinct and learned agent
            instinct_act = best_instinct_agent.get_action(feat_state, False)
            learned_act = best_learned_agent.get_action(feat_state, False)

            # best_actions_instinct.append(ACTION_STRING[int(instinct_act)])
            # best_actions_learned.append(ACTION_STRING[int(learned_act)])

            instinct_state_actions[f'{index}'] = ACTION_STRING[int(instinct_act)]
            learned_state_actions[f'{index}'] = ACTION_STRING[int(learned_act)]

            
        results_instinct.append(instinct_state_actions)
        results_learned.append(learned_state_actions)

        
    results_instinct_df = pd.DataFrame(results_instinct)
    results_learned_df = pd.DataFrame(results_learned)


    results_instinct_df.to_csv(f'baldwin_explore_results/results_best_instinct_{ts}.csv', index=False)
    results_learned_df.to_csv(f'baldwin_explore_results/results_best_learned_{ts}.csv', index=False)

    #plot heatmap
    #-----------------------------------------------------------------------
    states_only_instinct = results_instinct_df.drop(columns=['gen', 'best_score'])
    states_only_learned = results_learned_df.drop(columns=['gen', 'best_score'])

    assimilation_map = (results_instinct_df == results_learned_df).astype(int)


    plt.figure(figsize=(16, 8))
    sns.heatmap(assimilation_map, 
                cmap="RdYlGn", 
                cbar_kws={'label': '0: learned | 1: innate'},
                linewidths=0.05, 
                linecolor='gray')

    plt.title("where learned behavior differs from instictual in best performing indiv", fontsize=16)
    plt.xlabel("state (0-71)", fontsize=12)
    plt.ylabel("generation", fontsize=12)

    plt.tight_layout()
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig(f"baldwin_explore_results/best_learn_inst_match_map_{ts}.png", dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()

    #create line chart of aggreeement per generation
    #-----------------------------------------------------------------------
    agreement_per_gen = assimilation_map.mean(axis=1) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(results_instinct_df['gen'], agreement_per_gen,
            marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=4)

    plt.title("Action Agreement Between Best Indiv Instinct and Learned Behavior", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Agreement Percentage (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 105)


    plt.tight_layout()
    plt.savefig(f"baldwin_explore_results/agreement_trend_best_{ts}.png", dpi=300)
    # plt.show()
    plt.close()

def get_every_state():
    """generates every permutaion in a 5 long array of pos 0-1 being trinary and 2-4being binary
    then stuffs it in a dict to align with the rest of the code"""
    pos_0_1 = [-1, 0, 1]
    pos_2_4 = [0, 1]
    all_states = list(itertools.product(pos_0_1, pos_0_1, pos_2_4, pos_2_4, pos_2_4))
    all_states_dict_array = []
    for state in all_states:
        all_states_dict_array.append({
            "food_front/back_norm": state[0],
            "food_right/left_norm": state[1],
            "obstacle_front": state[2],
            "obstacle_right": state[3],
            "obstacle_left": state[4],
        })
    return all_states_dict_array


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

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_params(ts)


    bald_ga = pygad.GA(
                num_generations=generations,
                sol_per_pop= pop,
                num_genes= gene_size,
                keep_elitism=3,
                num_parents_mating=num_parents_mating,
                mutation_probability= mutation_probability,
                mutation_type='random',
                fitness_func=fitness_func_learning,
                # parallel_processing=['process', 4]
                crossover_type="uniform",
                parent_selection_type='tournament',
                )

    bald_total = 0




    bald_ga.run()

    bald_solution, solution_fitness, solution_idx = bald_ga.best_solution()

    print("\ndone bald")
    


    all_states = get_every_state()
    mode_actions_of_generations(all_states, ts)
    elite_actions_of_generations(all_states, ts)