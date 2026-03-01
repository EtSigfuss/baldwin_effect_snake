import os
import warnings

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import numpy as np
import pandas as pd
import pygad

from agent_interface import run_episode
from constants import ACTIONS, SNAKE_FEATURE_COLS
from snake import Snake
from torch_logic import Agent

seed_value = 1234
rng = np.random.default_rng(seed_value)#global rng

#enviroment parames
height = 10
width = 10
lifespan = 20000

#NN params
episodes_per_life = 15
input_size = len(SNAKE_FEATURE_COLS)
hidden_size = 2**3
output_size = len(ACTIONS)

# GA hyperparams
pop = 50
generations = 20
mutation_rate = 0.10
mutation_scale = 0.10
gene_size = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size

#observation settings
watch_mod = 0

#trial and arch runs
arch_runs = 20
trial_amt = 10

def fitness_func_learning(ga_instance, solution, solution_idx):
    """
    An attempt at using deep q learning via torch for running snake

    """

    render_mode_life = None
    render_life = False


    current_gen = ga_instance.generations_completed


    print(f"generation = {current_gen} solution = {solution_idx}                ", end="\r")



    agent = Agent(solution=solution,
                  input_size=input_size,
                  hidden_size=hidden_size,
                  output_size = output_size,
                  epsilon=.2,
                  decay_actions=100,
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
    return total_score/episodes_per_life


def fitness_func_instinct(ga_instance, solution, solution_idx):
    """
    Tests a static chromosome over multiple episodes
    the phenome never changes from pure instinct
    """

    total_score = 0

    seed_value = int(rng.integers(low=0, high=2**32))
    env = Snake(
        width=width,
        height=height,
        lifespan=lifespan,
        state_includes_sensory=True,
        render_mode=None,
        seed=seed_value
        )
    
    agent = Agent(solution=solution,
                  input_size=input_size,
                  hidden_size=hidden_size,
                  output_size = output_size
                  )

    for _ in range(episodes_per_life):
        score = run_episode(env=env, learn=False, agent=agent)
        total_score += score
        env.reset()
    env.close()
    return total_score/episodes_per_life

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

    results = pd.DataFrame(columns=['run#', 'instinct score' ,'bald score'])

    instinct_total = 0
    bald_total = 0

    for _ in range(arch_runs):
        print("arch run: ", _+1, " out of ", arch_runs)

        ga_intinct = pygad.GA(num_generations=generations,
                sol_per_pop= pop,
                num_genes= gene_size,
                num_parents_mating=5,
                mutation_percent_genes= mutation_scale,
                mutation_probability= mutation_rate,
                mutation_type='random',
                fitness_func=fitness_func_instinct,
                parallel_processing=['process', 4] 
                )

        bald_ga = pygad.GA(num_generations=generations,
                sol_per_pop= pop,
                num_genes= gene_size,
                num_parents_mating=5,
                mutation_percent_genes= mutation_scale,
                mutation_probability= mutation_rate,
                mutation_type='random',
                fitness_func=fitness_func_learning,
                parallel_processing=['process', 4] 
                )

        ga_intinct.run()
        instinct_solution, solution_fitness, solution_idx = ga_intinct.best_solution()
        print("done instinct")


        bald_ga.run()
        bald_solution, solution_fitness, solution_idx = bald_ga.best_solution()
        print("\ndone bald                  ")

        instinct_agent = Agent(solution=instinct_solution,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size = output_size,
                        decay_actions= trial_amt)

        bald_agent = Agent(solution=bald_solution,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size = output_size,
                        decay_actions= trial_amt)


        trial_env = Snake(
                width=width,
                height=height,
                lifespan=lifespan,
                state_includes_sensory=True,
                render_mode="pygame",
                seed=seed_value,
                frame_rate=60
                )
        instinct_score = 0
        bald_score = 0

        for i in range(trial_amt):
            instinct_score += run_episode(env=trial_env,
                        render=True, agent=instinct_agent)
        for i in range(trial_amt):
            bald_score += run_episode(env=trial_env,
                        render=True, agent=bald_agent, learn= True)
            
        print("instinct score ", instinct_score)
        print("bald score ", bald_score)

        instinct_total += instinct_score
        bald_total += bald_score
        trial_env.close()

        current_results = pd.DataFrame({'run#': _, 'instinct score': instinct_score ,'bald score':bald_score})
        results = pd.concat([results, current_results], ignore_index=True)


    print("instinct total: ",instinct_total)
    print("bald total: ", bald_total)

    results['bald average'] = results['bald score'].mean()
    results['instinct average'] = results['instinct score'].mean()

    results.to_csv('baldwin_instinct_compare_results/results', mode='a', index=False, header=False)