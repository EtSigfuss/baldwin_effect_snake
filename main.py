import numpy as np
import pygad

from constants import SNAKE_FEATURE_COLS, episodes_per_life
from snake import Snake
from solve_snake import run_episode
from torch_logic import Agent

seed_value = 1234
rng = np.random.default_rng(seed_value)#global rng

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
learn_rate = 0.01
learn_rate_floor = 0.01
learn_rate_decay = (learn_rate_floor/learn_rate)**(1/episodes_per_life)
explore_rate = 0.2
explore_rate_floor = 0.001
explore_rate_decay_generation = (explore_rate_floor/explore_rate)**(1/generations)
explore_rate_decay_episode = (explore_rate_floor/explore_rate)**(1/episodes_per_life)
discount = 0.95

def fitness_func_learning(ga_instance, solution, solution_idx):
    """
    An attempt at using deep q learning via torch for running snake

    """
    global watch_mod

    render_mode_life = None
    render_life = False

    
    #allows you to watch the elite of prev gen
    if watch_mod % pop == 0:
        render_mode_life = "pygame"
        render_life = True


    watch_mod += 1

    agent = Agent()

    total_score = 0

    # generation_learn_rate = learn_rate*learn_rate_decay**generation
    # generation_explore_rate = explore_rate*explore_rate_decay_generation**generation
    env = Snake(
        width=width,
        height=height,
        lifespan=lifespan,
        state_includes_location=False,
        state_includes_sensory=True,
        render_mode=render_mode_life,
        seed=rng.integers(1,1000000),
        frame_rate=60
        )
    for _ in range(episodes_per_life):
        _, score = run_episode(env=env,
                                discount=discount,
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
        state_includes_location=False,
        state_includes_sensory=True,
        render_mode=None,
        seed=seed_value
        )

    for _ in range(episodes_per_life):
        _, score = run_episode(env=env,
                            solution=solution)
        total_score += score
        env.reset()
    env.close()
    return total_score/episodes_per_life

env = Snake(
        width=width,
        height=height,
        lifespan=lifespan,
        state_includes_location=False,
        state_includes_sensory=True,
        render_mode="pygame",
        seed=seed_value
        )

ga = pygad.GA(num_generations=generations,
            sol_per_pop= pop,
            num_genes= (len(SNAKE_FEATURE_COLS)+1)*4,
            num_parents_mating=5,
            mutation_percent_genes= mutation_scale,
            mutation_probability= mutation_rate,
            mutation_type='random',
            fitness_func=fitness_func_instinct,
            )


ga.run()

print("done instinct")
input()

solution, solution_fitness, solution_idx = ga.best_solution()

for i in range(1,10):
    run_episode(env=env,
                solution=solution,
                render=True)
    
#bald ga time
bald_ga = pygad.GA(num_generations=generations,
            sol_per_pop= pop,
            num_genes= (len(SNAKE_FEATURE_COLS)+1)*4,
            num_parents_mating=5,
            mutation_percent_genes= mutation_scale,
            mutation_probability= mutation_rate,
            mutation_type='random',
            fitness_func=fitness_func_instinct,
            )

bald_ga.run()

print("done instinct")
input()

solution, solution_fitness, solution_idx = ga.best_solution()

for i in range(1,10):
    run_episode(env=env,
                solution=solution,
                render=True)


print(solution)
