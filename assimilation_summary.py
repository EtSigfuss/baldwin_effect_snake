import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baldwin_explore import get_every_state


def lookup_state(state_id):
    s = get_every_state()[state_id]
    return {
        "food_fb": s["food_front/back_norm"], 
        "food_rl": s["food_right/left_norm"],
        "obs_f":   s["obstacle_front"], 
        "obs_r":   s["obstacle_right"], 
        "obs_l":   s["obstacle_left"]
    }

base_dir = "./baldwin_explore_results"

# find all assimilation files in any sub-folder
path_pattern = os.path.join(base_dir, "**/assimilation_events_*.csv")
assimilation_files = glob.glob(path_pattern, recursive=True)

all_runs = []
run_amount = 0
for file in assimilation_files:
    df = pd.read_csv(file)
    df['experiment_id'] = os.path.basename(os.path.dirname(file))
    all_runs.append(df)

if not all_runs:
    print("no data found.")

total_df  = pd.concat(all_runs, ignore_index=True)
total_df = total_df.drop_duplicates()
run_amount = total_df['experiment_id'].nunique()




action_breakdown = total_df.groupby('novel').agg(
    events        = ('novel', 'count'),
    fixation_rate = ('fixed', lambda x: round(x.mean() * 100, 1)),
    mean_lag      = ('lag',   lambda x: round(x.mean(), 2)),
)


state_consistency = total_df.groupby('state').agg(
    times_assimilated = ('state',         'count'),
    runs_appeared_in  = ('experiment_id', 'nunique'),
    fixation_rate     = ('fixed',         lambda x: round(x.mean() * 100, 1)),
)

per_run = total_df.groupby('experiment_id').agg(
    events   = ('state', 'count'),
    unique_state_appearances = ('state', 'nunique'),
    fixed    = ('fixed', 'sum'),
    mean_lag = ('lag',   'mean'),
).round(2)

state_summary = total_df.groupby('state')['fixed'].agg(['count', 'mean'])

stats = {
    "runs found":              run_amount,
    "total unique events":     len(total_df),
    "mean lag":                total_df['lag'].mean(),
    "median lag":              total_df['lag'].median(),
    "fixation rate (%)":       (total_df['fixed'].sum() / len(total_df)) * 100,
    "most common action":      total_df['novel'].mode()[0],
    "avg events per gen":      total_df.groupby('inst_adopted_at').size().mean(),
    "events per run (mean)":   total_df.groupby('experiment_id').size().mean(),
    "events per run (std)":    total_df.groupby('experiment_id').size().std(),
    "lag vs fixed corr":       total_df[['lag', 'fixed']].corr().loc['lag', 'fixed'],
    "state vs fix rate corr":  state_summary['count'].corr(state_summary['mean']),
    "state re-entry rate": 1 - per_run['unique_state_appearances'].sum() / per_run['events'].sum()
}

print("assimilation stats: ")

for key, val in stats.items():
    v = f"{val:.2f}" if isinstance(val, (np.float64, float)) else val
    print(f"{key}: {v}")

print("\naction breakdown:")
print(action_breakdown.to_string())

print("\nstate consistency across runs (top 10 by appearance amt):")
print(state_consistency.sort_values('runs_appeared_in', ascending=False).head(10).to_string())

state_by_fix_rate = state_consistency.sort_values('fixation_rate', ascending=False)

print("\nstate consistency across runs (top 10 by fixation rate):")
print(state_by_fix_rate.head(10).to_string())


print("\nphysical scenarios:")
for state_id, row in state_by_fix_rate.iterrows():
    senses = lookup_state(state_id)
    fix_val = row['fixation_rate']
    print(f"State {state_id:<2} | fix rate: {fix_val:>5}% | Sensors: {senses}")






print("\nstate consistency across runs (top 10 by times_assimilated):")
print(state_consistency.sort_values('times_assimilated', ascending=False).head(10).to_string())
 
print("\nper run:")
print(per_run.to_string())


#-----------------------------------------------------------------
#fixation rate across states
plot_df = state_consistency.reset_index()

plt.figure(figsize=(10, 6))

plt.scatter(
    plot_df['state'], 
    plot_df['fixation_rate'], 
    s=plot_df['runs_appeared_in'] * 50, 
    alpha=0.6, 
    edgecolors='white', 
    linewidth=1.5,
    c='tab:blue'
)

plt.title('state vs. fixation rate', fontsize=14, fontweight='bold')
plt.xlabel('State ID', fontsize=12)
plt.ylabel('Fixation Rate (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("state vs. fixation rate.png")
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------
#look for state fixation rate clustering around mean

state_rates = total_df.groupby('state')['fixed'].mean() * 100

plt.figure(figsize=(9, 5))

n, bins, patches = plt.hist(state_rates, bins=20, range=(0, 100), 
                            color='teal', edgecolor='black', alpha=0.7)

plt.axvline(state_rates.mean(), color='red', linestyle='dashed', linewidth=2, 
            label=f'global average ({state_rates.mean():.1f}%)')

plt.title('number of states with simular fixation rates', fontsize=14)
plt.xlabel('Fixation Rate (%)', fontsize=12)
plt.ylabel('Number of States', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.savefig("number of states with simular fixation rates.png")
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------
# average fixation rate by sense 
sensor_data = []
for state_id, row in state_consistency.iterrows():
    senses = lookup_state(state_id)
    senses['fixation_rate'] = row['fixation_rate']
    sensor_data.append(senses)
sensor_df = pd.DataFrame(sensor_data)

sensory_analysis = {
    "food ahead (1)":    sensor_df[sensor_df['food_fb'] == 1]['fixation_rate'].mean(),
    "food inline horiz (0)": sensor_df[sensor_df['food_fb'] == 0]['fixation_rate'].mean(),
    "food behind (-1)":  sensor_df[sensor_df['food_fb'] == -1]['fixation_rate'].mean(),
    
    "food right (1)":    sensor_df[sensor_df['food_rl'] == 1]['fixation_rate'].mean(),
    "food in line (0)":sensor_df[sensor_df['food_rl'] == 0]['fixation_rate'].mean(),
    "food left (-1)":   sensor_df[sensor_df['food_rl'] == -1]['fixation_rate'].mean(),
    
    "wall front (1)":    sensor_df[sensor_df['obs_f'] == 1]['fixation_rate'].mean(),
    "wall right (1)":sensor_df[sensor_df['obs_r'] == 1]['fixation_rate'].mean(),
    "wall left (1)":sensor_df[sensor_df['obs_l'] == 1]['fixation_rate'].mean(),

    "no wall front (0)":    sensor_df[sensor_df['obs_f'] == 0]['fixation_rate'].mean(),
    "no wall right (0)":sensor_df[sensor_df['obs_r'] == 0]['fixation_rate'].mean(),
    "no wall left (0)":sensor_df[sensor_df['obs_l'] == 0]['fixation_rate'].mean(),
}

sensory_series = pd.Series(sensory_analysis).sort_values()

plt.figure(figsize=(12, 7))
colors = ['gray' if '(0' in x else 'skyblue' for x in sensory_series.index]
colors = ['tomato' if 'wall' in name else color for name, color in zip(sensory_series.index, colors)]

sensory_series.plot(kind='barh', color=colors, edgecolor='black', alpha=0.8)

plt.axvline(state_rates.mean(), color='red', linestyle='--', label=f'Global Avg ({state_rates.mean():.1f}%)')
plt.title('average fixation rate by sense ', fontsize=14, fontweight='bold')
plt.xlabel('Average Fixation Rate (%)', fontsize=12)
plt.ylabel('Sensory State')
plt.legend()
plt.grid(axis='x', linestyle=':', alpha=0.6)

plt.savefig("average fixation by sense.png")
plt.tight_layout()
plt.show()