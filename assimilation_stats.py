import glob
import os

import numpy as np
import pandas as pd

base_dir = "./baldwin_explore_results"

# find all assimilation files in any sub-folder
path_pattern = os.path.join(base_dir, "**/assimilation_events_*.csv")
assimilation_files = glob.glob(path_pattern, recursive=True)

all_runs = []
for file in assimilation_files:
    df = pd.read_csv(file)
    df['experiment_id'] = os.path.basename(os.path.dirname(file))
    all_runs.append(df)

if not all_runs:
    print("no data found.")

total_df  = pd.concat(all_runs, ignore_index=True)

stats = {
    "total unique events": len(total_df),
    "mean lag": total_df['lag'].mean(),
    "median lag": total_df['lag'].median(),
    "fixation rate (%)": (total_df['fixed'].sum() / len(total_df)) * 100,
    "most common action": total_df['novel'].mode()[0],
    "avg events per gen": total_df.groupby('inst_adopted_at').size().mean()
}

action_breakdown = total_df.groupby('novel')['lag'].mean().to_dict()


print("assimilation stats: ")
for key, val in stats.items():
    v = f"{val:.2f}" if isinstance(val, np.float64) else val
    print(f"{key}: {v}")