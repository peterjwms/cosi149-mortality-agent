import csv
from pathlib import Path

import pandas as pd

filename = "AI_agent_train_sepsis.csv"
with open(Path(filename), "r", encoding='utf-8'):
    reader = csv.reader(filename)
    # want to ignore "",bloc,icustayid,charttime" - first 4 columns
    i = 0

    df = pd.read_csv(Path(filename))
    print(df.head)

    means = df.mean(axis=0)
    maxes = df.max(axis=0)
    mins = df.min(axis=0)
    stdevs = df.var(axis=0)
    print(means)

    stats = pd.DataFrame({"mean": means, "maxes": maxes, "mins": mins, "stdevs": stdevs})

    real_stats = df.describe()
    print(real_stats)

    # print(stats)
    # print(df.loc)