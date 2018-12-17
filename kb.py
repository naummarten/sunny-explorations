from ast import literal_eval

import pandas as pd

from sunny import get_solved

kb = pd.read_csv("mznc1215_csp.csv", sep="|", names=["inst", "feat", "time"], converters={"feat": literal_eval, "time": literal_eval})
for data in kb.time:
    for solver in data:
        data[solver] = data[solver]["time"]
solvers = list(kb.time[0].keys())

solvers.sort(key=lambda solver: get_solved([solver], kb), reverse=True)
best_solver = solvers[0]
for solver in solvers:
    solved = get_solved([solver], kb)
    time = sum([time[solver] for time in kb.time])
    # print(solver, solved, round(time / len(kb), 2))