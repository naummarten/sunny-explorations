from sklearn.model_selection import RepeatedKFold

from dist import euclidean_dist
from sunny import sunny

def run_test(solvers, backup_solver, k, kb, repeats, dist_func=euclidean_dist):
    trials = 0
    solves = 0
    total_time = 0
    oracle_solves = 0
    oracle_total_time = 0
    single_best_solves = 0
    single_best_total_time = 0
    total_subportfolio_size = 0
    max_subportfolio_size = 0
    rkf = RepeatedKFold(n_splits=5, n_repeats=repeats)
    for train_indexes, test_indexes in rkf.split(kb):
        train_kb = kb.iloc[train_indexes].copy()
        test_kb = kb.iloc[test_indexes].copy()
        for index, inst in test_kb.iterrows():
            trials += 1
            schedule = sunny(inst, solvers, backup_solver, k, 1200, train_kb, dist_func)
            total_subportfolio_size += len(schedule)
            for solver, time in schedule:
                if time > inst.time[solver]:
                    solves += 1
                    total_time += inst.time[solver]
                    break
                total_time += time
            best_time = min(inst.time[solver] for solver in solvers)
            oracle_total_time += best_time
            if best_time < 1200: oracle_solves += 1
            single_best_total_time += inst.time[backup_solver]
            if inst.time[backup_solver] < 1200: single_best_solves += 1
    return solves / trials, total_time / trials, total_subportfolio_size / trials,
    # print(solves, oracle_solves, single_best_solves, trials)
    # print(total_time / trials, oracle_total_time / trials, single_best_total_time / trials)
