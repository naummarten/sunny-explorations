from itertools import combinations

from dist import euclidean_dist, constant_dist

def sunny(inst, solvers, backup_solver, k, time, kb, dist_func=euclidean_dist):
    features = inst.feat
    neighbors = get_neighbors(features, k, kb, dist_func)
    sub_portfolio = get_sub_portfolio(neighbors, solvers)
    backup_slots = (k - get_solved(sub_portfolio, neighbors))
    slots = sum([get_solved([solver], neighbors) for solver in sub_portfolio]) + backup_slots
    slot_time = time / slots
    schedule = {backup_solver: 0}
    for solver in sub_portfolio:
        schedule[solver] = get_solved([solver], neighbors) * slot_time
    schedule[backup_solver] += backup_slots * slot_time
    sorted_schedule = [(solver, schedule[solver]) for solver in schedule]
    sorted_schedule.sort(key=lambda x: sum([time[x[0]] for time in neighbors.time]))
    return sorted_schedule

def get_neighbors(features, k, kb, dist_func):
    kb["dist"] = [dist_func(features, f) for f in kb.feat]
    return kb.nsmallest(k, "dist")

def get_sub_portfolio(neighbors, solvers):
    max_solved = 0
    average_time = 0
    sub_portfolio = []
    solvable = len([time for time in neighbors.time if any([time[solver] < 1200 for solver in solvers])])
    for subset_size in range(1, len(solvers) + 1):
        if max_solved == solvable: break
        for subset in combinations(solvers, subset_size):
            num_solved = 0
            for time in neighbors.time:
                if any([time[solver] < 1200 for solver in subset]):
                    num_solved += 1
            if num_solved >= max_solved:
                total_time = 0
                for solver in subset:
                    total_time += sum([time[solver] for time in neighbors.time])
                avg_time = total_time / len(subset)
                if num_solved == max_solved and avg_time > average_time: continue
                max_solved = num_solved
                average_time = avg_time
                sub_portfolio = subset
    return sub_portfolio

def get_solved(solvers, instances):
    return len([time for time in instances.time if any([time[solver] < 1200 for solver in solvers])])
