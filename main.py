from multiprocessing import Pool as ThreadPool

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from dist import euclidean_dist, manhattan_dist, max_dist, hamming_dist, constant_dist
from kb import solvers, kb, best_solver
from runtest import run_test

def test_backup_solver(backup_solver):
    print("Backup solver:", backup_solver)
    return run_test(solvers, backup_solver, 5, kb, 100)

def test_k_nearest_neighbors(k):
    print("Nearest neighbors:", k)
    return run_test(solvers, best_solver, k, kb, 100)

def test_solvers(solvers):
    print("Solvers:", solvers)
    return run_test(solvers, best_solver, 5, kb, 100)

def test_dist_func(dist_func):
    print("Dist func: ", dist_func.__name__)
    return run_test(solvers, best_solver, 5, kb, 100, dist_func)

def test_feature_reduction(num_features):
    print("Number of features: ", num_features)
    reduced_kb = kb.copy()
    variances = []
    for i in range(0, len(reduced_kb.feat[0])):
        feat = [f[i] for f in reduced_kb.feat]
        mean = sum(feat) / len(feat)
        variance = sum((f - mean) ** 2 for f in feat) / len(feat)
        variances.append((i, variance))
    variances.sort(key=lambda x: x[1], reverse=False)
    reduced_feats = [x[0] for x in variances[0:num_features]]
    for i, feat in enumerate(reduced_kb.feat):
        reduced_kb.feat[i] = [f for j, f in enumerate(feat) if j in reduced_feats]
    return run_test(solvers, best_solver, 5, reduced_kb, 100)

def graph_backup_solvers(df, pool):
    results = pool.map(test_backup_solver, solvers)
    df["solver"] = solvers
    df["psi"] = [x[0] for x in results]
    sb.barplot(data=df, y="solver", x="psi")
    plt.xlim(0.8, 1.0)

def graph_nearest_neighbors(df, pool):
    results = pool.map(test_k_nearest_neighbors, range(1, 21))
    df["k"] = [x for x in range(1, 21)]
    df["psi"] = [x[0] for x in results]
    sb.pointplot(data=df, x="k", y="psi", color="orange")

def graph_solvers(df, pool):
    results = pool.map(test_solvers, [solvers[0:i] for i in range(1, len(solvers) + 1)])
    df["portfolio size"] = [x for x in range(1, len(solvers) + 1)]
    df["average sub-portfolio size"] = [x[2] for x in results]
    sb.barplot(data=df, x="portfolio size", y="average sub-portfolio size", color="gold")
    plt.ylim(0.8, 1.6)

def graph_dist_funcs(df, pool):
    funcs = [euclidean_dist, manhattan_dist, max_dist, hamming_dist, constant_dist]
    results = pool.map(test_dist_func, funcs)
    df["distance function"] = [x.__name__ for x in funcs]
    df["psi"] = [x[0] for x in results]
    sb.barplot(data=df, x="distance function", y="psi", color="green")
    plt.ylim(0.6, 1.0)

def graph_feature_reduction(df, pool):
    num_features = [1, 3, 6, 10, 20, 30, 75]
    results = pool.map(test_feature_reduction, num_features)
    df["number of features"] = num_features
    df["psi"] = [x[0] for x in results]
    sb.barplot(data=df, x="number of features", y="psi", color="blue")
    plt.ylim(0.8, 1.0)

if __name__ == '__main__':
    df = pd.DataFrame()
    pool = ThreadPool(6)
    graph_feature_reduction(df, pool)
    plt.show()
    print(df)
