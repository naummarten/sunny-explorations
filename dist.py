def euclidean_dist(features1, features2):
    return sum([(f1 - f2) ** 2 for f1, f2 in zip(features1, features2)])

def manhattan_dist(features1, features2):
    return sum([abs(f1 - f2) for f1, f2 in zip(features1, features2)])

def max_dist(features1, features2):
    return max([abs(f1 - f2) for f1, f2 in zip(features1, features2)])

def hamming_dist(features1, features2):
    return sum([1 if f1 == f2 else 0 for f1, f2 in zip(features1, features2)])

def constant_dist(features1, features2):
    return 0