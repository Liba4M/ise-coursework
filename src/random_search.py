import random
from utils import is_better


def random_search(X, y, budget, maximize):
    best_score = None
    best_index = None
    n = len(X)

    for _ in range(budget):
        idx = random.randint(0, n - 1)
        score = y.iloc[idx]

        if is_better(score, best_score, maximize):
            best_score = score
            best_index = idx

    return best_score, best_index
