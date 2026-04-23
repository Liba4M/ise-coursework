import random

from utils import is_better


def _normalise_features(X):
    """Normalise each column to [0, 1] so distances are comparable."""
    X_norm = X.copy().astype(float)

    for column in X_norm.columns:
        col_min = X_norm[column].min()
        col_max = X_norm[column].max()

        if col_max == col_min:
            X_norm[column] = 0.0
        else:
            X_norm[column] = (X_norm[column] - col_min) / (col_max - col_min)

    return X_norm


def get_neighbors(X_norm, index, k=10):
    """Return the indices of the k closest configurations to the current one."""
    current = X_norm.iloc[index]
    distances = ((X_norm - current) ** 2).sum(axis=1) ** 0.5
    distances = distances.sort_values()

    neighbors = [idx for idx in distances.index if idx != index]
    return neighbors[:k]


def guided_local_search(X, y, budget, maximize, k=10):
    X_norm = _normalise_features(X)
    n = len(X)

    current_index = random.randint(0, n - 1)
    current_score = y.iloc[current_index]

    best_index = current_index
    best_score = current_score

    used_budget = 1

    while used_budget < budget:
        neighbors = get_neighbors(X_norm, current_index, k=k)
        improved = False

        for idx in neighbors:
            if used_budget >= budget:
                break

            score = y.iloc[idx]
            used_budget += 1

            if is_better(score, current_score, maximize):
                current_index = idx
                current_score = score
                improved = True

                if is_better(score, best_score, maximize):
                    best_score = score
                    best_index = idx

        if not improved and used_budget < budget:
            current_index = random.randint(0, n - 1)
            current_score = y.iloc[current_index]
            used_budget += 1

            if is_better(current_score, best_score, maximize):
                best_score = current_score
                best_index = current_index

    return best_score, best_index
