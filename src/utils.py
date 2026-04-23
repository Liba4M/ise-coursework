import pandas as pd


def load_dataset(path, target_column):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def is_better(new_score, best_score, maximize):
    if best_score is None:
        return True
    if maximize:
        return new_score > best_score
    return new_score < best_score
