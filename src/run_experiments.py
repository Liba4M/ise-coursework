from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from guided_local_search import guided_local_search
from random_search import random_search
from utils import load_dataset


RESULTS_DIR = Path("results")
RUNS = 30
BUDGET = 50


def _summary_dataframe(random_results, guided_results):
    rows = []

    for method_name, scores in (
        ("Random Search", random_results),
        ("Guided Local Search", guided_results),
    ):
        rows.append(
            {
                "method": method_name,
                "median": float(np.median(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
            }
        )

    return pd.DataFrame(rows)


def _wilcoxon_dataframe(random_results, guided_results):
    try:
        statistic, p_value = wilcoxon(random_results, guided_results)
    except ValueError:
        statistic, p_value = np.nan, np.nan

    return pd.DataFrame(
        [
            {
                "test": "Wilcoxon signed-rank",
                "statistic": statistic,
                "p_value": p_value,
            }
        ]
    )


def run_experiment(dataset_name, path, target_column, maximize, budget=BUDGET, runs=RUNS):
    X, y = load_dataset(path, target_column)

    random_results = []
    guided_results = []

    for run in range(1, runs + 1):
        random_score, _ = random_search(X, y, budget, maximize)
        guided_score, _ = guided_local_search(X, y, budget, maximize)

        random_results.append(random_score)
        guided_results.append(guided_score)

    raw_results = pd.DataFrame(
        {
            "run": list(range(1, runs + 1)),
            "random_score": random_results,
            "guided_score": guided_results,
        }
    )

    summary = _summary_dataframe(random_results, guided_results)
    wilcoxon_result = _wilcoxon_dataframe(random_results, guided_results)

    RESULTS_DIR.mkdir(exist_ok=True)
    dataset_prefix = dataset_name.lower()

    raw_results.to_csv(RESULTS_DIR / f"{dataset_prefix}_raw_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"{dataset_prefix}_summary.csv", index=False)
    wilcoxon_result.to_csv(RESULTS_DIR / f"{dataset_prefix}_wilcoxon.csv", index=False)

    print(f"\n--- {dataset_name} ---")
    for _, row in summary.iterrows():
        print(
            f"{row['method']}: "
            f"median={row['median']:.6f}, "
            f"mean={row['mean']:.6f}, "
            f"std={row['std']:.6f}"
        )

    wilcoxon_row = wilcoxon_result.iloc[0]
    print(
        "Wilcoxon signed-rank: "
        f"statistic={wilcoxon_row['statistic']}, "
        f"p-value={wilcoxon_row['p_value']}"
    )

    return raw_results, summary, wilcoxon_result


if __name__ == "__main__":
    run_experiment("storm", "data/storm.csv", "<$latency", maximize=False)
    run_experiment("x264", "data/x264.csv", "performance", maximize=True)
