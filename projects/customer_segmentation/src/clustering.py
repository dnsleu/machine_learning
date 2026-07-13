from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from typing import Dict, List, Optional, Tuple, Iterable


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k_distance_elbow_plot(
    df: pd.DataFrame,
    ncols: int = 4,
    min_samples_values: Optional[Iterable[int]] = None,
    S: float = 1.0,
    metric: str = "euclidean",
) -> Tuple[plt.Figure, List[float], Dict[int, float]]: # type: ignore
    """
    k-distance elbow plots (one per min_samples/k) and returns:
      - fig: matplotlib Figure
      - eps_guess_list: list of eps guesses in the same order as min_samples_values
      - eps_guess_by_ms: dict mapping min_samples -> eps_guess (np.nan if none)
    """

    def k_distance_curve(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        nn = NearestNeighbors(n_neighbors=k, metric=metric)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # distance to the k-th neighbor for each point
        kth = distances[:, k - 1]

        # sort descending
        y = np.sort(kth)[::-1]
        x = np.arange(1, len(y) + 1)

        kl = KneeLocator(
            x, y,
            S=S,
            curve="convex",
            direction="decreasing"
        )

        return x, y, kl.knee_y  # None if no knee found

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    if min_samples_values is None:
        min_samples_values = range(6, 10)
    min_samples_values = list(min_samples_values)

    eps_guess_list: List[float] = []
    eps_guess_by_ms: Dict[int, float] = {}

    n = len(min_samples_values)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.5 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, ms in enumerate(min_samples_values):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        x, y, eps_guess = k_distance_curve(X_scaled, k=ms)

        ax.plot(x, y, linewidth=1)
        ax.set_title(f"min_samples = {ms}")
        ax.set_xlabel("Points (sorted)")
        ax.set_ylabel(f"{ms}-NN distance")

        if eps_guess is not None:
            ax.axhline(eps_guess, linestyle="--")
            ax.text(
                0.02, 0.95,
                f"eps≈{eps_guess:.3f}",
                transform=ax.transAxes,
                va="top"
            )
            eps_guess_list.append(float(eps_guess))
            eps_guess_by_ms[ms] = float(eps_guess)
        else:
            ax.text(
                0.02, 0.95,
                "eps≈(no knee found)",
                transform=ax.transAxes,
                va="top"
            )
            eps_guess_list.append(float("nan"))
            eps_guess_by_ms[ms] = float("nan")

    # hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    return fig, eps_guess_list, eps_guess_by_ms