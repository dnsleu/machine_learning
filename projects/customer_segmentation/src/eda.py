from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import scikit_posthocs as sp 

from scipy.stats import shapiro, levene, f_oneway, kruskal

@dataclass(frozen=True)
class EDAConfig:
    feature_cols: list[str]
    cat_col: Optional[str] = None
    random_state: int = 42

# Data loading & summary

def overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """dtype, missing count, misisng %, nunique"""
    out = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing %": (df.isna().mean() * 100).round(2),
            "nunique": df.nunique(dropna=False),
        }
    )
    return out.sort_values(["missing", "nunique"], ascending=[False,False])

def desribe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # to summarize central tendency / dispersion / shape for numeric cols
    return df[cols].describe().T

def plot_distributions(df: pd.DataFrame, cols: list[str], bins: int = 30) -> None:
    """Histogram + KDE per feature"""
    for c in cols:
        plt.figure(figsize=(8,5))
        sns.histplot(df[c], bins=bins, kde=True) # type: ignore
        plt.title(f"Distribution: {c}")
        plt.tight_layout()
        plt.show()

def load_csv(data_path: str) -> pd.DataFrame:
    out = pd.read_csv(data_path)
    return out

def plot_pairwise(df: pd.DataFrame, cols: list[str], hue: Optional[str] = None, sample: int = 800, diag_kind: str = "kde") -> None:
    """Pairplot of columns"""
    plot_df = df[cols + ([hue] if hue else [])].copy()
    if sample and len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)

    sns.pairplot(plot_df, hue=hue if hue else None, diag_kind=diag_kind) # type: ignore
    plt.suptitle("Pairplot", y=1.02)
    plt.show()

def plot_corr_heatmap(df: pd.DataFrame, cols: list[str]) -> None:
    corr = df[cols].corr()
    plt.figure(figsize=(8,5))
    sns.heatmap(corr, annot=True)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()

# PCA graph

def pca_2d(df: pd.DataFrame, cols: list[str], standardize: bool = True, random_state: int = 42) -> Tuple[pd.DataFrame, PCA, Optional[StandardScaler]]:
    """Returns dataframe with columns PC1 and PC2; PCA object (explained variance ratio, components); scaler"""

    X = df[cols].to_numpy(dtype=float)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)

    proj = pd.DataFrame(Z, columns=["PC1", "PC2"])
    return proj, pca, scaler

def pca_loadings(pca: PCA, cols: list[str]) -> pd.DataFrame:
    """
    Loadings table: mapping from original features onto principal components. pca.components_ with shape (n_componenets, n_features)
    """

    return pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

def plot_pca_scatter(proj: pd.DataFrame, hue: Optional[pd.Series] = None, title: str = "PCA 2D projection") -> None:
    plt.figure(figsize=(8,5))
    if hue is None:
        plt.scatter(proj["PC1"], proj["PC2"])
    else:
        sns.scatterplot(x=proj["PC1"], y=proj["PC2"], hue=hue, s=40)
        plt.legend(title=hue.name)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def age_bins(
        df: pd.DataFrame,
        age_col: str = "Age",
        new_col: str = "Age_group",
        bins: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        right: bool = False,
        include_lowest: bool = True
) -> pd.DataFrame:
    """
    Default age bins: [18, 25, 35, 45, 55, 71]
    """
    out = df.copy()

    age_categories =["Young adults","Young professionals", "Mid-career adults", "Pre-retirement individuals", "Older adults / retired"]

    if bins is None:
        bins = [18, 25, 35, 45, 55, 71]

    if labels is None:
        assert len(age_categories) == len(bins) - 1, "Need one label per bin interval"
        labels = [
            f"{bins[i]} - {bins[i+1] - 1}: {age_categories[i]}" for i in range(len(bins) - 1)
        ]

    out[new_col] = pd.cut(
        out[age_col],
        bins=bins,
        labels=labels,
        right=right,
        include_lowest=include_lowest
    )
    return out

def quantile_bins(
        df: pd.DataFrame,
        col: str,
        new_col: str,
        q: int = 4,
        labels: Optional[List[str]] = None,
        duplicates: str = "drop"
) -> pd.DataFrame:
    """
    Quantile bins for income
    """
    out = df.copy()
    out[new_col] =  pd.qcut(out[col], q=q, labels=labels, duplicates=duplicates) # type: ignore
    return out

def is_categorical(arr_or_dtype) -> bool:
    dtype = getattr(arr_or_dtype, "dtype", arr_or_dtype)  # Series/Index -> dtype, else keep
    return isinstance(dtype, pd.CategoricalDtype)

def boxplot_by_group(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        title: Optional[str] = None,
        order: Optional[List[str]] = None,
        show_points: bool = True,
        rotate_sticks: int = 0
) -> None:
    plot_df = df[[group_col, value_col]].dropna()

    if order is None:
        s = plot_df[group_col]

    # 1) Ordered categories
    if is_categorical(s) and s.cat.ordered: # type: ignore
        order = [str(x) for x in s.cat.categories] # type: ignore

    else:
        uniq = [str(x) for x in s.astype(str).dropna().unique()] # type: ignore

        def sort_key(lbl: str):
            # "Q1 (low)" -> 1
            m = re.match(r"^\s*Q(\d+)\b", lbl, flags=re.IGNORECASE)
            if m:
                return (0, int(m.group(1)))

            # "18 - 24 ..." -> 18
            m = re.match(r"^\s*(\d+)", lbl)
            if m:
                return (1, int(m.group(1)))

            return (2, lbl.lower())

        order = sorted(uniq, key=sort_key)

    grouped = [
        plot_df.loc[plot_df[group_col].astype(str) == grp, value_col].values for grp in order
    ]

    fig, ax = plt.subplots(figsize = (15,5))
    ax.boxplot(grouped, tick_labels=order, showfliers=True) # type: ignore

    if show_points:
        rng = np.random.default_rng(0)
        for i, y in enumerate(grouped, start=1):
            x = rng.normal(loc=i, scale=0.04, size=len(y))
            ax.scatter(x, y, alpha=0.35, s=12) # type: ignore

    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} by {group_col}")

    if rotate_sticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_sticks, ha = "right")
    plt.tight_layout()
    plt.show()

@dataclass
class GroupTestResult:
    group_col: str
    value_col: str
    n_groups: int
    group_sizes: Dict[str, int]

    # Assumptions
    shapiro_pvalues: Dict[str, float]
    levene_stat: float
    levene_pvalue: float

    # Chosen omnibus test
    test_name: str
    test_stat: float
    test_pvalue: float

    decision_rule: str

    # omnibus metadata + effect size
    n_total: int
    k_groups: int
    epsilon_sq: Optional[float] = None
    epsilon_sq_ci95: Optional[Tuple[float, float]] = None

def _collect_groups(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        min_group_size: int = 3,
) -> Tuple[List[np.ndarray], List[str], Dict[str, int]]:
    tmp = df[[group_col, value_col]].dropna().copy()
    tmp[group_col] = tmp[group_col].astype(str)

    group_sizes = tmp.groupby(group_col)[value_col].size().to_dict()

    kept_labels = [g for g, n in group_sizes.items() if n >= min_group_size]
    kept_labels = sorted(kept_labels) # type: ignore

    arrays = [tmp.loc[tmp[group_col] == g, value_col].to_numpy() for g in kept_labels] # type: ignore
    group_sizes_kept = {g: group_sizes[g] for g in kept_labels}

    return arrays, kept_labels, group_sizes_kept # type: ignore

def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    """
    Epsilon-squared effect size for Kruskal–Wallis:
        ε² = (H - k + 1) / (n - k)
    """
    if n <= k:
        return np.nan
    return (H - k + 1.0) / (n - k)

def bootstrap_epsilon_sq_ci_kruskal(
    groups: list[np.ndarray],
    n: int,
    k: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    boot_eps = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        boot_groups = []
        for g in groups:
            if len(g) == 0:
                boot_groups.append(g)
                continue
            idx = rng.integers(0, len(g), size=len(g))
            boot_groups.append(g[idx])

        Hb, _ = kruskal(*boot_groups)
        boot_eps[b] = epsilon_squared_kruskal(float(Hb), n=n, k=k)

    lo, hi = np.nanquantile(boot_eps, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def run_group_difference_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = 0.05,
    min_group_size: int = 3,
    shapiro_max_n: int = 5000,
    levene_center: str = "median",
    # NEW: effect-size config (applies when Kruskal-Wallis is selected)
    kruskal_effect_ci: bool = True,
    kruskal_n_boot: int = 2000,
    kruskal_ci: float = 0.95,
    kruskal_seed: int = 0,
) -> GroupTestResult:
    """
    1) Shapiro per group (normality)
    2) Levene / Brown-Forsythe (variance equality)
    3) Choose:
       - Classic ANOVA if normal and equal variances
       - Welch ANOVA if normal but unequal variances
       - Kruskal-Wallis if normality violated
    """
    groups, labels, sizes = _collect_groups(df, value_col, group_col, min_group_size=min_group_size)

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups with >= {min_group_size} rows to compare.")

    n_total = int(sum(len(g) for g in groups))
    k_groups = int(len(groups))

    # Shapiro
    shapiro_p = {}
    for g_arr, g_label in zip(groups, labels):
        if len(g_arr) > shapiro_max_n:
            shapiro_p[g_label] = np.nan
        else:
            sh_stat, sh_pv = shapiro(g_arr)
            shapiro_p[g_label] = round(float(sh_pv), 4)

    # Levene
    lev_stat, lev_pv = levene(*groups, center=levene_center)
    lev_stat = float(lev_stat)
    lev_pv = float(lev_pv)

    shapiro_vals = [p for p in shapiro_p.values() if not np.isnan(p)]
    normal_enough = (len(shapiro_vals) == 0) or all(p >= alpha for p in shapiro_vals)
    equal_var = lev_pv >= alpha

    epsilon_sq = None
    epsilon_sq_ci95 = None

    # omnibus test
    if normal_enough and equal_var:
        stat, pv = f_oneway(*groups, equal_var=True)
        test_name = "One-way ANOVA"
        decision_rule = "Normality not rejected (Shapiro) + equal variances not rejected (Levene) ⇒ classic ANOVA."
    elif normal_enough and not equal_var:
        stat, pv = f_oneway(*groups, equal_var=False)
        test_name = "Welch's ANOVA"
        decision_rule = "Normality not rejected (Shapiro) but variances differ (Levene) ⇒ Welch ANOVA."
    else:
        stat, pv = kruskal(*groups)
        test_name = "Kruskal-Wallis"
        decision_rule = "Normality rejected in ≥1 group (Shapiro) -> Kruskal-Wallis as robust omnibus test."

        # effect size + CI (ε² for Kruskal–Wallis)
        if kruskal_effect_ci:
            H = float(stat)
            epsilon_sq = float(epsilon_squared_kruskal(H, n=n_total, k=k_groups))
            lo, hi = bootstrap_epsilon_sq_ci_kruskal(
                groups=groups,
                n=n_total,
                k=k_groups,
                n_boot=kruskal_n_boot,
                ci=kruskal_ci,
                seed=kruskal_seed,
            )
            epsilon_sq_ci95 = (float(lo), float(hi))

    return GroupTestResult(
        group_col=group_col,
        value_col=value_col,
        n_groups=k_groups,
        group_sizes=sizes,
        shapiro_pvalues=shapiro_p,
        levene_stat=lev_stat,
        levene_pvalue=lev_pv,
        test_name=test_name,
        test_stat=float(stat),
        test_pvalue=float(pv),
        decision_rule=decision_rule,
        n_total=n_total,
        k_groups=k_groups,
        epsilon_sq=None if epsilon_sq is None else round(epsilon_sq, 4),
        epsilon_sq_ci95=None if epsilon_sq_ci95 is None else (round(epsilon_sq_ci95[0], 4), round(epsilon_sq_ci95[1], 4)),
    )

def summarize_group_test(result: GroupTestResult, alpha: float = 0.05) -> str:
    sig = result.test_pvalue < alpha
    decision = "Reject H0" if sig else "Fail to reject H0"

    effect_txt = ""
    eps = getattr(result, "epsilon_sq", None)
    ci = getattr(result, "epsilon_sq_ci95", None)

    if eps is not None:
        if ci is not None:
            lo, hi = ci
            effect_txt = f" Effect size ε²={eps:.4f}, 95% CI=({lo:.4f}, {hi:.4f})."
        else:
            effect_txt = f" Effect size ε²={eps:.4f}."

    n_total = getattr(result, "n_total", None)
    n_total_txt = "—" if n_total is None else str(n_total)

    # Shapiro p-values
    if isinstance(result.shapiro_pvalues, float):
        shapiro_txt = f"{result.shapiro_pvalues:,.4%}"
    else:
        shapiro_txt = {
            group: f"{p:,.4%}" if isinstance(p, float) else p
            for group, p in result.shapiro_pvalues.items()
        }

    return (
        f"{result.test_name} on {result.value_col} by {result.group_col}: "
        f"stat={result.test_stat:,.4f}, p={result.test_pvalue:,.4%}. "
        f"{decision} at α={alpha}. "
        f"(k={result.n_groups}, n={n_total_txt}) "
        f"Assumptions: Levene p={result.levene_pvalue:,.4%}; "
        f"Shapiro p-values per group (NaN=skipped): {shapiro_txt}. "
        f"Rule used: {result.decision_rule}"
        f"{effect_txt}"
    )

def group_descriptives(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return count, mean, median, std, IQR, min, max per group.
    """
    x = df[[group_col, value_col]].dropna().copy()
    x[group_col] = x[group_col].astype(str)

    if order is None:
        order = list(pd.unique(x[group_col]))

    def iqr(s: pd.Series) -> float:
        return float(s.quantile(0.75) - s.quantile(0.25))

    out = (
        x.groupby(group_col, dropna=False)[value_col]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            iqr=iqr,
            min="min",
            max="max",
        )
        .reindex(order)
    )
    return out # type: ignore

def get_group_arrays(
        df: pd.DataFrame, 
        group_col: str,
        value_col: str,
        order: Optional[List[str]] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Returns (labels, list of arrays) in specified order
    """

    x = df[[group_col, value_col]].dropna().copy()
    x[group_col] = x[group_col].astype(str)

    if order is None:
        order = list(pd.unique(x[group_col]))

    arrays = [x.loc[x[group_col] == g, value_col].to_numpy() for g in order]
    return order, arrays


@dataclass(frozen=True)
class KruskalResult:
    H: float
    p: float
    k: int
    n: int
    epsilon_sq: float
    ci95: Tuple[float, float]

# Dunn-Holm post-hoc
def dunn_posthoc_holm(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Pairwise Dunn’s test with Holm correction.
    """

    x = df[[group_col, value_col]].dropna().copy()
    x[group_col] = x[group_col].astype(str)

    # change ordering
    if order is None:
        order = list(pd.unique(x[group_col]))
    x[group_col] = pd.Categorical(x[group_col], categories=order, ordered=True)

    pmat = sp.posthoc_dunn(
        x, val_col=value_col, group_col=group_col, p_adjust="holm"
    )
    # check labels are string
    pmat.index = pmat.index.astype(str)
    pmat.columns = pmat.columns.astype(str)
    return pmat


def significant_pairs_from_pmat(
    pmat: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Dunn p-value matrix and returns a list of significant pairs
    """
    rows = []
    labels = list(pmat.index)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p = float(pmat.iloc[i, j]) # type: ignore
            if np.isfinite(p) and p < alpha:
                rows.append({"group_a": labels[i], "group_b": labels[j], "p_adj": f"{p:.4%}"})
    return pd.DataFrame(rows).sort_values("p_adj", ascending=True, ignore_index=True)