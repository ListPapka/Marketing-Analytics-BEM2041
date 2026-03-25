#!/usr/bin/env python3
"""
Edit DATA_PATH below if your Excel file lives elsewhere.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

_MPL_CFG = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CFG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CFG))

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path("/SECRET/360buy_SurveyData.xlsx")
# Alternative: DATA_PATH = PROJECT_ROOT / "data" / "360buy_SurveyData.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42


SEGMENTATION_VARS = [
    "CusChoice",
    "ConstUp",
    "ReplacReminder",
    "ProdReturn",
    "ProInsuCov",
]
PROFILE_VARS = [
    "CusAgeYr",
    "CusGen",
    "LevEdn",
    "LevIncome",
    "CusAcct",
]
ALL_VARS = list(dict.fromkeys(PROFILE_VARS + SEGMENTATION_VARS))  

VALID_RANGES = {
    "CusGen": (0, 1),
    "CusAcct": (0, 1),
    "LevEdn": (1, 3),
    "LevIncome": (1, 5),
    "CusChoice": (1, 7),
    "ConstUp": (1, 7),
    "ReplacReminder": (1, 7),
    "ProdReturn": (1, 7),
    "ProInsuCov": (1, 7),
}

AGE_MIN, AGE_MAX = 18, 100

K_CANDIDATES = list(range(2, 7))
MIN_CLUSTER_FRAC = 0.05  


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Excel file not found: {path}\n"
            "Set DATA_PATH at the top of this script to your file location."
        )
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    missing_expected = set(ALL_VARS) - set(out.columns)
    if missing_expected:
        raise ValueError(f"Missing expected columns: {sorted(missing_expected)}")

    for c in ALL_VARS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    n0 = len(out)
    dup_mask = out.duplicated()
    n_dup = int(dup_mask.sum())
    if n_dup:
        out = out.loc[~dup_mask].reset_index(drop=True)
        notes.append(f"Removed {n_dup} duplicate row(s); {n0 - n_dup} rows remain.")

    miss_seg = out[SEGMENTATION_VARS].isna().any(axis=1).sum()
    if miss_seg:
        notes.append(
            f"Rows with any missing segmentation variable: {miss_seg} "
            "(these rows excluded from clustering after cleaning invalid codes)."
        )

    invalid_age = (out["CusAgeYr"] < AGE_MIN) | (out["CusAgeYr"] > AGE_MAX)
    n_inv_age = int(invalid_age.sum())
    if n_inv_age:
        out.loc[invalid_age, "CusAgeYr"] = np.nan
        notes.append(
            f"Set {n_inv_age} CusAgeYr value(s) outside [{AGE_MIN}, {AGE_MAX}] to NaN."
        )

    for col, (lo, hi) in VALID_RANGES.items():
        bad = (out[col] < lo) | (out[col] > hi)
        n_bad = int(bad.sum())
        if n_bad:
            out.loc[bad, col] = np.nan
            notes.append(
                f"Set {n_bad} invalid {col} value(s) outside [{lo}, {hi}] to NaN."
            )

    complete_seg = out[SEGMENTATION_VARS].notna().all(axis=1)
    n_drop = int((~complete_seg).sum())
    if n_drop:
        notes.append(
            f"Excluded {n_drop} row(s) with missing/invalid segmentation base; "
            f"{int(complete_seg.sum())} rows used for clustering."
        )
    out_analysis = out.loc[complete_seg].reset_index(drop=True)

    return out_analysis, notes


def descriptive_tables(df: pd.DataFrame) -> None:
    desc = df[ALL_VARS].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"50%": "median"})
    desc.to_csv(TABLES_DIR / "descriptive_numeric_summary.csv")

    freq_rows = []
    for col in ALL_VARS:
        vc = df[col].value_counts(dropna=False).sort_index()
        for val, cnt in vc.items():
            freq_rows.append(
                {
                    "variable": col,
                    "value": val,
                    "count": cnt,
                    "percent": 100.0 * cnt / len(df),
                }
            )
    pd.DataFrame(freq_rows).to_csv(TABLES_DIR / "frequency_all_variables.csv", index=False)

    with pd.ExcelWriter(TABLES_DIR / "descriptive_summary_pack.xlsx") as writer:
        desc.to_excel(writer, sheet_name="numeric_describe")
        pd.DataFrame(freq_rows).to_excel(writer, sheet_name="frequencies", index=False)


def eta_squared_one_way(groups: list[np.ndarray]) -> float:
    all_vals = np.concatenate([g for g in groups if len(g) > 0])
    if len(all_vals) < 2:
        return float("nan")
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        ss_between += len(g) * (np.mean(g) - grand_mean) ** 2
    if ss_total <= 0:
        return float("nan")
    return float(ss_between / ss_total)


def evaluate_kmeans_k(
    X: np.ndarray, k: int, n_init: int = 20
) -> tuple[np.ndarray, dict]:
    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=n_init,
        max_iter=500,
    )
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    _, counts = np.unique(labels, return_counts=True)
    metrics = {
        "k": k,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "cluster_sizes": ",".join(str(int(c)) for c in sorted(counts)),
    }
    return labels, metrics


def choose_k(metrics_df: pd.DataFrame, n_samples: int) -> tuple[int, str]:

    min_required = max(5, int(np.ceil(MIN_CLUSTER_FRAC * n_samples)))
    viable = metrics_df[metrics_df["min_cluster_size"] >= min_required].copy()
    rationale = (
        f"Minimum cluster size constraint: >= {min_required} respondents per cluster "
        f"(max(5, {MIN_CLUSTER_FRAC:.0%} of n))."
    )
    if viable.empty:
        rationale += (
            " No k satisfied the size rule; fell back to best silhouette among all k "
            "(review cluster sizes in outputs — very small segments may be unstable)."
        )
        viable = metrics_df.copy()
    viable = viable.sort_values(
        by=["silhouette", "calinski_harabasz"], ascending=[False, False]
    )
    best_row = viable.iloc[0]
    k_best = int(best_row["k"])
    rationale += (
        f" Selected k={k_best} with silhouette={best_row['silhouette']:.4f}, "
        f"CH={best_row['calinski_harabasz']:.2f}, DB={best_row['davies_bouldin']:.4f}."
    )
    return k_best, rationale


def cluster_profile_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    clusters = sorted(df[label_col].dropna().unique())
    n = len(df)
    for k in clusters:
        sub = df[df[label_col] == k]
        rows.append(
            {
                "cluster": int(k),
                "n": len(sub),
                "pct_sample": 100.0 * len(sub) / n,
            }
        )
        for v in ALL_VARS:
            rows[-1][f"{v}_mean"] = sub[v].mean()
            rows[-1][f"{v}_median"] = sub[v].median()
    return pd.DataFrame(rows)


def plot_dendrogram(X: np.ndarray) -> None:

    Z = linkage(X, method="ward")
    plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=40,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
    )
    plt.title("Hierarchical clustering (Ward) — truncated dendrogram")
    plt.xlabel("Cluster index (or size)")
    plt.ylabel("Ward linkage distance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dendrogram_ward_truncated.png", dpi=200)
    plt.close()


def plot_metric_comparison(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    k = metrics_df["k"].values
    axes[0].bar(k, metrics_df["silhouette"], color="steelblue")
    axes[0].set_title("Silhouette (higher is better)")
    axes[0].set_xticks(k)
    axes[0].set_xlabel("k")
    axes[1].bar(k, metrics_df["calinski_harabasz"], color="seagreen")
    axes[1].set_title("Calinski–Harabasz (higher is better)")
    axes[1].set_xticks(k)
    axes[1].set_xlabel("k")
    axes[2].bar(k, metrics_df["davies_bouldin"], color="coral")
    axes[2].set_title("Davies–Bouldin (lower is better)")
    axes[2].set_xticks(k)
    axes[2].set_xlabel("k")
    plt.suptitle("KMeans internal validity by k (standardized segmentation variables)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_metrics_by_k.png", dpi=200)
    plt.close()


def plot_cluster_heatmap(centroids: pd.DataFrame) -> None:

    M = centroids.values.astype(float)
    row_mean = M.mean(axis=1, keepdims=True)
    row_std = M.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    Z = (M - row_mean) / row_std
    plt.figure(figsize=(10, max(3, 0.5 * len(centroids))))
    im = plt.imshow(Z, aspect="auto", cmap="RdYlBu_r")
    plt.colorbar(im, label="Row z-score (within cluster across attributes)")
    plt.yticks(range(len(centroids)), [f"Cluster {i}" for i in centroids.index])
    plt.xticks(range(len(SEGMENTATION_VARS)), SEGMENTATION_VARS, rotation=45, ha="right")
    plt.title("Cluster centres (segmentation variables)\nrow-standardized for heatmap contrast")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_centres_heatmap_segmentation.png", dpi=200)
    plt.close()


def statistical_tests(df: pd.DataFrame, label_col: str) -> pd.DataFrame:

    results = []
    clusters = sorted(df[label_col].dropna().unique().astype(int))

    # Age: ANOVA (continuous); Kruskal–Wallis shown as sensitivity check
    groups_age = [df.loc[df[label_col] == c, "CusAgeYr"].dropna().values for c in clusters]
    groups_age = [g for g in groups_age if len(g) > 0]
    if len(groups_age) >= 2:
        try:
            f_stat, p_a = stats.f_oneway(*groups_age)
        except ValueError:
            f_stat, p_a = np.nan, np.nan
        results.append(
            {
                "variable": "CusAgeYr",
                "test": "ANOVA (one-way)",
                "statistic": f_stat,
                "p_value": p_a,
            }
        )
        try:
            stat_k, p_k = stats.kruskal(*groups_age)
        except ValueError:
            stat_k, p_k = np.nan, np.nan
        results.append(
            {
                "variable": "CusAgeYr",
                "test": "Kruskal-Wallis (sensitivity)",
                "statistic": stat_k,
                "p_value": p_k,
            }
        )

    kw_vars = ["LevEdn", "LevIncome"] + SEGMENTATION_VARS
    for var in kw_vars:
        groups = [df.loc[df[label_col] == c, var].dropna().values for c in clusters]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat_k, p_k = stats.kruskal(*groups)
        except ValueError:
            stat_k, p_k = np.nan, np.nan
        results.append(
            {
                "variable": var,
                "test": "Kruskal-Wallis",
                "statistic": stat_k,
                "p_value": p_k,
            }
        )

    chi_vars = ["CusGen", "CusAcct"]
    for var in chi_vars:
        tab = pd.crosstab(df[label_col], df[var])
        if tab.size == 0 or tab.shape[1] < 2:
            results.append(
                {
                    "variable": var,
                    "test": "Chi-square",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "skipped (insufficient variation)",
                }
            )
            continue
        try:
            chi2, p_c, _, _ = stats.chi2_contingency(tab)
            results.append(
                {
                    "variable": var,
                    "test": "Chi-square",
                    "statistic": chi2,
                    "p_value": p_c,
                }
            )
        except ValueError as e:
            results.append(
                {
                    "variable": var,
                    "test": "Chi-square",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": str(e),
                }
            )

    return pd.DataFrame(results)


def separation_strength(df: pd.DataFrame, label_col: str, variables: list[str]) -> pd.DataFrame:
    clusters = sorted(df[label_col].dropna().unique().astype(int))
    rows = []
    for var in variables:
        groups = [df.loc[df[label_col] == c, var].dropna().values for c in clusters]
        eta2 = eta_squared_one_way(groups)
        rows.append({"variable": var, "eta_squared": eta2})
    out = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    return out


def main() -> None:
    ensure_dirs()

    print("=" * 72)
    print("360buy — segmentation analysis")
    print("=" * 72)
    raw = load_raw_data(DATA_PATH)
    print(f"Loaded: {DATA_PATH.name} | shape={raw.shape}")
    df_clean, cleaning_notes = clean_data(raw)
    cleaning_log = pd.DataFrame({"cleaning_note": cleaning_notes})
    cleaning_log.to_csv(TABLES_DIR / "cleaning_log.csv", index=False)
    print("\n--- Cleaning log ---")
    for line in cleaning_notes:
        print(" •", line)

    descriptive_tables(df_clean)

    X = df_clean[SEGMENTATION_VARS].values.astype(float)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    plot_dendrogram(X_std)


    metrics_rows = []
    for k in K_CANDIDATES:
        _, m = evaluate_kmeans_k(X_std, k)
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(TABLES_DIR / "cluster_metrics_by_k.csv", index=False)
    plot_metric_comparison(metrics_df)

    n_obs = X_std.shape[0]
    k_chosen, k_rationale = choose_k(metrics_df, n_obs)

    interpretability_note = (
        "Interpretability: solutions with fewer clusters are typically easier to "
        "act on (messaging, product priorities); higher k may improve statistical "
        "fit but can produce thin segments — check minimum cluster sizes in "
        "cluster_metrics_by_k.csv."
    )
    with open(TABLES_DIR / "chosen_k_rationale.txt", "w", encoding="utf-8") as f:
        f.write(k_rationale + "\n\n" + interpretability_note + "\n")


    km_final = KMeans(
        n_clusters=k_chosen,
        random_state=RANDOM_STATE,
        n_init=30,
        max_iter=500,
    )
    labels = km_final.fit_predict(X_std)
    df_out = df_clean.copy()
    df_out["cluster_kmeans"] = labels.astype(int) + 1

    centroids_raw = df_out.groupby("cluster_kmeans")[SEGMENTATION_VARS].mean()
    centroids_raw.to_csv(TABLES_DIR / "cluster_centroids_segmentation_means.csv")
    plot_cluster_heatmap(centroids_raw)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SEGMENTATION_VARS))
    width = 0.8 / k_chosen
    for i, c in enumerate(sorted(df_out["cluster_kmeans"].unique())):
        vals = centroids_raw.loc[c].values
        ax.bar(x + i * width, vals, width, label=f"Cluster {int(c)}")
    ax.set_xticks(x + width * (k_chosen - 1) / 2)
    ax.set_xticklabels(SEGMENTATION_VARS, rotation=30, ha="right")
    ax.set_ylabel("Mean score (original units)")
    ax.set_title("Segmentation variables: cluster means")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_profile_segmentation_bars.png", dpi=200)
    plt.close()

    prof = cluster_profile_table(df_out, "cluster_kmeans")
    prof.to_csv(TABLES_DIR / "cluster_profile_full.csv", index=False)

    seg_cols = []
    for v in SEGMENTATION_VARS:
        seg_cols.extend([f"{v}_mean", f"{v}_median"])
    prof[["cluster", "n", "pct_sample"] + seg_cols].to_csv(
        TABLES_DIR / "cluster_profile_segmentation_only.csv", index=False
    )

    df_out.to_csv(TABLES_DIR / "cleaned_data_with_clusters.csv", index=False)
    sig_df = statistical_tests(df_out, "cluster_kmeans")
    sig_df.to_csv(TABLES_DIR / "cluster_difference_tests.csv", index=False)
    sep_seg = separation_strength(df_out, "cluster_kmeans", SEGMENTATION_VARS)
    sep_demo = separation_strength(
        df_out, "cluster_kmeans", ["CusAgeYr", "LevEdn", "LevIncome", "CusGen", "CusAcct"]
    )
    sep_seg.to_csv(TABLES_DIR / "eta_squared_segmentation_vars.csv", index=False)
    sep_demo.to_csv(TABLES_DIR / "eta_squared_profiling_vars.csv", index=False)

    print("\n--- Chosen k ---")
    print(k_rationale)
    print(interpretability_note)

    print("\n--- Cluster sizes ---")
    sizes = df_out["cluster_kmeans"].value_counts().sort_index()
    for c, s in sizes.items():
        print(f"  Cluster {int(c)}: n={int(s)} ({100.0 * s / len(df_out):.1f}%)")

    print("\n--- Strongest separation (eta^2) among segmentation variables ---")
    print(sep_seg.to_string(index=False))

    machine_lines = []
    machine_lines.append("MACHINE SUMMARY (objective; for interpretation elsewhere)")
    machine_lines.append("=" * 60)
    machine_lines.append(f"Recommended number of clusters (KMeans, heuristic): {k_chosen}")
    machine_lines.append("")
    machine_lines.append("Evaluation metrics by k (standardized inputs):")
    for _, row in metrics_df.iterrows():
        machine_lines.append(
            f"  k={int(row['k'])}: silhouette={row['silhouette']:.4f}, "
            f"CH={row['calinski_harabasz']:.2f}, DB={row['davies_bouldin']:.4f}, "
            f"min_size={int(row['min_cluster_size'])}, sizes=[{row['cluster_sizes']}]"
        )
    machine_lines.append("")
    machine_lines.append("Cluster sizes (final solution):")
    for c, s in sizes.items():
        machine_lines.append(
            f"  Cluster {int(c)}: n={int(s)} ({100.0 * s / len(df_out):.1f}%)"
        )
    machine_lines.append("")
    machine_lines.append(
        "Variables with strongest separation across clusters (segmentation base, eta^2):"
    )
    for _, r in sep_seg.head(5).iterrows():
        machine_lines.append(f"  {r['variable']}: eta^2={r['eta_squared']:.4f}")
    machine_lines.append("")
    machine_lines.append(
        "Provisional neutral labels (use in reports until named substantively):"
    )
    for c in sorted(df_out["cluster_kmeans"].unique()):
        machine_lines.append(f"  Cluster {int(c)}")

    summary_text = "\n".join(machine_lines)
    print("\n" + summary_text)
    with open(TABLES_DIR / "machine_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print("\nOutputs written to:")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
