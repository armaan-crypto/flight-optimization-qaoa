"""
analyze.py
==========
Reads results.csv produced by testing.py and generates publication-quality
Matplotlib figures for the research paper.

Figures produced
----------------
  fig1_route_cost.png         Total route cost: QAOA vs A* across network sizes
  fig2_optimality_gap.png     Optimality gap (%) across network sizes
  fig3_feasibility.png        Feasibility index comparison
  fig4_runtime_scaling.png    Runtime scaling curves (log scale)
  fig5_nodes_vs_iters.png     Nodes expanded (A*) / candidates ranked (QAOA)
  fig6_fare_error.png         Fare difference QAOA vs A* by route
  fig7_summary_table.png      Key metrics summary table
  fig8_gap_heatmap.png        Optimality gap heatmap: route × network size
  fig9_success_by_category.png Success rate by route category (fixed/stress/random)
  fig10_cost_violin.png       Cost distribution violin plots by network size
  fig11_timeout_rate.png      Timeout rate by solver and network size

Run
---
  python analyze.py [--csv results.csv] [--out figures/]

Dependencies
------------
  pip install pandas matplotlib numpy scipy
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "astar":      "#2166AC",
    "qaoa":       "#D6604D",
    "astar_fill": "#92C5DE",
    "qaoa_fill":  "#F4A582",
    "grid":       "#DDDDDD",
    "subtext":    "#555555",
}

mpl.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,
    "lines.linewidth":       1.6,
    "lines.markersize":      6,
    "axes.linewidth":        0.8,
    "axes.grid":             True,
    "grid.color":            PALETTE["grid"],
    "grid.linewidth":        0.5,
    "grid.linestyle":        "--",
    "legend.framealpha":     0.92,
    "legend.edgecolor":      "#CCCCCC",
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.08,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
})

SOLVER_LABEL = {"astar": "A*", "qaoa": "QAOA"}
SOLVER_COLOR = {"astar": PALETTE["astar"], "qaoa": PALETTE["qaoa"]}
SOLVER_FILL  = {"astar": PALETTE["astar_fill"], "qaoa": PALETTE["qaoa_fill"]}
MARKERS      = {"astar": "o", "qaoa": "s"}

FIG_W_SINGLE = 5.5
FIG_W_DOUBLE = 7.5

FULL_SENTINEL = -1   # stored in CSV when no subsampling was done


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_results(csv_path):
    df = pd.read_csv(csv_path)

    # Coerce all numeric columns safely
    for col in ["trial_id", "total_route_cost", "total_fare_usd", "feasibility",
                "runtime_seconds", "nodes_expanded", "iterations",
                "num_legs", "optimality_gap", "timed_out"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # network_size: stored as int (-1 for full graph)
    if "network_size" in df.columns:
        df["network_size"] = pd.to_numeric(df["network_size"], errors="coerce").astype("Int64")

    # Guard optional columns
    if "optimality_gap" not in df.columns:
        df["optimality_gap"] = np.nan
    if "route_category" not in df.columns:
        df["route_category"] = "fixed"
    if "timed_out" not in df.columns:
        df["timed_out"] = 0

    df["optimality_gap_pct"] = df["optimality_gap"] * 100

    # Human-readable size label for axis ticks
    df["size_label"] = df["network_size"].apply(
        lambda x: "full" if x == FULL_SENTINEL else str(int(x))
    )

    # Ordered categorical so plots sort correctly (50 < 100 < … < full)
    unique_sizes  = sorted(df["network_size"].dropna().unique())
    size_labels   = ["full" if s == FULL_SENTINEL else str(int(s)) for s in unique_sizes]
    df["size_ord"] = pd.Categorical(
        df["size_label"], categories=size_labels, ordered=True
    )

    df["route"] = df["origin"] + "→" + df["destination"]

    # Drop rows that timed out from quality metrics (keep for timeout figure)
    df["timed_out"] = df["timed_out"].fillna(0)

    return df, unique_sizes, size_labels


def ci95(std, n):
    n = np.maximum(n, 1)
    return 1.96 * std / np.sqrt(n)


def _solver_line(ax, grp, solver, x_col, y_col, std_col, n_col):
    """Plot mean line + CI band for one solver. x_col must already be numeric."""
    s = grp[grp["solver"] == solver].sort_values(x_col)
    if s.empty:
        return
    x = np.arange(len(s))          # evenly spaced positions
    y = s[y_col].values
    e = ci95(s[std_col].values, s[n_col].values)
    ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver],
            label=SOLVER_LABEL[solver], zorder=3)
    ax.fill_between(x, y - e, y + e, alpha=0.18,
                    color=SOLVER_FILL[solver], zorder=2)
    return s[x_col].values   # return tick labels


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — Total Route Cost
# ══════════════════════════════════════════════════════════════════════════════

def fig_route_cost(df, size_labels, out_dir):
    clean = df[df["timed_out"] == 0].copy()
    grp = (
        clean.groupby(["solver", "size_ord"], observed=True)["total_route_cost"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    grp["ci"] = ci95(grp["std"], grp["count"])

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.8))
    tick_labels = None
    for solver in ["astar", "qaoa"]:
        s = grp[grp["solver"] == solver]
        if s.empty:
            continue
        x = np.arange(len(s))
        y = s["mean"].values
        e = s["ci"].values
        ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver],
                label=SOLVER_LABEL[solver], zorder=3)
        ax.fill_between(x, y - e, y + e, alpha=0.18,
                        color=SOLVER_FILL[solver], zorder=2)
        if tick_labels is None:
            tick_labels = s["size_ord"].astype(str).tolist()

    if tick_labels:
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Network Size (airports)")
    ax.set_ylabel("Mean Total Route Cost\n(sigmoid score, lower = better)")
    ax.set_title("Route Cost: QAOA vs. A* Across Network Sizes")
    ax.legend(title="Solver")
    savefig(fig, os.path.join(out_dir, "fig1_route_cost.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Optimality Gap
# ══════════════════════════════════════════════════════════════════════════════

def fig_optimality_gap(df, size_labels, out_dir):
    qaoa = df[(df["solver"] == "qaoa") & df["optimality_gap_pct"].notna()
              & (df["timed_out"] == 0)].copy()
    if qaoa.empty:
        print("  [SKIP] fig2: no optimality gap data.")
        return

    grp = (
        qaoa.groupby("size_ord", observed=True)["optimality_gap_pct"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    grp["ci"] = ci95(grp["std"], grp["count"])

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.8))
    x = np.arange(len(grp))
    y = grp["mean"].values
    e = grp["ci"].values

    ax.plot(x, y, marker="s", color=PALETTE["qaoa"], zorder=3, label="Mean gap")
    ax.fill_between(x, y - e, y + e, alpha=0.18,
                    color=PALETTE["qaoa_fill"], zorder=2, label="95 % CI")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="A* optimum")

    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp["size_ord"].astype(str).tolist())
    ax.set_xlabel("Network Size (airports)")
    ax.set_ylabel("Optimality Gap (%)\n[(QAOA − A*) / A*] × 100")
    ax.set_title("QAOA Optimality Gap vs. A* Across Network Sizes")
    ax.legend()
    savefig(fig, os.path.join(out_dir, "fig2_optimality_gap.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Feasibility Index
# ══════════════════════════════════════════════════════════════════════════════

def fig_feasibility(df, size_labels, out_dir):
    # Include timed-out rows as infeasible (feasibility=0)
    grp = (
        df.groupby(["solver", "size_ord"], observed=True)["feasibility"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    grp["ci"] = ci95(grp["std"], grp["count"])

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.8))
    tick_labels = None
    for solver in ["astar", "qaoa"]:
        s = grp[grp["solver"] == solver]
        if s.empty:
            continue
        x = np.arange(len(s))
        y = s["mean"].values
        e = s["ci"].values
        ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver],
                label=SOLVER_LABEL[solver], zorder=3)
        ax.fill_between(x, np.clip(y-e, 0, 1), np.clip(y+e, 0, 1),
                        alpha=0.18, color=SOLVER_FILL[solver], zorder=2)
        if tick_labels is None:
            tick_labels = s["size_ord"].astype(str).tolist()

    if tick_labels:
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)

    ax.set_ylim(-0.05, 1.10)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Network Size (airports)")
    ax.set_ylabel("Feasibility Index\n(proportion of trials with valid route)")
    ax.set_title("Feasibility Index: QAOA vs. A*")
    ax.legend(title="Solver")
    savefig(fig, os.path.join(out_dir, "fig3_feasibility.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Runtime Scaling
# ══════════════════════════════════════════════════════════════════════════════

def fig_runtime(df, size_labels, out_dir):
    clean = df[(df["timed_out"] == 0) & df["runtime_seconds"].notna()].copy()
    # Use numeric network_size for regression; exclude full (-1)
    clean_reg = clean[clean["network_size"] != FULL_SENTINEL].copy()

    grp = (
        clean.groupby(["solver", "size_ord"], observed=True)["runtime_seconds"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    grp["ci"] = ci95(grp["std"], grp["count"])

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.8))
    tick_labels = None

    for solver in ["astar", "qaoa"]:
        s = grp[grp["solver"] == solver]
        if s.empty:
            continue
        x = np.arange(len(s))
        y = s["mean"].values
        e = s["ci"].values
        ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver],
                label=SOLVER_LABEL[solver], zorder=3)
        ax.fill_between(x, np.maximum(y-e, 1e-6), y+e, alpha=0.18,
                        color=SOLVER_FILL[solver], zorder=2)
        if tick_labels is None:
            tick_labels = s["size_ord"].astype(str).tolist()

        # Power-law fit on numeric sizes only
        if HAS_SCIPY:
            reg_s = (
                clean_reg[clean_reg["solver"] == solver]
                .groupby("network_size")["runtime_seconds"]
                .mean().reset_index()
            )
            if len(reg_s) >= 3 and (reg_s["runtime_seconds"] > 0).all():
                lx = np.log(reg_s["network_size"].values.astype(float))
                ly = np.log(reg_s["runtime_seconds"].values)
                slope, intercept, *_ = scipy_stats.linregress(lx, ly)
                # Map fitted curve onto the same evenly-spaced x positions
                # (only for the non-"full" points)
                n_numeric = len(reg_s)
                x_fit = np.linspace(0, n_numeric - 1, 200)
                y_fit = np.exp(intercept) * np.exp(slope * np.linspace(lx.min(), lx.max(), 200))
                ax.plot(x_fit, y_fit, linestyle="--", linewidth=0.9,
                        color=SOLVER_COLOR[solver], alpha=0.55,
                        label=f"{SOLVER_LABEL[solver]} fit (α={slope:.2f})")

    if tick_labels:
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)

    ax.set_yscale("log")
    ax.set_xlabel("Network Size (airports)")
    ax.set_ylabel("Mean Runtime (seconds, log scale)")
    ax.set_title("Runtime Scaling: QAOA vs. A*")
    ax.legend(fontsize=8)
    savefig(fig, os.path.join(out_dir, "fig4_runtime_scaling.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Search Effort
# ══════════════════════════════════════════════════════════════════════════════

def fig_search_effort(df, size_labels, out_dir):
    astar_df = df[(df["solver"] == "astar") & df["nodes_expanded"].notna()
                  & (df["timed_out"] == 0)]
    qaoa_df  = df[(df["solver"] == "qaoa")  & df["iterations"].notna()
                  & (df["timed_out"] == 0)]

    if astar_df.empty and qaoa_df.empty:
        print("  [SKIP] fig5: no search effort data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, 3.8))

    def _panel(ax, data, col, solver, ylabel, title):
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["subtext"])
            ax.set_title(title)
            return
        grp = (
            data.groupby("size_ord", observed=True)[col]
            .agg(["mean", "std", "count"]).reset_index()
        )
        grp["ci"] = ci95(grp["std"], grp["count"])
        x = np.arange(len(grp))
        y = grp["mean"].values
        e = grp["ci"].values
        ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver], zorder=3)
        ax.fill_between(x, np.maximum(y-e, 0.5), y+e, alpha=0.18,
                        color=SOLVER_FILL[solver], zorder=2)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp["size_ord"].astype(str).tolist(), fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("Network Size (airports)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    _panel(axes[0], astar_df, "nodes_expanded", "astar",
           "Nodes Expanded (log)", "A* — Search Effort")
    _panel(axes[1], qaoa_df,  "iterations",     "qaoa",
           "Candidates Ranked (log)", "QAOA — Ranking Effort")

    fig.suptitle("Search Effort vs. Network Size", fontsize=11, y=1.01)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig5_nodes_vs_iters.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Fare Error Proxy
# ══════════════════════════════════════════════════════════════════════════════

def fig_fare_error(df, size_labels, out_dir):
    fare_errors_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)),
                                    "fare_errors.csv")
    clean = df[df["timed_out"] == 0]

    if os.path.exists(fare_errors_path):
        err = pd.read_csv(fare_errors_path)
        routes = err["route"].tolist()
        rmse   = err["rmse"].values
        mae    = err["mae"].values
        title  = "LSTM Fare Prediction Error by Route\n(RMSE and MAE on held-out test set)"
        ylabel = "Fare Prediction Error (USD)"
        note   = None
    else:
        pivot = clean.pivot_table(
            index=["route", "network_size", "trial_id"],
            columns="solver", values="total_fare_usd",
        ).dropna()
        if pivot.empty:
            print("  [SKIP] fig6: no paired fare data.")
            return
        pivot["abs_diff"] = (pivot["qaoa"] - pivot["astar"]).abs()
        grp = pivot.groupby("route")["abs_diff"].agg(
            rmse=lambda x: np.sqrt((x**2).mean()),
            mae="mean",
        ).reset_index().sort_values("rmse", ascending=False)
        routes = grp["route"].tolist()
        rmse   = grp["rmse"].values
        mae    = grp["mae"].values
        title  = "Routing-Induced Fare Spread by Route\n(QAOA vs. A* absolute fare difference)"
        ylabel = "Absolute Fare Difference (USD)"
        note   = "Place LSTM test-set errors in fare_errors.csv to replace this proxy."

    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.4 * len(routes) + 2), 4.2))
    x = np.arange(len(routes))
    w = 0.35
    ax.bar(x - w/2, rmse, w, label="RMSE", color=PALETTE["astar"], alpha=0.85)
    ax.bar(x + w/2, mae,  w, label="MAE",  color=PALETTE["qaoa"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if note:
        ax.text(0.99, 0.97, note, transform=ax.transAxes, fontsize=7,
                va="top", ha="right", color=PALETTE["subtext"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))
    savefig(fig, os.path.join(out_dir, "fig6_fare_error.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 7 — Summary Table
# ══════════════════════════════════════════════════════════════════════════════

def fig_summary_table(df, size_labels, out_dir):
    clean = df[df["timed_out"] == 0]
    grp = clean.groupby(["solver", "size_ord"], observed=True).agg(
        cost_mean=("total_route_cost", "mean"),
        cost_std =("total_route_cost", "std"),
        fare_mean=("total_fare_usd",   "mean"),
        feas_mean=("feasibility",      "mean"),
        rt_mean  =("runtime_seconds",  "mean"),
        rt_std   =("runtime_seconds",  "std"),
        n        =("total_route_cost", "count"),
    ).reset_index()

    gap_grp = (
        df[(df["solver"] == "qaoa") & df["optimality_gap_pct"].notna()]
        .groupby("size_ord", observed=True)["optimality_gap_pct"]
        .mean().reset_index().rename(columns={"optimality_gap_pct": "gap"})
    )
    grp = grp.merge(gap_grp, on="size_ord", how="left")
    grp.loc[grp["solver"] == "astar", "gap"] = 0.0

    rows = []
    for _, r in grp.iterrows():
        gap_s = (f"{r['gap']:+.2f}%" if pd.notna(r["gap"]) and r["solver"] == "qaoa" else "—")
        cs    = f"±{r['cost_std']:.4f}" if pd.notna(r.get("cost_std")) else ""
        rs    = f"±{r['rt_std']:.3f}"   if pd.notna(r.get("rt_std"))  else ""
        rows.append([
            SOLVER_LABEL.get(r["solver"], r["solver"]),
            str(r["size_ord"]),
            f"{r['cost_mean']:.4f}{cs}",
            f"${r['fare_mean']:.2f}",
            f"{r['feas_mean']:.3f}",
            f"{r['rt_mean']:.3f}s{rs}",
            gap_s,
            str(int(r["n"])),
        ])

    col_labels = ["Solver", "Size", "Route Cost (μ±σ)", "Fare (μ)",
                  "Feasibility", "Runtime (μ±σ)", "Opt. Gap", "n"]

    fig_h = max(2.5, 0.32 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE + 2, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2166AC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, row_data in enumerate(rows):
        bg = "#EAF2FB" if row_data[0] == "A*" else "#FEF0EC"
        for j in range(len(col_labels)):
            tbl[i+1, j].set_facecolor(bg)

    ax.set_title("Summary Metrics: QAOA vs. A*", fontsize=10,
                 pad=8, fontweight="bold")
    savefig(fig, os.path.join(out_dir, "fig7_summary_table.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 8 — Optimality Gap Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_gap_heatmap(df, size_labels, out_dir):
    qaoa = df[(df["solver"] == "qaoa") & df["optimality_gap_pct"].notna()
              & (df["timed_out"] == 0)]
    if qaoa.empty:
        print("  [SKIP] fig8: no gap data.")
        return

    pivot = qaoa.pivot_table(
        index="route", columns="size_ord",
        values="optimality_gap_pct", aggfunc="mean",
        observed=True,
    )
    if pivot.empty:
        print("  [SKIP] fig8: pivot empty.")
        return

    # Limit rows to 40 for readability; sort by mean gap descending
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
    if len(pivot) > 40:
        pivot = pivot.iloc[:40]

    fig_h = max(3.5, 0.28 * len(pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, fig_h))

    vals  = pivot.values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmax  = max(abs(finite).max(), 0.01) if len(finite) else 1.0
    im    = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Network Size (airports)")
    ax.set_ylabel("Route")
    ax.set_title("QAOA Optimality Gap (%) — Route × Network Size\n"
                 "(red = QAOA worse than A*; top 40 routes by mean gap)")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > vmax * 0.55 else "black")

    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="Optimality Gap (%)")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig8_gap_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 9 — Success Rate by Route Category
# ══════════════════════════════════════════════════════════════════════════════

def fig_success_by_category(df, size_labels, out_dir):
    """
    Fraction of trials returning a feasible route, broken out by route category
    (fixed / stress / random) and solver.  Shows whether QAOA degrades faster
    on hard routes than A* does.
    """
    cats = [c for c in ["fixed", "stress", "random"]
            if c in df["route_category"].unique()]
    if not cats:
        print("  [SKIP] fig9: no route_category data.")
        return

    fig, axes = plt.subplots(1, len(cats),
                             figsize=(FIG_W_DOUBLE * len(cats) / 2, 3.8),
                             sharey=True)
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        sub = df[df["route_category"] == cat]
        grp = (
            sub.groupby(["solver", "size_ord"], observed=True)["feasibility"]
            .agg(["mean", "std", "count"]).reset_index()
        )
        grp["ci"] = ci95(grp["std"], grp["count"])
        tick_labels = None

        for solver in ["astar", "qaoa"]:
            s = grp[grp["solver"] == solver]
            if s.empty:
                continue
            x = np.arange(len(s))
            y = s["mean"].values
            e = s["ci"].values
            ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver],
                    label=SOLVER_LABEL[solver], zorder=3)
            ax.fill_between(x, np.clip(y-e, 0, 1), np.clip(y+e, 0, 1),
                            alpha=0.18, color=SOLVER_FILL[solver], zorder=2)
            if tick_labels is None:
                tick_labels = s["size_ord"].astype(str).tolist()

        if tick_labels:
            ax.set_xticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels, fontsize=8)

        ax.set_ylim(-0.05, 1.10)
        ax.set_title(f"Category: {cat}", fontsize=10)
        ax.set_xlabel("Network Size (airports)")
        ax.legend(title="Solver", fontsize=8)

    axes[0].set_ylabel("Feasibility Rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.suptitle("Route Success Rate by Category and Network Size",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig9_success_by_category.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 10 — Cost Distribution Violins
# ══════════════════════════════════════════════════════════════════════════════

def fig_cost_violin(df, size_labels, out_dir):
    clean = df[df["timed_out"] == 0]
    sizes = sorted(clean["size_ord"].dropna().unique())   # already categorical
    if not sizes:
        return

    # Cap at 8 panels for readability
    if len(sizes) > 8:
        sizes = sizes[:8]

    fig, axes = plt.subplots(
        1, len(sizes),
        figsize=(max(FIG_W_DOUBLE, 1.8 * len(sizes)), 3.8),
        sharey=True,
    )
    if len(sizes) == 1:
        axes = [axes]

    for ax, sz in zip(axes, sizes):
        sub = clean[clean["size_ord"] == sz]
        data_by_solver, labels, colors = [], [], []
        for solver in ["astar", "qaoa"]:
            vals = sub[sub["solver"] == solver]["total_route_cost"].dropna().values
            if len(vals) >= 3:          # violinplot needs ≥ 3 points
                data_by_solver.append(vals)
                labels.append(SOLVER_LABEL[solver])
                colors.append(SOLVER_COLOR[solver])
            elif len(vals) > 0:         # fall back to a strip
                ax.scatter(
                    [["astar", "qaoa"].index(solver) + 1] * len(vals),
                    vals, color=SOLVER_COLOR[solver], s=18, alpha=0.7, zorder=3,
                )

        if data_by_solver:
            parts = ax.violinplot(data_by_solver, showmedians=True, showextrema=True)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            for pn in ("cbars", "cmins", "cmaxes", "cmedians"):
                if pn in parts:
                    parts[pn].set_edgecolor("#333333")
                    parts[pn].set_linewidth(0.8)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"n = {sz}", fontsize=9)
        ax.grid(True, axis="y", linewidth=0.4)

    axes[0].set_ylabel("Total Route Cost (sigmoid score)")
    fig.suptitle("Route Cost Distribution: QAOA vs. A* by Network Size",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig10_cost_violin.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 11 — Timeout Rate
# ══════════════════════════════════════════════════════════════════════════════

def fig_timeout_rate(df, size_labels, out_dir):
    if "timed_out" not in df.columns or df["timed_out"].sum() == 0:
        print("  [SKIP] fig11: no timeouts recorded.")
        return

    grp = (
        df.groupby(["solver", "size_ord"], observed=True)["timed_out"]
        .agg(["mean", "count"]).reset_index()
    )

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.8))
    tick_labels = None

    for solver in ["astar", "qaoa"]:
        s = grp[grp["solver"] == solver]
        if s.empty:
            continue
        x = np.arange(len(s))
        y = s["mean"].values
        ax.plot(x, y, marker=MARKERS[solver], color=SOLVER_COLOR[solver],
                label=SOLVER_LABEL[solver], zorder=3)
        if tick_labels is None:
            tick_labels = s["size_ord"].astype(str).tolist()

    if tick_labels:
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)

    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Network Size (airports)")
    ax.set_ylabel("Timeout Rate")
    ax.set_title(f"Task Timeout Rate by Solver and Network Size\n"
                 f"(timed-out = task killed after hard deadline)")
    ax.legend(title="Solver")
    savefig(fig, os.path.join(out_dir, "fig11_timeout_rate.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: {args.csv} not found. Run testing.py first.")

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.csv} …")
    df, unique_sizes, size_labels = load_results(args.csv)
    print(f"  {len(df)} rows loaded.")
    print(f"  Solvers:    {sorted(df['solver'].unique())}")
    print(f"  Sizes:      {size_labels}")
    cats = df['route_category'].unique().tolist() if 'route_category' in df.columns else []
    print(f"  Categories: {cats}")
    print(f"  Routes:     {df['route'].nunique()} unique")
    n_to = int(df['timed_out'].sum()) if 'timed_out' in df.columns else 0
    print(f"  Timeouts:   {n_to}")
    print(f"  Output →    {args.out}/\n")

    steps = [
        ("Fig 1  — Route cost",               fig_route_cost),
        ("Fig 2  — Optimality gap",            fig_optimality_gap),
        ("Fig 3  — Feasibility index",         fig_feasibility),
        ("Fig 4  — Runtime scaling",           fig_runtime),
        ("Fig 5  — Search effort",             fig_search_effort),
        ("Fig 6  — Fare error",                fig_fare_error),
        ("Fig 7  — Summary table",             fig_summary_table),
        ("Fig 8  — Gap heatmap",               fig_gap_heatmap),
        ("Fig 9  — Success by category",       fig_success_by_category),
        ("Fig 10 — Cost violins",              fig_cost_violin),
        ("Fig 11 — Timeout rate",              fig_timeout_rate),
    ]

    for label, fn in steps:
        print(f"{label} …")
        try:
            fn(df, size_labels, args.out)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    print(f"\nAll figures written to {args.out}/")


if __name__ == "__main__":
    main()