"""
Compare AIns–RT relationships between ses-T1 (pre-TMS) and ses-T3 (post-TMS).

Inputs (produced by mid_analysisNEWEST.py):
    T1: /Users/braveDP/Desktop/MID_Analysis_Output/roi_analysis/AIns_RT_merged.csv
    T3: /Users/braveDP/Desktop/MID_Analysis_Output_T3/roi_analysis/AIns_RT_merged.csv

Outputs (in /Users/braveDP/Desktop/MID_Compare_T1_T3/):
    - per_subject_slopes.csv      : per-subject, per-condition AIns~RT slope, T1 & T3
    - paired_tests_results.csv    : paired t-test / Wilcoxon results per condition
    - spaghetti_loss5.png         : individual subject lines, loss5 condition
    - spaghetti_by_condition.png  : small-multiples spaghetti plot, all conditions
    - figure_4c_overlay.png       : T1 vs T3 scatter+regression overlay (loss5)
    - figure_4d_overlay.png       : T1 vs T3 bar comparison, all conditions

NOTE: With n=4 paired subjects at T3, treat all p-values as exploratory.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── CONFIG ────────────────────────────────────────────────────────────────
T1_MERGED = Path("/Users/braveDP/Desktop/R_AIns_MID_Analysis_Output/roi/AIns_RT_merged.csv") #editHere
T3_MERGED = Path("/Users/braveDP/Desktop/R_AIns_MID_Analysis_Output_T3/roi/AIns_RT_merged.csv") #editHere

OUT_DIR = Path("/Users/braveDP/Desktop/R_AIns_MID_Compare_T1_T3") #editHere
FIG_DIR = OUT_DIR / "figures"

CONDITIONS = ["gain5", "loss5", "gain1", "loss1", "gain0", "loss0"]
CONDITION_LABELS = {
    "gain5": "+$5", "loss5": "-$5",
    "gain1": "+$1", "loss1": "-$1",
    "gain0": "+$0", "loss0": "-$0",
}
CONDITION_COLORS = {
    "gain5": "#1A9850", "loss5": "#D73027",
    "gain1": "#66BD63", "loss1": "#F46D43",
    "gain0": "#A6D96A", "loss0": "#FDAE61",
}

MIN_N_FOR_REGRESSION = 3  # minimum subjects per session/condition to fit a slope


# ── LOAD + AGGREGATE ─────────────────────────────────────────────────────

def load_subject_condition(csv_path: Path, session_label: str) -> pd.DataFrame:
    """
    Load an AIns_RT_merged.csv and collapse to one row per
    subject x condition (averaging across runs/sessions already done
    upstream, but we re-average here in case of duplicate rows).
    Adds a 'session' column with the given label (T1 / T3).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"subject", "condition", "AIns_mean_beta_ses_avg", "mean_RT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df_subj = (
        df.groupby(["subject", "condition"])[["AIns_mean_beta_ses_avg", "mean_RT"]]
        .mean()
        .reset_index()
    )
    df_subj["session"] = session_label
    return df_subj


def compute_slopes(df_subj: pd.DataFrame, session_label: str) -> pd.DataFrame:
    """
    For each condition, fit AIns_beta ~ mean_RT across subjects (matching
    the original script's Figure 4d regression direction: y = AIns_beta,
    x = mean_RT). Returns one row per condition with slope/intercept/n.

    This is the GROUP-level slope (one number per condition per session),
    used for the Figure 4d-style overlay bar plot.
    """
    rows = []
    for cond in CONDITIONS:
        d = df_subj[df_subj["condition"] == cond].dropna(
            subset=["AIns_mean_beta_ses_avg", "mean_RT"]
        )
        if len(d) < MIN_N_FOR_REGRESSION:
            rows.append({
                "session": session_label, "condition": cond,
                "slope": np.nan, "se": np.nan, "p": np.nan,
                "r": np.nan, "n": len(d),
            })
            continue
        slope, intercept, r, p, se = stats.linregress(d["mean_RT"], d["AIns_mean_beta_ses_avg"])
        rows.append({
            "session": session_label, "condition": cond,
            "slope": slope, "se": se, "p": p, "r": r, "n": len(d),
        })
    return pd.DataFrame(rows)


# ── PAIRED PER-SUBJECT ANALYSIS ──────────────────────────────────────────

def build_paired_table(df_t1: pd.DataFrame, df_t3: pd.DataFrame) -> pd.DataFrame:
    """
    Merge T1 and T3 subject x condition tables on (subject, condition),
    keeping only subjects present in BOTH sessions.

    Returns long-format df with columns:
        subject, condition, session, AIns_mean_beta_ses_avg, mean_RT
    restricted to paired subjects only.
    """
    subs_t1 = set(df_t1["subject"].unique())
    subs_t3 = set(df_t3["subject"].unique())
    paired_subs = subs_t1 & subs_t3

    if not paired_subs:
        warnings.warn("No overlapping subject IDs found between T1 and T3 files.")

    only_t1 = subs_t1 - subs_t3
    only_t3 = subs_t3 - subs_t1
    if only_t1:
        print(f"  Subjects only in T1 (excluded from paired analysis): {sorted(only_t1)}")
    if only_t3:
        print(f"  Subjects only in T3 (excluded from paired analysis): {sorted(only_t3)}")
    print(f"  Paired subjects (n={len(paired_subs)}): {sorted(paired_subs)}")

    combined = pd.concat([df_t1, df_t3], ignore_index=True)
    combined = combined[combined["subject"].isin(paired_subs)]
    return combined


def paired_tests_per_condition(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each condition, run a paired t-test and Wilcoxon signed-rank test
    on AIns_mean_beta_ses_avg (T3 - T1) and on mean_RT (T3 - T1), across
    paired subjects.

    Returns a long-format results table.
    """
    results = []
    for cond in CONDITIONS:
        d = combined[combined["condition"] == cond]
        pivot_beta = d.pivot(index="subject", columns="session", values="AIns_mean_beta_ses_avg")
        pivot_rt = d.pivot(index="subject", columns="session", values="mean_RT")

        for metric_name, pivot in [("AIns_beta", pivot_beta), ("mean_RT", pivot_rt)]:
            pivot = pivot.dropna()
            n = len(pivot)
            if n < 2 or "T1" not in pivot.columns or "T3" not in pivot.columns:
                results.append({
                    "condition": cond, "metric": metric_name, "n_paired": n,
                    "mean_T1": np.nan, "mean_T3": np.nan, "mean_diff_T3_minus_T1": np.nan,
                    "t_stat": np.nan, "t_p": np.nan,
                    "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
                })
                continue

            t1_vals = pivot["T1"].values
            t3_vals = pivot["T3"].values
            diff = t3_vals - t1_vals

            t_stat, t_p = stats.ttest_rel(t3_vals, t1_vals)

            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                # all-zero differences or n too small
                w_stat, w_p = np.nan, np.nan

            results.append({
                "condition": cond, "metric": metric_name, "n_paired": n,
                "mean_T1": np.mean(t1_vals), "mean_T3": np.mean(t3_vals),
                "mean_diff_T3_minus_T1": np.mean(diff),
                "t_stat": t_stat, "t_p": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            })

    return pd.DataFrame(results)


def per_subject_within_session_slopes(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For descriptive purposes: compute, for each subject, the slope of
    AIns_beta ~ mean_RT ACROSS CONDITIONS within each session.

    This gives one slope per subject per session, suitable for a paired
    comparison of "how strongly does AIns track RT" pre vs post TMS.

    Requires >= MIN_N_FOR_REGRESSION conditions with valid data per
    subject x session.
    """
    rows = []
    for (sub, ses), d in combined.groupby(["subject", "session"]):
        d = d.dropna(subset=["AIns_mean_beta_ses_avg", "mean_RT"])
        if len(d) < MIN_N_FOR_REGRESSION:
            rows.append({"subject": sub, "session": ses, "slope": np.nan,
                          "r": np.nan, "p": np.nan, "n_conditions": len(d)})
            continue
        slope, intercept, r, p, se = stats.linregress(d["mean_RT"], d["AIns_mean_beta_ses_avg"])
        rows.append({"subject": sub, "session": ses, "slope": slope,
                      "r": r, "p": p, "n_conditions": len(d)})
    return pd.DataFrame(rows)


# ── PLOTS ─────────────────────────────────────────────────────────────────

def plot_spaghetti_loss5(combined: pd.DataFrame, fig_dir: Path) -> None:
    """
    One line per subject, AIns loss5 beta at T1 -> T3.
    Two panels: AIns beta and mean RT.
    """
    d = combined[combined["condition"] == "loss5"]
    pivot_beta = d.pivot(index="subject", columns="session", values="AIns_mean_beta_ses_avg").dropna()
    pivot_rt = d.pivot(index="subject", columns="session", values="mean_RT").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for ax, pivot, ylabel, title in [
        (axes[0], pivot_beta, "R AIns beta (-$5 anticipation)", "R AIns -$5 beta: T1 vs T3"), #editHere
        (axes[1], pivot_rt, "Mean RT (ms)", "loss5 RT: T1 vs T3"),
    ]:
        for sub, row in pivot.iterrows():
            ax.plot(["T1", "T3"], [row["T1"], row["T3"]], marker="o",
                    color="#4575B4", alpha=0.7, linewidth=1.5)
            ax.text(1.02, row["T3"], sub, fontsize=7, va="center")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.3, 1.5)

    fig.suptitle("Subject-level change, loss5 condition (pre vs post TMS)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "spaghetti_loss5.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'spaghetti_loss5.png'}")


def plot_spaghetti_by_condition(combined: pd.DataFrame, fig_dir: Path) -> None:
    """
    Small-multiples: one subplot per condition, subject lines T1 -> T3
    for AIns_mean_beta_ses_avg.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    axes = axes.flatten()

    for ax, cond in zip(axes, CONDITIONS):
        d = combined[combined["condition"] == cond]
        pivot = d.pivot(index="subject", columns="session", values="AIns_mean_beta_ses_avg").dropna()

        for sub, row in pivot.iterrows():
            ax.plot(["T1", "T3"], [row["T1"], row["T3"]], marker="o",
                    color=CONDITION_COLORS[cond], alpha=0.7, linewidth=1.5)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(CONDITION_LABELS[cond], fontsize=11)
        ax.set_ylabel("AIns beta", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.3, 1.3)

    fig.suptitle("R AIns beta, pre (T1) vs post (T3) TMS — by trial type", fontsize=13) #editHere
    fig.tight_layout()
    fig.savefig(str(fig_dir / "spaghetti_by_condition.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'spaghetti_by_condition.png'}")


def plot_figure_4c_overlay(combined: pd.DataFrame, fig_dir: Path) -> None:
    """
    Overlay T1 and T3 scatter + regression line for loss5
    (AIns_beta on x, RT on y, matching the original figure_4c).
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    colors = {"T1": "#4575B4", "T3": "#D73027"}

    for ses in ["T1", "T3"]:
        d = combined[(combined["condition"] == "loss5") & (combined["session"] == ses)].dropna(
            subset=["AIns_mean_beta_ses_avg", "mean_RT"]
        )
        if d.empty:
            continue

        ax.scatter(d["AIns_mean_beta_ses_avg"], d["mean_RT"],
                    color=colors[ses], s=60, edgecolors="white", linewidths=0.5,
                    label=f"{ses} (n={len(d)})", zorder=3)

        if len(d) >= MIN_N_FOR_REGRESSION:
            slope, intercept, r, p, se = stats.linregress(d["AIns_mean_beta_ses_avg"], d["mean_RT"])
            x_range = np.linspace(d["AIns_mean_beta_ses_avg"].min(),
                                   d["AIns_mean_beta_ses_avg"].max(), 100)
            y_fit = slope * x_range + intercept
            ax.plot(x_range, y_fit, color=colors[ses], linewidth=2, zorder=2,
                    label=f"{ses}: r={r:.2f}, p={p:.3f}")
        else:
            print(f"  Figure 4c overlay: {ses} has only {len(d)} subjects — "
                  f"regression line skipped.")

    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xlabel("R AIns Loss Anticipation Activity\n(-$5 beta)", fontsize=11) #editHere
    ax.set_ylabel("Mean Reaction Time (ms)", fontsize=11)
    ax.set_title("R AIns -$5 anticipation vs. RT\nT1 (pre-TMS) vs T3 (post-TMS)", fontsize=11) #editHere
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "figure_4c_overlay.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'figure_4c_overlay.png'}")


def plot_figure_4d_overlay(slopes_t1: pd.DataFrame, slopes_t3: pd.DataFrame, fig_dir: Path) -> None:
    """
    Grouped bar chart: T1 vs T3 regression slopes (AIns_beta ~ mean_RT)
    for each condition, side by side.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(CONDITIONS))
    width = 0.35

    for offset, (df_s, label, color) in [
        (-width / 2, (slopes_t1, "T1 (pre-TMS)", "#4575B4")),
        (width / 2, (slopes_t3, "T3 (post-TMS)", "#D73027")),
    ][0:0]:
        pass  # placeholder to keep structure clear; real loop below

    for ses_label, df_s, color, offset in [
        ("T1 (pre-TMS)", slopes_t1, "#4575B4", -width / 2),
        ("T3 (post-TMS)", slopes_t3, "#D73027", width / 2),
    ]:
        for xi, cond in enumerate(CONDITIONS):
            row = df_s[df_s["condition"] == cond].iloc[0]
            if np.isnan(row["slope"]):
                continue
            ax.bar(xi + offset, row["slope"], width=width, color=color,
                   yerr=row["se"], capsize=4,
                   error_kw={"linewidth": 1.2, "ecolor": "black"},
                   label=ses_label if xi == 0 else None, zorder=3)
            if not np.isnan(row["p"]) and row["p"] < 0.05:
                ypos = row["slope"] + (row["se"] * 1.3 if row["slope"] >= 0 else -row["se"] * 1.3)
                ax.text(xi + offset, ypos, "*", ha="center",
                        va="bottom" if row["slope"] >= 0 else "top",
                        fontsize=14, color="black")

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=11)
    ax.set_xlabel("Trial Type", fontsize=11)
    ax.set_ylabel("Regression Coefficient\n(R AIns activity ~ Reaction time)", fontsize=11) #editHere
    ax.set_title("R AIns-RT regression slope by trial type: T1 vs T3", fontsize=12) #editHere
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "figure_4d_overlay.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'figure_4d_overlay.png'}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading T1 and T3 merged data...")
    df_t1 = load_subject_condition(T1_MERGED, "T1")
    df_t3 = load_subject_condition(T3_MERGED, "T3")
    print(f"  T1: {df_t1['subject'].nunique()} subjects, "
          f"T3: {df_t3['subject'].nunique()} subjects")

    print("\nBuilding paired subject table (subjects present in both T1 & T3)...")
    combined = build_paired_table(df_t1, df_t3)
    combined.to_csv(OUT_DIR / "paired_subject_condition_data.csv", index=False)

    print("\nRunning paired tests (T3 vs T1) per condition...")
    paired_results = paired_tests_per_condition(combined)
    paired_results.to_csv(OUT_DIR / "paired_tests_results.csv", index=False)
    print(paired_results.to_string(index=False))

    print("\nComputing per-subject AIns~RT slopes (across conditions), per session...")
    subj_slopes = per_subject_within_session_slopes(combined)
    subj_slopes.to_csv(OUT_DIR / "per_subject_slopes.csv", index=False)
    print(subj_slopes.to_string(index=False))

    # Paired test on per-subject slopes (descriptive headline number)
    slope_pivot = subj_slopes.pivot(index="subject", columns="session", values="slope").dropna()
    if len(slope_pivot) >= 2 and "T1" in slope_pivot.columns and "T3" in slope_pivot.columns:
        diff = slope_pivot["T3"] - slope_pivot["T1"]
        t_stat, t_p = stats.ttest_rel(slope_pivot["T3"], slope_pivot["T1"])
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        print(f"\nPer-subject AIns~RT slope, T3 vs T1 (n={len(slope_pivot)}):")
        print(f"  mean T1 slope = {slope_pivot['T1'].mean():.4f}")
        print(f"  mean T3 slope = {slope_pivot['T3'].mean():.4f}")
        print(f"  paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
        print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p}")
    else:
        print("\nNot enough subjects with valid slopes in both sessions for a paired slope test.")

    print("\nGenerating group-level regression slopes (Figure 4d-style) for each session...")
    slopes_t1 = compute_slopes(df_t1, "T1")
    slopes_t3 = compute_slopes(df_t3, "T3")
    slopes_t1.to_csv(OUT_DIR / "group_slopes_T1.csv", index=False)
    slopes_t3.to_csv(OUT_DIR / "group_slopes_T3.csv", index=False)

    print("\nGenerating plots...")
    plot_spaghetti_loss5(combined, FIG_DIR)
    plot_spaghetti_by_condition(combined, FIG_DIR)
    plot_figure_4c_overlay(combined, FIG_DIR)
    plot_figure_4d_overlay(slopes_t1, slopes_t3, FIG_DIR)

    print(f"\nDone. Outputs in: {OUT_DIR}")
    print("\nReminder: with n<=4 paired subjects at T3, treat all p-values as "
          "exploratory/preliminary only.")


if __name__ == "__main__":
    main()
