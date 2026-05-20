"""Recompute downstream accuracy artefacts using the LLM-reviewed
manual_correct column. Replaces the heuristic-based numbers in:

  outputs/study2b_steering/study1_accuracy_by_population.csv
  outputs/study2b_steering/accuracy_by_condition.csv
  outputs/study2b_steering/accuracy_vs_study1_baseline.csv
  outputs/study2b_steering/mechanistic_grounding_classification.csv
  outputs/study2b_steering/figures/accuracy_comparison_study1_baseline.png

Reads:
  outputs/study2b_steering/study1_accuracy_heuristic.csv (for n_total/completed)
  outputs/study2b_steering/accuracy_heuristic.csv        (for n_total/completed)
  outputs/study2b_steering/study1_accuracy_manual_review.csv (LLM scores)
  outputs/study2b_steering/accuracy_manual_review.csv        (LLM scores)
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "study2b_steering"
FIG = OUT / "figures"

CONDITION_ORDER = [
    "baseline",
    "HYPO_neg_1.0", "HYPO_neg_0.5", "HYPO_pos_0.5", "HYPO_pos_1.0",
    "TEST_neg_1.0", "TEST_neg_0.5", "TEST_pos_0.5", "TEST_pos_1.0",
    "JUDGE_neg_2.0", "JUDGE_neg_1.0", "JUDGE_neg_0.5", "JUDGE_pos_0.5", "JUDGE_pos_1.0",
    "MONITOR_neg_1.0", "MONITOR_neg_0.5", "MONITOR_pos_0.5", "MONITOR_pos_1.0",
    "PLAN_neg_1.0", "PLAN_neg_0.5", "PLAN_pos_0.5", "PLAN_pos_1.0",
]
STEER_CATEGORIES = ["HYPO", "TEST", "JUDGE", "MONITOR", "PLAN"]
PROBE_F1_THRESHOLD = 0.6


def odds_ratio_ci(a, b, c, d):
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_hat = (aa * dd) / (bb * cc)
    log_or = math.log(or_hat)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    z = 1.959963984540054
    return or_hat, math.exp(log_or - z * se), math.exp(log_or + z * se)


def sig_stars(p):
    if p != p:
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def merge_review_into_heuristic(heur_df, review_df, key_cols):
    """Add a `manual_correct` column to heur_df by joining on key_cols."""
    review = review_df[key_cols + ["manual_correct"]].copy()
    merged = heur_df.merge(review, on=key_cols, how="left")
    return merged


def s1_population(df):
    """Build study1_accuracy_by_population.csv from merged dataframe."""
    populations = ["set_a", "set_b_standard", "set_b_extended", "set_b_all"]
    rows = []
    for pop in populations:
        if pop == "set_b_all":
            sub = df[df["set"].str.lower() == "b"]
        else:
            sub = df[df["population"] == pop]
        n_total = len(sub)
        comp = sub[sub["completed"]]
        n_completed = len(comp)
        n_correct = float(comp["manual_correct"].fillna(0).sum())
        rows.append({
            "population": pop,
            "n_total": n_total,
            "n_completed": n_completed,
            "completion_rate": n_completed / n_total if n_total else float("nan"),
            "n_correct": n_correct,
            "accuracy_among_completed": n_correct / n_completed if n_completed else float("nan"),
            "accuracy_overall": n_correct / n_total if n_total else float("nan"),
        })
    return pd.DataFrame(rows)


def s2b_by_condition(df):
    rows = []
    for cond, g in df.groupby("condition"):
        comp = g[g["completed"]]
        n_comp = len(comp)
        n_correct = float(comp["manual_correct"].fillna(0).sum())
        rows.append({
            "condition": cond,
            "n_total": len(g),
            "n_completed": n_comp,
            "completion_rate": n_comp / len(g) if len(g) else float("nan"),
            "n_correct_llm_review": n_correct,
            "accuracy_among_completed": n_correct / n_comp if n_comp else float("nan"),
            "accuracy_overall": n_correct / len(g) if len(g) else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["_sort"] = out["condition"].apply(lambda c: "0" if c == "baseline" else "1" + c)
    out = out.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return out


def fisher_table(df_s2b, set_a_n_correct, set_a_n_completed):
    set_a_n_incorrect = set_a_n_completed - set_a_n_correct
    set_a_rate = set_a_n_correct / set_a_n_completed if set_a_n_completed else float("nan")
    rows = []
    for cond in CONDITION_ORDER:
        sub = df_s2b[(df_s2b["condition"] == cond) & (df_s2b["completed"])]
        n_completed = len(sub)
        n_correct = int(round(sub["manual_correct"].fillna(0).sum()))
        n_incorrect = n_completed - n_correct
        rate = n_correct / n_completed if n_completed else float("nan")

        if n_completed == 0:
            fisher_p = float("nan"); or_hat, lo, hi = (float("nan"),) * 3
        else:
            tbl = [[n_correct, n_incorrect], [set_a_n_correct, set_a_n_incorrect]]
            try:
                _, fisher_p = fisher_exact(tbl, alternative="two-sided")
                or_hat, lo, hi = odds_ratio_ci(n_correct, n_incorrect, set_a_n_correct, set_a_n_incorrect)
            except Exception:
                fisher_p = float("nan"); or_hat, lo, hi = (float("nan"),) * 3

        if rate == rate and set_a_rate == set_a_rate:
            direction = "higher" if rate > set_a_rate else "lower" if rate < set_a_rate else "equal"
        else:
            direction = "n/a"

        rows.append({
            "condition": cond,
            "n_completed": n_completed,
            "n_correct": n_correct,
            "accuracy_rate": rate,
            "set_a_n_completed": set_a_n_completed,
            "set_a_n_correct": set_a_n_correct,
            "set_a_accuracy_rate": set_a_rate,
            "fisher_p": fisher_p,
            "odds_ratio": or_hat,
            "ci_lower": lo,
            "ci_upper": hi,
            "direction": direction,
        })
    return pd.DataFrame(rows)


def plot_accuracy_comparison(pop_df, cmp_df):
    try:
        from statsmodels.stats.proportion import proportion_confint
        def wilson(k, n):
            if n == 0: return 0.0, 0.0
            return proportion_confint(k, n, alpha=0.05, method="wilson")
    except ImportError:
        def wilson(k, n):
            if n == 0: return 0.0, 0.0
            p = k / n; z = 1.959963984540054
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
            return max(0.0, centre - half), min(1.0, centre + half)

    set_a_row = pop_df[pop_df["population"] == "set_a"].iloc[0]
    set_b_row = pop_df[pop_df["population"] == "set_b_all"].iloc[0]
    bars = []
    for label, row, colour, hatch in [
        ("Set A (matched)", set_a_row, "#4a4a4a", None),
        ("Set B (cued ceiling)", set_b_row, "#888888", "//"),
    ]:
        n_comp = int(row["n_completed"]); n_cor = int(round(float(row["n_correct"])))
        rate = n_cor / n_comp if n_comp else 0.0
        lo, hi = wilson(n_cor, n_comp)
        bars.append((label, rate, lo, hi, None, colour, hatch))

    for cond in CONDITION_ORDER:
        r = cmp_df[cmp_df["condition"] == cond]
        if r.empty: continue
        r = r.iloc[0]
        n_comp = int(r["n_completed"]); n_cor = int(round(float(r["n_correct"])))
        rate = (n_cor / n_comp) if n_comp else 0.0
        lo, hi = wilson(n_cor, n_comp)
        if cond == "baseline":
            colour = "#000000"
        elif "_pos_" in cond:
            colour = "#3b73af"
        else:
            colour = "#d9534f"
        bars.append((cond, rate, lo, hi, r["fisher_p"], colour, None))

    fig, ax = plt.subplots(figsize=(15, 5))
    xs = np.arange(len(bars))
    rates = [b[1] for b in bars]
    yerr_low = [max(0.0, b[1] - b[2]) for b in bars]
    yerr_high = [max(0.0, b[3] - b[1]) for b in bars]
    colours = [b[5] for b in bars]
    container = ax.bar(xs, rates, color=colours, edgecolor="white",
                       yerr=[yerr_low, yerr_high], capsize=3, ecolor="#444444")
    for i, (label, rate, lo, hi, p, colour, hatch) in enumerate(bars):
        if hatch: container[i].set_hatch(hatch)
        if p is not None and p == p:
            stars = sig_stars(p)
            if stars: ax.text(i, hi + 0.012, stars, ha="center", fontsize=10)

    set_a_rate = bars[0][1]
    ax.axhline(set_a_rate, color="#4a4a4a", linestyle="--", alpha=0.6,
               label=f"Set A baseline ({set_a_rate*100:.1f}%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("LLM-reviewed accuracy among completed traces")
    ax.set_title("Accuracy vs. Study 1 Set A (matched baseline) and Set B (cued ceiling)\n"
                 "LLM-reviewed scores. Stars: Fisher's exact vs Set A "
                 "(* p<0.05, ** p<0.01, *** p<0.001) - uncorrected. "
                 "Hatched bar = Set B (includes completion cue).")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(0.30, max(b[3] for b in bars) * 1.15))
    fig.tight_layout()
    out = FIG / "accuracy_comparison_study1_baseline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def update_mechanistic_grounding(pop_df, cmp_df):
    probe_csv = ROOT / "outputs" / "study2_probes" / "probe_per_category_f1.csv"
    probe = pd.read_csv(probe_csv)
    probe_l20 = probe[(probe["layer"] == 20) & (probe["aggregation"] == "mean_pool")].iloc[0]
    eff = pd.read_csv(OUT / "on_target_effects.csv")

    set_a_row = pop_df[pop_df["population"] == "set_a"].iloc[0]
    base_acc = float(set_a_row["accuracy_among_completed"])
    base_n_correct = int(round(float(set_a_row["n_correct"])))
    base_n_completed = int(set_a_row["n_completed"])

    rows = []
    for cat in STEER_CATEGORIES:
        f1 = float(probe_l20[f"{cat}_f1"])
        cat_eff = eff[(eff["category"] == cat) & (eff["alpha"] == 1.0)]
        cat_eff_sig = cat_eff[cat_eff["p_bonf"] < 0.05]
        if not cat_eff_sig.empty:
            best = cat_eff_sig.iloc[cat_eff_sig["cohen_d"].abs().argmax()]
            steering_significant = True
        else:
            best = cat_eff.iloc[cat_eff["cohen_d"].abs().argmax()]
            steering_significant = False
        cond = best["condition"]
        cmp_row = cmp_df[cmp_df["condition"] == cond]
        if not cmp_row.empty:
            cond_acc = float(cmp_row["accuracy_rate"].iloc[0])
            cond_fisher_p = float(cmp_row["fisher_p"].iloc[0])
        else:
            cond_acc = float("nan"); cond_fisher_p = float("nan")
        acc_change = cond_acc - base_acc if cond_acc == cond_acc else float("nan")

        f1_grounded = f1 >= PROBE_F1_THRESHOLD
        if not f1_grounded:
            cls = "PERFORMATIVE"
            note = f"Probe F1 ({f1:.3f}) below threshold ({PROBE_F1_THRESHOLD})"
        elif not steering_significant:
            cls = "WEAKLY_GROUNDED"
            note = (f"Probe F1 ok ({f1:.3f}); steering not significant after Bonferroni "
                    f"(best p_bonf={best['p_bonf']:.3f}, d={best['cohen_d']:.2f})")
        else:
            base_meaningful = base_acc >= 0.05
            if not base_meaningful:
                cls = "PARTIALLY_GROUNDED"
                note = (f"Significant steering ({best['condition']}: shift={best['shift']*100:+.1f}pp, "
                        f"p_bonf={best['p_bonf']:.3g}); accuracy unmeasurable against "
                        f"Study 1 Set A baseline of {base_acc*100:.2f}% "
                        f"({base_n_correct}/{base_n_completed}); best-condition "
                        f"accuracy={cond_acc*100:.2f}% (Fisher p vs Set A = {cond_fisher_p:.3g})")
            elif acc_change < -0.10:
                cls = "PARTIALLY_GROUNDED"
                note = (f"Accuracy degraded by {acc_change*100:.1f}pp under steering "
                        f"({cond_acc*100:.1f}% vs Set A {base_acc*100:.1f}%, "
                        f"Fisher p={cond_fisher_p:.3g})")
            else:
                cls = "GROUNDED"
                note = (f"No accuracy degradation under steering "
                        f"({acc_change*100:+.1f}pp vs Set A; Fisher p={cond_fisher_p:.3g})")

        rows.append({
            "category": cat,
            "probe_f1_layer20_meanpool": round(f1, 4),
            "probe_grounded_threshold_06": f1_grounded,
            "best_condition": cond,
            "on_target_shift_pp": round(best["shift"] * 100, 2),
            "cohen_d": round(best["cohen_d"], 3),
            "wilcoxon_p_raw": round(best["p_raw"], 5),
            "wilcoxon_p_bonf": round(best["p_bonf"], 5) if best["p_bonf"] == best["p_bonf"] else float("nan"),
            "steering_significant_005": steering_significant,
            "accuracy_baseline_set_a": round(base_acc, 4),
            "accuracy_baseline_n": f"{base_n_correct}/{base_n_completed}",
            "accuracy_steered": round(cond_acc, 4) if cond_acc == cond_acc else float("nan"),
            "accuracy_change_pp": round(acc_change * 100, 2) if acc_change == acc_change else float("nan"),
            "accuracy_fisher_p_vs_set_a": round(cond_fisher_p, 5) if cond_fisher_p == cond_fisher_p else float("nan"),
            "classification": cls,
            "note": note,
        })

    out = pd.DataFrame(rows)
    out_path = OUT / "mechanistic_grounding_classification.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(out[["category", "probe_f1_layer20_meanpool", "on_target_shift_pp",
               "cohen_d", "wilcoxon_p_bonf", "classification"]].to_string(index=False))


def main():
    # Load heuristic per-trace files (with completion flags + counts)
    s1_h = pd.read_csv(OUT / "study1_accuracy_heuristic.csv")
    s2_h = pd.read_csv(OUT / "accuracy_heuristic.csv")

    # Load LLM-reviewed CSVs (only completed traces, with manual_correct filled)
    s1_r = pd.read_csv(OUT / "study1_accuracy_manual_review.csv")
    s2_r = pd.read_csv(OUT / "accuracy_manual_review.csv")

    s1_merged = merge_review_into_heuristic(s1_h, s1_r, ["set", "population", "task_id", "trace_id", "seed"])
    s2_merged = merge_review_into_heuristic(s2_h, s2_r, ["condition", "task_id", "trace_id", "seed"])

    # Sanity checks
    print(f"S1 merged: {len(s1_merged)} (manual_correct filled in {s1_merged['manual_correct'].notna().sum()})")
    print(f"S2 merged: {len(s2_merged)} (manual_correct filled in {s2_merged['manual_correct'].notna().sum()})")

    # Population summary
    pop_df = s1_population(s1_merged)
    pop_path = OUT / "study1_accuracy_by_population.csv"
    pop_df.to_csv(pop_path, index=False)
    print(f"\nSaved {pop_path}")
    print(pop_df.to_string(index=False))

    # Per-condition summary
    by_cond = s2b_by_condition(s2_merged)
    cond_path = OUT / "accuracy_by_condition.csv"
    by_cond.to_csv(cond_path, index=False)
    print(f"\nSaved {cond_path}")

    # Fisher's exact vs set_a
    set_a_row = pop_df[pop_df["population"] == "set_a"].iloc[0]
    set_a_n_correct = int(round(float(set_a_row["n_correct"])))
    set_a_n_completed = int(set_a_row["n_completed"])
    cmp_df = fisher_table(s2_merged, set_a_n_correct, set_a_n_completed)
    cmp_path = OUT / "accuracy_vs_study1_baseline.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"Saved {cmp_path}")
    print(cmp_df[["condition", "n_completed", "n_correct", "accuracy_rate",
                  "set_a_accuracy_rate", "fisher_p", "direction"]].round(4).to_string(index=False))

    # Plot
    plot_accuracy_comparison(pop_df, cmp_df)

    # Mechanistic grounding
    update_mechanistic_grounding(pop_df, cmp_df)


if __name__ == "__main__":
    main()
