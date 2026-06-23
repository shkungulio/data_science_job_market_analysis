"""
analysis.py
-----------
Answers key questions from the cleaned job data.
Run AFTER cleaning.py.

Questions answered:
1. What are the most in-demand skills for data science jobs?
2. How do skill requirements vary by seniority?
3. What does the salary distribution look like (where available)?
4. Which skill combinations appear together most often?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Skill columns start with "skill_"
def get_skill_cols(df):
    return [c for c in df.columns if c.startswith("skill_")]


def skill_names(skill_cols):
    return [c.replace("skill_", "").replace("_", " ").title() for c in skill_cols]


# ── Q1: Most in-demand skills ─────────────────────────────────────────────────

def plot_top_skills(df, top_n=20, output="chart_top_skills.png"):
    skill_cols = get_skill_cols(df)
    counts = df[skill_cols].sum().sort_values(ascending=False).head(top_n)
    labels = [c.replace("skill_", "").replace("_", " ").title() for c in counts.index]
    pct = (counts / len(df) * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels[::-1], pct.values[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("% of Job Postings Mentioning Skill")
    ax.set_title(f"Top {top_n} Most In-Demand Skills\n(Data Science Remote Jobs)", fontsize=14, pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    for bar, val in zip(bars, pct.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")

    print("\nTop 10 skills:")
    for label, val in zip(labels[:10], pct.values[:10]):
        print(f"  {label:<25} {val:.1f}%")


# ── Q2: Skills by seniority ───────────────────────────────────────────────────

def plot_skills_by_seniority(df, skills_to_show=None, output="chart_skills_by_seniority.png"):
    """
    Heatmap: seniority on y-axis, top skills on x-axis,
    values = % of postings at that seniority mentioning each skill.
    """
    if skills_to_show is None:
        # Auto-select top 12 skills overall
        skill_cols = get_skill_cols(df)
        top_skills = df[skill_cols].sum().sort_values(ascending=False).head(12).index.tolist()
    else:
        top_skills = [f"skill_{s}" for s in skills_to_show]

    levels = ["junior", "mid", "senior", "lead", "unknown"]
    levels = [l for l in levels if l in df["seniority"].unique()]

    heatmap_data = []
    for level in levels:
        subset = df[df["seniority"] == level]
        if len(subset) == 0:
            row = [0] * len(top_skills)
        else:
            row = [(subset[col].sum() / len(subset) * 100) for col in top_skills]
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=levels,
        columns=[c.replace("skill_", "").replace("_", " ").title() for c in top_skills]
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap_df.values, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=10)

    for i in range(len(levels)):
        for j in range(len(top_skills)):
            val = heatmap_df.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if val > 50 else "black")

    plt.colorbar(im, ax=ax, label="% of Postings")
    ax.set_title("Skill Demand by Seniority Level", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


# ── Q3: Salary distribution ───────────────────────────────────────────────────

def plot_salary_distribution(df, output="chart_salary_distribution.png"):
    salary_df = df[df["salary_mid_usd"].notna() & (df["salary_mid_usd"] > 0)].copy()

    if len(salary_df) < 5:
        print(f"Only {len(salary_df)} jobs with salary data — skipping salary chart.")
        print("Note: this is realistic! Most remote job boards hide salaries.")
        print("Document this finding in your writeup — it's an insight, not a failure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overall histogram
    axes[0].hist(salary_df["salary_mid_usd"] / 1000, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Salary (USD thousands)")
    axes[0].set_ylabel("Number of Jobs")
    axes[0].set_title("Salary Distribution (Mid-point)")
    median_sal = salary_df["salary_mid_usd"].median()
    axes[0].axvline(median_sal / 1000, color="red", linestyle="--", label=f"Median: ${median_sal/1000:.0f}k")
    axes[0].legend()

    # Right: salary by seniority (box plot)
    seniority_order = ["junior", "mid", "senior", "lead"]
    seniority_order = [s for s in seniority_order if s in salary_df["seniority"].unique()]
    groups = [salary_df[salary_df["seniority"] == s]["salary_mid_usd"].values / 1000
              for s in seniority_order]
    groups = [g for g in groups if len(g) > 0]

    if groups:
        axes[1].boxplot(groups, labels=seniority_order[:len(groups)], patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        axes[1].set_xlabel("Seniority")
        axes[1].set_ylabel("Salary (USD thousands)")
        axes[1].set_title("Salary by Seniority Level")

    plt.suptitle("Salary Analysis — Remote Data Science Jobs", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")

    print(f"\nSalary stats ({len(salary_df)} jobs with data):")
    print(f"  Median: ${salary_df['salary_mid_usd'].median()/1000:.0f}k")
    print(f"  Mean:   ${salary_df['salary_mid_usd'].mean()/1000:.0f}k")
    print(f"  Range:  ${salary_df['salary_mid_usd'].min()/1000:.0f}k – ${salary_df['salary_mid_usd'].max()/1000:.0f}k")


# ── Q4: Co-occurring skills ───────────────────────────────────────────────────

def plot_skill_cooccurrence(df, top_n_pairs=15, output="chart_skill_cooccurrence.png"):
    """
    Find which skills most often appear together in the same posting.
    """
    skill_cols = get_skill_cols(df)
    pair_counts = Counter()

    for _, row in df.iterrows():
        present = [col for col in skill_cols if row[col]]
        for pair in combinations(present, 2):
            # Sort so (A,B) and (B,A) are the same
            pair_counts[tuple(sorted(pair))] += 1

    top_pairs = pair_counts.most_common(top_n_pairs)
    labels = [
        f"{a.replace('skill_','').replace('_',' ').title()} + "
        f"{b.replace('skill_','').replace('_',' ').title()}"
        for (a, b), _ in top_pairs
    ]
    values = [count for _, count in top_pairs]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels[::-1], values[::-1], color="#55A868", edgecolor="white")
    ax.set_xlabel("Number of Job Postings")
    ax.set_title(f"Top {top_n_pairs} Most Co-occurring Skill Pairs", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")

    print(f"\nTop 5 skill pairs:")
    for (a, b), count in top_pairs[:5]:
        skill_a = a.replace("skill_", "").replace("_", " ").title()
        skill_b = b.replace("skill_", "").replace("_", " ").title()
        print(f"  {skill_a} + {skill_b}: {count} jobs")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_analysis(input_file="cleaned_jobs.csv"):
    print(f"Loading cleaned data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Shape: {df.shape}\n")

    print("── Q1: Top skills ─────────────────────────────")
    plot_top_skills(df)

    print("\n── Q2: Skills by seniority ────────────────────")
    plot_skills_by_seniority(df)

    print("\n── Q3: Salary distribution ────────────────────")
    plot_salary_distribution(df)

    print("\n── Q4: Skill co-occurrence ────────────────────")
    plot_skill_cooccurrence(df)

    print("\nAll charts saved. Open them to review before writing up.")


if __name__ == "__main__":
    run_analysis()
