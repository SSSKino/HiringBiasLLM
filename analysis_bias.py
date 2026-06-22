import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = "bias_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Load data
# =====================================================

exp1 = pd.read_csv("result/all_exp1_scores.csv")
exp3 = pd.read_csv("result/all_exp3_scores.csv")

# =====================================================
# Recommendation Transition Matrix
# =====================================================

print("Generating Recommendation Transition Matrix...")

transition_summary = []

for model in exp1["model"].unique():

    exp1_model = exp1[exp1["model"] == model]
    exp3_model = exp3[exp3["model"] == model]

    merged = exp1_model.merge(
        exp3_model,
        on="pair_id",
        suffixes=("_exp1", "_exp3")
    )

    transition = pd.crosstab(
        merged["recommendation_exp1"],
        merged["recommendation_exp3"]
    )

    transition.to_csv(
        f"{OUTPUT_DIR}/{model}_transition_matrix.csv"
    )

    plt.figure(figsize=(7,6))

    sns.heatmap(
        transition,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"{model} Recommendation Transition Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/recommendation_transition_{model}.png",
        dpi=300
    )

    plt.close()

    changed = (
        merged["recommendation_exp1"]
        !=
        merged["recommendation_exp3"]
    ).sum()

    total = len(merged)

    transition_summary.append({
        "model": model,
        "changed_count": changed,
        "total": total,
        "change_rate_percent": round(
            changed / total * 100,
            2
        )
    })

transition_summary = pd.DataFrame(
    transition_summary
)

transition_summary.to_csv(
    f"{OUTPUT_DIR}/recommendation_transition_summary.csv",
    index=False
)

# =====================================================
# Inter-model Agreement
# =====================================================

print("Generating Inter-model Agreement...")

agreement_records = []

for label, df in [
    ("Exp1", exp1),
    ("Exp3", exp3)
]:

    pivot = df.pivot_table(
        index=[
            "industry",
            "region",
            "jd_soc_code",
            "cv_id"
        ],
        columns="model",
        values="score_mean"
    )

    pivot["pair_std"] = pivot.std(
        axis=1
    )

    pivot["experiment"] = label

    agreement_records.append(
        pivot.reset_index()
    )

agreement_df = pd.concat(
    agreement_records,
    ignore_index=True
)

# =====================================================
# Boxplot
# =====================================================

plt.figure(figsize=(8,6))

sns.boxplot(
    data=agreement_df,
    x="experiment",
    y="pair_std"
)

plt.ylabel(
    "Std(score_mean across models)"
)

plt.title(
    "Inter-model Agreement: Exp1 vs Exp3"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/intermodel_agreement_boxplot.png",
    dpi=300
)

plt.close()

# =====================================================
# Industry Agreement
# =====================================================

industry_agreement = (
    agreement_df
    .groupby(
        ["industry", "experiment"]
    )["pair_std"]
    .mean()
    .reset_index()
)

industry_agreement.to_csv(
    f"{OUTPUT_DIR}/agreement_summary.csv",
    index=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=industry_agreement,
    x="industry",
    y="pair_std",
    hue="experiment"
)

plt.ylabel(
    "Mean Std(score_mean)"
)

plt.title(
    "Inter-model Agreement by Industry"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/intermodel_agreement_by_industry.png",
    dpi=300
)

plt.close()

# =====================================================
# Summary
# =====================================================

exp1_mean = (
    agreement_df[
        agreement_df["experiment"] == "Exp1"
    ]["pair_std"].mean()
)

exp3_mean = (
    agreement_df[
        agreement_df["experiment"] == "Exp3"
    ]["pair_std"].mean()
)

with open(
    f"{OUTPUT_DIR}/bias_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INTER-MODEL AGREEMENT ANALYSIS\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        f"Exp1 Mean Std: {exp1_mean:.4f}\n"
    )

    f.write(
        f"Exp3 Mean Std: {exp3_mean:.4f}\n\n"
    )

    if exp3_mean > exp1_mean:
        f.write(
            "Region information increased disagreement between models.\n"
        )
    else:
        f.write(
            "Region information reduced disagreement between models.\n"
        )
print("SCRIPT STARTED")
print()
print("Analysis Complete.")
print(f"Results saved to: {OUTPUT_DIR}")