import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results/tables/model_comparison.csv")
feature_sets = df["feature_set"].unique()
models = df["model"].unique()


def grouped_bar(metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = range(len(models))
    for i, fs in enumerate(feature_sets):
        subset = df[df["feature_set"] == fs].set_index("model").loc[models, metric]
        ax.bar([xi + i * width for xi in x], subset.values, width, label=fs)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Model and Feature Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved results/figures/{filename}")


grouped_bar("accuracy", "Accuracy", "comparison_accuracy.png")
grouped_bar("f1", "F1 Score", "comparison_f1.png")
grouped_bar("fpr_standard", "False Positive Rate", "comparison_fpr.png")
