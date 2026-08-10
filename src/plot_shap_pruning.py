import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results/tables/shap_pruned_comparison.csv")
metrics = ["accuracy", "precision", "recall", "f1", "fpr"]

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(metrics))
width = 0.35
for i, (_, row) in enumerate(df.iterrows()):
    values = [row[m] for m in metrics]
    ax.bar([xi + i * width for xi in x], values, width, label=row["condition"])
ax.set_xticks([xi + width / 2 for xi in x])
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_title("SHAP-Pruned vs Original GIWRF Feature Set (21 vs 26 features)")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/shap_pruning_comparison.png", dpi=150)
plt.close()
print("Saved results/figures/shap_pruning_comparison.png")
