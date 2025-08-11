import matplotlib.pyplot as plt
import pandas as pd
import os
df_all = pd.read_csv("all_metrics_vs_quality copy.csv")
save_dir = "results_eval_dtd_org2/plots"
metrics = ["IoU Class 1", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for name in df_all["Dataset"].unique():
        df_subset = df_all[df_all["Dataset"] == name]
        plt.plot(df_subset["JPEG Quality"], df_subset[metric], label=name)
    plt.xlabel("JPEG Quality")
    plt.ylabel(metric)
    plt.title(f"{metric} vs JPEG Quality across datasets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric.lower().replace(' ', '_')}_vs_quality_.png")
    plt.close()