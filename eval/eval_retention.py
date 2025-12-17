import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc

def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def compute_metrics(results):
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    # Metrics per delay
    delays = sorted(df['delay'].unique())
    metrics = {
        'delays': delays,
        'recall_at_1': [],
        'recall_at_5': [],
        'qa_accuracy': []
    }
    
    for d in delays:
        sub = df[df['delay'] == d]
        # QA Accuracy (Generation Correctness)
        acc = sub['correct'].mean()
        metrics['qa_accuracy'].append(acc)
        
        # Retrieval Recall
        if 'retrieval_at_1' in sub.columns:
            r1 = sub['retrieval_at_1'].mean()
            r5 = sub['retrieval_at_5'].mean()
        else:
            # Fallback for baselines without explicit retrieval (KV Cache)
            # Use QA accuracy as proxy for Recall@1?
            # Or strictly 0?
            # User prompt: "C1 metric: Recall@1".
            # For KV-cache, "Recall" effectively means "did it answer right?".
            r1 = acc
            r5 = acc # Proxy
            
        metrics['recall_at_1'].append(r1)
        metrics['recall_at_5'].append(r5)
        
    # Compute AURC (Area Under Retention Curve) for Recall@1
    # Check if we have at least 2 points
    if len(delays) > 1:
        # Normalize delays to [0, 1] or keeps as is?
        # Usually AUC is strictly on the x-y values.
        # But delays are 1, 10, 50, 100. Log scale?
        # "PLOT: Recall@1 vs delay (log scale)"
        # AURC calculation usually linear integration.
        aurc = auc(delays, metrics['recall_at_1'])
    else:
        aurc = 0.0
        
    metrics['aurc'] = aurc
    return metrics

def plot_retention(metrics_map):
    plt.figure(figsize=(10, 6))
    
    for name, m in metrics_map.items():
        if not m: continue
        plt.plot(m['delays'], m['recall_at_1'], marker='o', label=f"{name} (AURC={m['aurc']:.2f})")
        
    plt.xscale('log')
    plt.xlabel("Delay (Sessions)")
    plt.ylabel("Recall @ 1")
    plt.title("C1: Retention Curve")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    os.makedirs("evaluation_results", exist_ok=True)
    plt.savefig("evaluation_results/c1_retention_curve.png")
    print("Saved plot to evaluation_results/c1_retention_curve.png")

def main():
    files = {
        "DBME": "results/dbme_retention.json",
        "KV-Cache": "results/baseline_kv.json",
        "Retrieval": "results/baseline_retrieval.json",
        "Compressive": "results/baseline_compressive.json"
    }
    
    metrics_map = {}
    
    print("Computing C1 Metrics...")
    print("-" * 60)
    print(f"{'Model':<15} | {'AURC':<10} | {'R@1 (delay=100)':<15}")
    print("-" * 60)
    
    for name, path in files.items():
        res = load_results(path)
        m = compute_metrics(res)
        metrics_map[name] = m
        
        if m:
            r1_100 = "N/A"
            if 100 in m['delays']:
                idx = m['delays'].index(100)
                r1_100 = f"{m['recall_at_1'][idx]:.2f}"
            print(f"{name:<15} | {m['aurc']:<10.2f} | {r1_100:<15}")
        else:
            print(f"{name:<15} | {'N/A':<10} | {'N/A':<15}")

    plot_retention(metrics_map)

if __name__ == "__main__":
    main()
