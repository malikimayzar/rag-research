import json
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def create_radar_chart(json_path, output_path="results/metrics/radar_chart.png"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    result = data[-1]
    metrics = {
        'Faithfulness': result.get('avg_faithfulness', 0),
        'Answer Relevancy': result.get('avg_answer_relevancy', 0),
        'Context Precision': result.get('avg_context_precision', 0),
        'Context Recall': result.get('avg_context_recall', 0),
        'Answer Correctness': result.get('avg_answer_correctness', 0)
    }

    # Data setup
    labels = list(metrics.keys())
    values = list(metrics.values())
    values = [0 if np.isnan(v) else v for v in values]
    
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    values += values[:1]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='grey', size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=result['exp_id'])
    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title(f"RAG Ablation Study: {result['exp_id']}", size=16, color='navy', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OK] Radar Chart saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    create_radar_chart("results/metrics/ragas_results.json")