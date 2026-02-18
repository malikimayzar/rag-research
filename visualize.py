import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"[INFO] Working dir: {os.getcwd()}")

DATA_PATH = "results/metrics/ablation_final.json"
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] File tidak ditemukan: {DATA_PATH}")
    sys.exit(1)

os.makedirs("results/figures", exist_ok=True)

# ── Load & parse data ───────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)

rows = []
for e in raw:
    c = e['config']
    fm = e.get('failure_mode_distribution', {})
    rows.append({
        'exp_id':            c['exp_id'],
        'method':            c['retrieval_method'].upper(),
        'chunk_size':        c['chunk_size'],
        'overlap':           c['overlap'],
        'faithfulness':      e['avg_faithfulness'],
        'ctx_relevance':     e['avg_context_relevance'],
        'ans_relevance':     e['avg_answer_relevance'],
        'hallucination':     e['hallucination_rate'],
        'abstention_rate':   e['honest_abstention_rate'],
        'latency':           e['avg_latency'],
        'correct':           fm.get('correct', 0),
        'partial_context':   fm.get('partial_context', 0),
        'honest_abstention': fm.get('honest_abstention', 0),
        'total_queries':     e['total_queries'],
    })

df = pd.DataFrame(rows).sort_values('faithfulness', ascending=False).reset_index(drop=True)
df['rank']   = df.index + 1
df['label']  = df['exp_id'] + '\n' + df['method'] + ' c' + df['chunk_size'].astype(str)
df['config'] = df['method'] + '\nchunk=' + df['chunk_size'].astype(str)

COLORS = {'BM25': '#1F77B4', 'DENSE': '#2CA02C', 'HYBRID': '#FF7F0E'}
df['color'] = df['method'].map(COLORS)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})

print(df[['exp_id','method','chunk_size','faithfulness','ctx_relevance','latency']].to_string(index=False))

# ── Fig 1: Leaderboard ──────────────────────────────────────────
print("\n[PLOT] Figure 1 — Leaderboard...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 1 — Experiment Leaderboard', fontsize=14, fontweight='bold')

for ax, (metric, title) in zip(axes, [('faithfulness','Faithfulness Score'),('ctx_relevance','Context Relevance Score')]):
    bars = ax.barh(df['label'][::-1], df[metric][::-1], color=df['color'][::-1], edgecolor='white', height=0.6)
    for bar, val in zip(bars, df[metric][::-1]):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(title, fontsize=11)
    ax.axvline(df[metric].mean(), color='gray', linestyle='--', alpha=0.5, linewidth=1)

legend = [mpatches.Patch(color=v, label=k) for k,v in COLORS.items()]
axes[0].legend(handles=legend, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('results/figures/fig1_leaderboard.png', bbox_inches='tight', dpi=150)
plt.close()
print("  → Saved: results/figures/fig1_leaderboard.png")

# ── Fig 2: Chunk size effect ────────────────────────────────────
print("[PLOT] Figure 2 — Chunk Size Effect...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Figure 2 — Chunk Size Effect per Method', fontsize=14, fontweight='bold')

for ax, (metric, ylabel) in zip(axes, [('faithfulness','Faithfulness'),('ctx_relevance','Context Relevance'),('latency','Latency (s)')]):
    for method, color in COLORS.items():
        sub = df[df['method']==method].sort_values('chunk_size')
        if len(sub) == 2:
            ax.plot(sub['chunk_size'], sub[metric], 'o-', color=color, label=method, linewidth=2, markersize=8)
            for _, row in sub.iterrows():
                ax.annotate(f"{row[metric]:.2f}", (row['chunk_size'], row[metric]),
                            textcoords='offset points', xytext=(5,5), fontsize=8)
    ax.set_xticks([256, 512])
    ax.set_xlabel('Chunk Size', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9)
    if metric != 'latency':
        ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('results/figures/fig2_chunksize.png', bbox_inches='tight', dpi=150)
plt.close()
print("  → Saved: results/figures/fig2_chunksize.png")

# ── Fig 3: Heatmap ──────────────────────────────────────────────
print("[PLOT] Figure 3 — Heatmap...")
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Figure 3 — Metrics Heatmap', fontsize=14, fontweight='bold')

hm_cols = ['faithfulness','ctx_relevance','ans_relevance','hallucination','abstention_rate']
hm_labels = ['Faithfulness','Context\nRelevance','Answer\nRelevance','Hallucination\nRate','Abstention\nRate']
sns.heatmap(df.set_index('exp_id')[hm_cols], annot=True, fmt='.3f', cmap='RdYlGn',
            linewidths=0.5, linecolor='white', ax=ax, vmin=0, vmax=1,
            xticklabels=hm_labels, annot_kws={'size':11,'weight':'bold'})
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('results/figures/fig3_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("  → Saved: results/figures/fig3_heatmap.png")

# ── Fig 4: Failure modes ────────────────────────────────────────
print("[PLOT] Figure 4 — Failure Modes...")
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Figure 4 — Failure Mode Distribution', fontsize=14, fontweight='bold')

x = np.arange(len(df))
total = df['total_queries']
b1 = ax.bar(x, df['correct']/total*100, 0.5, label='Correct', color='#2ECC71', edgecolor='white')
b2 = ax.bar(x, df['partial_context']/total*100, 0.5, label='Partial Context', color='#F39C12', edgecolor='white', bottom=df['correct']/total*100)
b3 = ax.bar(x, df['honest_abstention']/total*100, 0.5, label='Honest Abstention', color='#95A5A6', edgecolor='white', bottom=(df['correct']+df['partial_context'])/total*100)

ax.set_xticks(x)
ax.set_xticklabels(df['config'], fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)

for bar_group in [b1, b2, b3]:
    for rect in bar_group:
        h = rect.get_height()
        if h > 5:
            ax.text(rect.get_x()+rect.get_width()/2., rect.get_y()+h/2,
                    f'{h:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('results/figures/fig4_failure_modes.png', bbox_inches='tight', dpi=150)
plt.close()
print("  → Saved: results/figures/fig4_failure_modes.png")

# ── Fig 5: Quality vs Latency ───────────────────────────────────
print("[PLOT] Figure 5 — Quality vs Latency...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Figure 5 — Quality vs Latency Trade-off', fontsize=14, fontweight='bold')

for _, row in df.iterrows():
    ax.scatter(row['latency'], row['faithfulness'], s=(row['ctx_relevance']+0.1)*800,
               color=COLORS[row['method']], alpha=0.75, edgecolors='white', linewidth=1.5, zorder=3)
    ax.annotate(f"{row['exp_id']}\n{row['method']} c{row['chunk_size']}",
                (row['latency'], row['faithfulness']), xytext=(8,4), textcoords='offset points', fontsize=8)

ax.set_xlabel('Average Latency (seconds)', fontsize=12)
ax.set_ylabel('Faithfulness Score', fontsize=12)
ax.set_ylim(0.5, 1.1)
ax.axhline(df['faithfulness'].mean(), color='gray', linestyle='--', alpha=0.4)
ax.axvline(df['latency'].mean(), color='gray', linestyle=':', alpha=0.4)
legend = [mpatches.Patch(color=v, label=k) for k,v in COLORS.items()]
ax.legend(handles=legend, fontsize=9)
ax.text(0.02, 0.97, 'Bubble size = Context Relevance', transform=ax.transAxes, fontsize=8, color='gray', va='top')

plt.tight_layout()
plt.savefig('results/figures/fig5_quality_latency.png', bbox_inches='tight', dpi=150)
plt.close()
print("  → Saved: results/figures/fig5_quality_latency.png")

# ── Fig 6: Radar ────────────────────────────────────────────────
print("[PLOT] Figure 6 — Radar Chart...")
method_avg = df.groupby('method')[['faithfulness','ctx_relevance','ans_relevance','abstention_rate']].mean()
method_avg['correct_rate'] = df.groupby('method').apply(lambda x: (x['correct']/x['total_queries']).mean())

categories = ['Faithfulness','Ctx Relevance','Ans Relevance','Correct Rate','Low Abstention']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
fig.suptitle('Figure 6 — Method Comparison (Radar)', fontsize=14, fontweight='bold')

for method in method_avg.index:
    r = method_avg.loc[method]
    vals = [r['faithfulness'], r['ctx_relevance'], r['ans_relevance'],
            r['correct_rate'], 1-r['abstention_rate']] + [r['faithfulness']]
    ax.plot(angles, vals, 'o-', linewidth=2, label=method, color=COLORS[method])
    ax.fill(angles, vals, alpha=0.1, color=COLORS[method])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/fig6_radar.png', bbox_inches='tight', dpi=150)
plt.close()
print("  → Saved: results/figures/fig6_radar.png")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SELESAI — 6 figures tersimpan di results/figures/")
print("="*60)
figs = [f for f in os.listdir('results/figures') if f.endswith('.png')]
for f in sorted(figs):
    print(f"  ✓ {f}")