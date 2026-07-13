"""Generates all analysis figures and tables for the project report
Run once after ./week2.sh completes.

Analysis 1: CI bar chart with error bars
Analysis 2: Weight sensitivity (TrustScore vs safety weight)
Analysis 3: Score difference heatmap (how much Gemma wins by)
Analysis 4: Text summary table

Outputs:
    results/ci_error_bars.png             — Bar chart with CI error bars
    results/ranking_flip_boundary.png     — Weight sensitivity plot
    results/score_difference_heatmap.png  — Heatmap of score differences
    results/analysis_summary.txt          — Text summary of all findings
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = "results"


def load_scores(model_name: str) -> dict:
    """Load scores from results/ directory"""
    path = Path(f"{RESULTS_DIR}/{model_name}/scores.json")
    with open(path) as f:
        return json.load(f)


def get_dim_scores(data: dict):
    """Extract dimension scores from loaded JSON"""
    dims = data["dimension_scores"]
    return {
        "safety": dims["safety"]["score"],
        "truthfulness": dims["truthfulness"]["score"],
        "consistency": dims["consistency"]["score"],
    }


def compute_trustscore(scores: dict, w_s: float, w_t: float, w_c: float) -> float:
    """Compute weighted trust score."""
    return w_s * scores["safety"] + w_t * scores["truthfulness"] + w_c * scores["consistency"]


# ANALYSIS 1: CI Bar Chart
def generate_ci_plot():
    """Bar chart with confidence interval error bars"""
    try:
        import matplotlib.pyplot as plt
        
        gemma = load_scores("gemma3_4b")
        llama = load_scores("llama3.1_8b")
        
        dimensions = ['Safety', 'Truthfulness', 'Consistency']
        colors = ['#3498db', '#e74c3c']
        x = np.arange(len(dimensions))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (label, data) in enumerate([('Gemma 3 4B', gemma), ('Llama 3.1 8B', llama)]):
            scores = []
            ci_lowers = []
            ci_uppers = []
            for dim in ['safety', 'truthfulness', 'consistency']:
                ds = data['dimension_scores'][dim]
                ci = data['confidence_intervals'][dim]
                scores.append(ds['score'])
                ci_lowers.append(ci['ci_lower'])
                ci_uppers.append(ci['ci_upper'])
            
            yerr = np.array([
                [s - l for s, l in zip(scores, ci_lowers)],
                [u - s for s, u in zip(scores, ci_uppers)]
            ])
            
            ax.bar(x + i * width, scores, width, label=label,
                   color=colors[i], yerr=yerr, capsize=5)
        
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(dimensions)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Dimension Scores with 95% Confidence Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/ci_error_bars.png", dpi=150)
        print(f"Saved {RESULTS_DIR}/ci_error_bars.png")
    except ImportError:
        print(" matplotlib not installed. Skipping CI plot.")


# ANALYSIS 2: Weight Sensitivity & Score Difference Heatmap
def generate_ranking_plots():
    """Generate weight sensitivity plot and score difference heatmap"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        gemma_data = load_scores("gemma3_4b")
        llama_data = load_scores("llama3.1_8b")
        g = get_dim_scores(gemma_data)
        l = get_dim_scores(llama_data)
        
        print(f"\n{'='*60}")
        print("Weight Sensitivity Analysis")
        print(f"{'='*60}")
        print(f"Gemma scores:  S={g['safety']:.4f}, T={g['truthfulness']:.4f}, C={g['consistency']:.4f}")
        print(f"Llama scores:  S={l['safety']:.4f}, T={l['truthfulness']:.4f}, C={l['consistency']:.4f}")
        print()
        
        # --- Plot 1: TrustScore vs Safety Weight ---
        w_s_range = np.linspace(0.0, 0.9, 200)
        gemma_trust = []
        llama_trust = []
        
        for w_s in w_s_range:
            remaining = 1.0 - w_s
            w_t = remaining * 0.35 / 0.6  # Keep T:C ratio from baseline
            w_c = remaining * 0.25 / 0.6
            gemma_trust.append(compute_trustscore(g, w_s, w_t, w_c))
            llama_trust.append(compute_trustscore(l, w_s, w_t, w_c))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(w_s_range, gemma_trust, 'b-', linewidth=2, label='Gemma 3 4B')
        ax.plot(w_s_range, llama_trust, 'r-', linewidth=2, label='Llama 3.1 8B')
        
        # Mark default configurations
        configs = [
            (0.40, "Baseline"),
            (0.60, "Safety-heavy"),
            (0.33, "Balanced"),
            (0.25, "Truthfulness-heavy"),
            (0.20, "Consistency-heavy"),
        ]
        markers = ['o', 's', '^', 'D', 'v']
        for (w_s, name), marker in zip(configs, markers):
            remaining = 1.0 - w_s
            w_t = remaining * 0.35 / 0.6
            w_c = remaining * 0.25 / 0.6
            g_score = compute_trustscore(g, w_s, w_t, w_c)
            l_score = compute_trustscore(l, w_s, w_t, w_c)
            ax.plot(w_s, g_score, 'b' + marker, markersize=8)
            ax.plot(w_s, l_score, 'r' + marker, markersize=8)
        
        ax.set_xlabel('Safety Weight (w_s)', fontsize=12)
        ax.set_ylabel('TrustScore', fontsize=12)
        ax.set_title('TrustScore vs Safety Weight', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Text box
        s_text = f'Gemma: S={g["safety"]:.2f}, T={g["truthfulness"]:.2f}, C={g["consistency"]:.2f}'
        l_text = f'Llama: S={l["safety"]:.2f}, T={l["truthfulness"]:.2f}, C={l["consistency"]:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, s_text + '\n' + l_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/ranking_flip_boundary.png", dpi=150)
        print(f"Saved {RESULTS_DIR}/ranking_flip_boundary.png")
        
        # --- Plot 2: Score Difference Heatmap ---
        w_s_range = np.linspace(0.0, 0.8, 50)
        t_share_range = np.linspace(0.1, 0.9, 50)
        
        diff = np.zeros((len(t_share_range), len(w_s_range)))
        
        for i, t_share in enumerate(t_share_range):
            for j, w_s in enumerate(w_s_range):
                remaining = 1.0 - w_s
                w_t = remaining * t_share
                w_c = remaining * (1.0 - t_share)
                g_score = compute_trustscore(g, w_s, w_t, w_c)
                l_score = compute_trustscore(l, w_s, w_t, w_c)
                diff[i, j] = g_score - l_score  # Positive = Gemma wins
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Custom colormap: red (Llama wins) -> white (tie) -> blue (Gemma wins)
        cmap = LinearSegmentedColormap.from_list('diff', 
            ['#e74c3c', '#ffffff', '#3498db'], N=256)
        
        im = ax.imshow(diff, aspect='auto', origin='lower', cmap=cmap,
                      vmin=-0.05, vmax=0.15,
                      extent=[w_s_range[0], w_s_range[-1], 
                             t_share_range[0], t_share_range[-1]])
        
        plt.colorbar(im, ax=ax, label='Score Difference (Gemma - Llama)')
        
        # Contour at zero (tie line)
        cs = ax.contour(w_s_range, t_share_range, diff, levels=[0], 
                       colors='black', linewidths=2, linestyles='--')
        ax.clabel(cs, inline=True, fontsize=10, fmt='Tie Line')
        
        ax.set_xlabel('Safety Weight (w_s)', fontsize=12)
        ax.set_ylabel('Truthfulness Share of Remaining', fontsize=12)
        ax.set_title('Score Difference: Gemma − Llama', fontsize=14)
        ax.text(0.5, -0.1, 'Blue = Gemma wins, Red = Llama wins, White = Tie',
                transform=ax.transAxes, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/score_difference_heatmap.png", dpi=150)
        print(f"Saved {RESULTS_DIR}/score_difference_heatmap.png")
        
        # --- Print key findings ---
        print(f"\nKey Findings:")
        print(f"  Gemma's max lead: {diff.max():.4f}")
        print(f"  Llama's max lead: {abs(diff.min()):.4f}")
        print(f"  Tie exists: {'Yes' if diff.min() < 0 and diff.max() > 0 else 'No'}")
        
    except ImportError as e:
        print(f"Could not generate plots: {e}")


# ANALYSIS 3: Text Summary
def generate_summary():
    """Generate text summary of all findings."""
    gemma = load_scores("gemma3_4b")
    llama = load_scores("llama3.1_8b")
    
    g_dims = gemma['dimension_scores']
    l_dims = llama['dimension_scores']
    
    lines = []
    lines.append("=" * 60)
    lines.append("MULTITRUSTSCORE — ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Dimension Scores:")
    lines.append(f"  Gemma:  S={g_dims['safety']['score']:.4f}  T={g_dims['truthfulness']['score']:.4f}  C={g_dims['consistency']['score']:.4f}")
    lines.append(f"  Llama:  S={l_dims['safety']['score']:.4f}  T={l_dims['truthfulness']['score']:.4f}  C={l_dims['consistency']['score']:.4f}")
    lines.append("")
    
    lines.append("Confidence Intervals (95%):")
    for model_name, data in [("Gemma", gemma), ("Llama", llama)]:
        lines.append(f"  {model_name}:")
        for dim in ['safety', 'truthfulness', 'consistency']:
            ci = data['confidence_intervals'][dim]
            lines.append(f"    {dim}: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    lines.append("")
    
    g_cm = g_dims['safety'].get('confusion_matrix', {})
    l_cm = l_dims['safety'].get('confusion_matrix', {})
    lines.append("Safety Confusion Matrix:")
    lines.append(f"                     Gemma   Llama")
    lines.append(f"  Malicious refused:  {g_cm.get('malicious_refused', 0):3d}     {l_cm.get('malicious_refused', 0):3d}")
    lines.append(f"  Malicious complied: {g_cm.get('malicious_complied', 0):3d}     {l_cm.get('malicious_complied', 0):3d}")
    lines.append(f"  Benign answered:    {g_cm.get('benign_answered', 0):3d}     {l_cm.get('benign_answered', 0):3d}")
    lines.append(f"  Benign refused:     {g_cm.get('benign_refused', 0):3d}     {l_cm.get('benign_refused', 0):3d}")
    lines.append("")
    
    lines.append("Weight Sensitivity (Baseline configs):")
    lines.append(f"  {'Config':30} {'Gemma':8} {'Llama':8} {'Winner':8}")
    lines.append("  " + "-" * 54)
    g = get_dim_scores(gemma)
    l = get_dim_scores(llama)
    configs = [
        ("Baseline", 0.40, 0.35, 0.25),
        ("Safety-heavy", 0.60, 0.25, 0.15),
        ("Balanced", 0.33, 0.33, 0.34),
        ("Truthfulness-heavy", 0.25, 0.50, 0.25),
        ("Consistency-heavy", 0.20, 0.40, 0.40),
    ]
    for name, w_s, w_t, w_c in configs:
        gs = compute_trustscore(g, w_s, w_t, w_c)
        ls = compute_trustscore(l, w_s, w_t, w_c)
        winner = "Gemma" if gs > ls else "Llama" if ls > gs else "Tie"
        lines.append(f"  {name:30} {gs:.4f}  {ls:.4f}  {winner}")
    lines.append("")
    
    lines.append("Ranking Instability:")
    lines.append("  Gemma wins on Safety and Consistency.")
    lines.append("  Llama wins on Truthfulness.")
    lines.append("  Overall ranking depends on chosen weights.")
    lines.append("  -> Do NOT claim a single winner.")
    
    with open(f"{RESULTS_DIR}/analysis_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {RESULTS_DIR}/analysis_summary.txt")


# MAIN
if __name__ == "__main__":
    print("Generating analysis...")
    print()
    generate_ci_plot()
    generate_ranking_plots()
    generate_summary()
    print("\nAnalysis complete!")