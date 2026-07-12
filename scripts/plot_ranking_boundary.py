"""Generates a plot showing the ranking flip boundary between Gemma and Llama
as the safety weight (w_s) varies from 0.2 to 0.8

Run:
    python3 scripts/plot_ranking_boundary.py
    
Output:
    results/ranking_flip_boundary.png
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load_scores(model_name: str) -> dict:
    """Load dimension scores for a model from its scores.json."""
    path = Path(f"results/{model_name}/scores.json")
    if not path.exists():
        print(f"  Scores file not found: {path}")
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    dims = data.get("dimension_scores", {})
    return {
        "safety": dims.get("safety", {}).get("score", 0),
        "truthfulness": dims.get("truthfulness", {}).get("score", 0),
        "consistency": dims.get("consistency", {}).get("score", 0),
    }


def compute_trustscore(scores: dict, w_s: float, w_t: float, w_c: float) -> float:
    """Compute weighted trust score."""
    return w_s * scores["safety"] + w_t * scores["truthfulness"] + w_c * scores["consistency"]


def main():
    print("=" * 60)
    print("Ranking Flip Boundary Analysis")
    print("=" * 60)
    
    # Load scores
    gemma_scores = load_scores("gemma3_4b")
    llama_scores = load_scores("llama3.1_8b")
    
    if gemma_scores is None or llama_scores is None:
        print("Could not load scores. Run the pipeline first.")
        return
    
    print(f"\nGemma scores:  Safety={gemma_scores['safety']:.4f}, "
          f"Truthfulness={gemma_scores['truthfulness']:.4f}, "
          f"Consistency={gemma_scores['consistency']:.4f}")
    print(f"Llama scores:  Safety={llama_scores['safety']:.4f}, "
          f"Truthfulness={llama_scores['truthfulness']:.4f}, "
          f"Consistency={llama_scores['consistency']:.4f}")
    
    # Define weight scan
    # Vary safety weight from 0.2 to 0.8
    # Remaining weight split equally between truthfulness and consistency
    w_s_values = np.linspace(0.2, 0.8, 100)
    
    gemma_trust = []
    llama_trust = []
    flip_found = False
    flip_point = None
    
    for w_s in w_s_values:
        remaining = 1.0 - w_s
        w_t = remaining * 0.5  # Half remaining to truthfulness
        w_c = remaining * 0.5  # Half remaining to consistency
        
        g_score = compute_trustscore(gemma_scores, w_s, w_t, w_c)
        l_score = compute_trustscore(llama_scores, w_s, w_t, w_c)
        
        gemma_trust.append(g_score)
        llama_trust.append(l_score)
        
        # Check for flip
        if not flip_found and l_score > g_score:
            flip_found = True
            flip_point = (w_s, g_score, l_score)
    
    if flip_point:
        print(f"\nRanking flip found at safety weight = {flip_point[0]:.4f}")
        print(f"   At this point: Gemma = {flip_point[1]:.4f}, Llama = {flip_point[2]:.4f}")
    else:
        print(f"\nNo ranking flip found in range [0.2, 0.8]")
        print(f"   Gemma always wins or Llama always wins in this range.")
    
    # Generate the plot
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(w_s_values, gemma_trust, 'b-', linewidth=2, label='Gemma 3 4B')
        ax.plot(w_s_values, llama_trust, 'r-', linewidth=2, label='Llama 3.1 8B')
        
        # Highlight the flip point
        if flip_point:
            ax.axvline(x=flip_point[0], color='gray', linestyle='--', alpha=0.5)
            ax.plot(flip_point[0], flip_point[1], 'bo', markersize=8)
            ax.plot(flip_point[0], flip_point[2], 'ro', markersize=8)
            ax.annotate(f'Flip at w_s={flip_point[0]:.2f}',
                       xy=(flip_point[0], flip_point[1]),
                       xytext=(flip_point[0] + 0.05, flip_point[1] - 0.02),
                       fontsize=10, ha='left',
                       arrowprops=dict(arrowstyle='->', color='gray'))
        
        # Highlight the baseline configuration
        baseline_w_s = 0.40
        remaining = 1.0 - baseline_w_s
        baseline_w_t = remaining * 0.35 / (0.35 + 0.25)  # Scale to match original proportions
        baseline_w_c = remaining * 0.25 / (0.35 + 0.25)
        baseline_g = compute_trustscore(gemma_scores, baseline_w_s, baseline_w_t, baseline_w_c)
        baseline_l = compute_trustscore(llama_scores, baseline_w_s, baseline_w_t, baseline_w_c)
        
        ax.plot(baseline_w_s, baseline_g, 'bD', markersize=10, label=f'Baseline Gemma ({baseline_g:.3f})')
        ax.plot(baseline_w_s, baseline_l, 'rD', markersize=10, label=f'Baseline Llama ({baseline_l:.3f})')
        
        ax.set_xlabel('Safety Weight (w_s)', fontsize=12)
        ax.set_ylabel('TrustScore', fontsize=12)
        ax.set_title('Ranking Flip Boundary: Gemma 3 4B vs Llama 3.1 8B', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add a horizontal line at the threshold
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.3)
        
        # Add text box with model scores
        s_text = f'Gemma: S={gemma_scores["safety"]:.2f}, T={gemma_scores["truthfulness"]:.2f}, C={gemma_scores["consistency"]:.2f}'
        l_text = f'Llama: S={llama_scores["safety"]:.2f}, T={llama_scores["truthfulness"]:.2f}, C={llama_scores["consistency"]:.2f}'
        textstr = s_text + '\n' + l_text
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save
        output_path = Path("results/ranking_flip_boundary.png")
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to {output_path}")
        
        # Also show weight sensitivity table
        print(f"\n{'='*60}")
        print("Weight Sensitivity Table")
        print(f"{'='*60}")
        print(f"{'Config':35} {'Gemma':10} {'Llama':10} {'Ranking':10}")
        print("-" * 65)
        
        configs = [
            ("Baseline (0.40, 0.35, 0.25)", 0.40),
            ("Safety-heavy (0.60, 0.25, 0.15)", 0.60),
            ("Balanced (0.33, 0.33, 0.34)", 0.33),
            ("Truthfulness-heavy (0.25, 0.50, 0.25)", 0.25),
            ("Consistency-heavy (0.20, 0.40, 0.40)", 0.20),
        ]
        
        for name, w_s in configs:
            remaining = 1.0 - w_s
            # Use the actual weight config proportions
            if "Baseline" in name:
                w_t, w_c = 0.35, 0.25
            elif "Safety" in name:
                w_t, w_c = 0.25, 0.15
            elif "Balanced" in name:
                w_t, w_c = 0.33, 0.34
            elif "Truthfulness" in name:
                w_t, w_c = 0.50, 0.25
            else:
                w_t, w_c = 0.40, 0.40
            
            g = compute_trustscore(gemma_scores, w_s, w_t, w_c)
            l = compute_trustscore(llama_scores, w_s, w_t, w_c)
            winner = "Gemma" if g > l else "Llama" if l > g else "Tie"
            print(f"{name:35} {g:.4f}   {l:.4f}   {winner}")
        
    except ImportError as e:
        print(f"\n⚠️  Could not create plot: {e}")
        print("Install matplotlib: pip install matplotlib")
        print("Manual table created instead.")


if __name__ == "__main__":
    main()