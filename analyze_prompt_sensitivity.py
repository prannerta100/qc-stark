"""Prompt sensitivity analysis: compare baseline vs structured prompt results.

Generates:
1. Paired bar chart: model accuracy under each prompt condition
2. Rank-order stability analysis (Kendall's tau)
3. Per-task sensitivity heatmap (which tasks are most affected by prompt?)
4. Statistical significance of ranking changes (bootstrap permutation test)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from collections import defaultdict
from scipy.stats import kendalltau

MINIMAL_FILE = Path("results/sensitivity_minimal_final.jsonl")
STRUCTURED_FILE = Path("results/full_benchmark_final.jsonl")
PLOTS_DIR = Path("results/plots/sensitivity")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "llama-70b", "gemini-flash-lite", "mistral-large-3",
    "claude-opus", "gpt-4.1-mini", "gemma-4-31b",
    "gpt-5.4", "o4-mini", "gemini-3.5-flash", "claude-sonnet-5",
]
MODEL_LABELS = {
    "llama-70b": "LLaMA-70B",
    "gemini-flash-lite": "Flash Lite",
    "mistral-large-3": "Mistral L3",
    "gemma-4-31b": "Gemma-4 31B",
    "gpt-4.1-mini": "GPT-4.1m",
    "o4-mini": "o4-mini",
    "gpt-5.4": "GPT-5.4",
    "claude-opus": "Opus 4.1",
    "claude-sonnet-5": "Sonnet 5",
    "gemini-3.5-flash": "Gemini 3.5F",
}
TASK_ORDER = ["A_stateprep", "B1_debugging", "B3_noise_logic", "C1_routing", "D1_qec",
              "E2_vqe", "F1_equivalence", "G1_trotter", "H1_oracle", "I1_noise", "J1_reverse"]
TASK_LABELS = {
    "A_stateprep": "State Prep",
    "B1_debugging": "Debugging",
    "B3_noise_logic": "Noise/Logic",
    "C1_routing": "Routing",
    "D1_qec": "QEC Decode",
    "E2_vqe": "VQE",
    "F1_equivalence": "Equivalence",
    "G1_trotter": "Trotter",
    "H1_oracle": "Oracle",
    "I1_noise": "Noise Fid",
    "J1_reverse": "Reverse Eng",
}
EXCLUDE_MODELS = set()


def load_results(filepath):
    records = []
    with open(filepath) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                det = r.get("details")
                if isinstance(det, dict) and "api_error" in det:
                    continue          # infrastructure failure, not a result
                if r.get("model") not in EXCLUDE_MODELS:
                    records.append(r)
            except:
                pass
    return records


def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    scores = np.array(scores, dtype=float)
    mean = np.mean(scores)
    if len(scores) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return mean, lo, hi


def compute_model_scores(records):
    """Returns {model: [scores]} for models in MODEL_ORDER."""
    model_scores = defaultdict(list)
    for r in records:
        if r["model"] in MODEL_ORDER:
            score = r["score"]
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                model_scores[r["model"]].append(float(score))
    return model_scores


def compute_task_model_scores(records):
    """Returns {(task, model): [scores]}."""
    scores = defaultdict(list)
    for r in records:
        if r["model"] in MODEL_ORDER:
            score = r["score"]
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                scores[(r["task_name"], r["model"])].append(float(score))
    return scores


def filter_to_matched_pairs(baseline, structured):
    """Only compare evaluations that exist in BOTH conditions (same task/model/level/seed)."""
    struct_keys = {(r["task_name"], r["model"], r["level"], r["seed"]) for r in structured}
    base_keys = {(r["task_name"], r["model"], r["level"], r["seed"]) for r in baseline}
    common = struct_keys & base_keys

    base_matched = [r for r in baseline
                    if (r["task_name"], r["model"], r["level"], r["seed"]) in common]
    struct_matched = [r for r in structured
                     if (r["task_name"], r["model"], r["level"], r["seed"]) in common]
    return base_matched, struct_matched, len(common)


def plot_paired_bars(baseline_scores, structured_scores):
    """Side-by-side bars: model accuracy under each prompt condition."""
    models = [m for m in MODEL_ORDER if m in baseline_scores and m in structured_scores]

    base_means, base_los, base_his = [], [], []
    struct_means, struct_los, struct_his = [], [], []

    for m in models:
        mean, lo, hi = bootstrap_ci(baseline_scores[m])
        base_means.append(mean)
        base_los.append(lo)
        base_his.append(hi)
        mean, lo, hi = bootstrap_ci(structured_scores[m])
        struct_means.append(mean)
        struct_los.append(lo)
        struct_his.append(hi)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, base_means, width, label='Minimal prompt',
                   color='#4C72B0', alpha=0.85,
                   yerr=[[m-l for m,l in zip(base_means, base_los)],
                         [h-m for m,h in zip(base_means, base_his)]],
                   capsize=3, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, struct_means, width, label='Structured prompt',
                   color='#DD8452', alpha=0.85,
                   yerr=[[m-l for m,l in zip(struct_means, struct_los)],
                         [h-m for m,h in zip(struct_means, struct_his)]],
                   capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('Prompt Sensitivity: Minimal vs Structured System Prompt', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    # Annotate differences
    for i, (bm, sm) in enumerate(zip(base_means, struct_means)):
        diff = sm - bm
        if abs(diff) > 0.01:
            color = 'green' if diff > 0 else 'red'
            ax.annotate(f'{diff:+.2f}', xy=(x[i], max(bm, sm) + 0.06),
                       ha='center', fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "paired_bars_prompt_sensitivity.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'paired_bars_prompt_sensitivity.png'}")


def plot_rank_stability(baseline_scores, structured_scores):
    """Show how model rankings change between prompts."""
    models = [m for m in MODEL_ORDER if m in baseline_scores and m in structured_scores]

    base_means = {m: np.mean(baseline_scores[m]) for m in models}
    struct_means = {m: np.mean(structured_scores[m]) for m in models}

    base_rank = sorted(models, key=lambda m: base_means[m], reverse=True)
    struct_rank = sorted(models, key=lambda m: struct_means[m], reverse=True)

    # Kendall's tau
    base_order = [base_rank.index(m) for m in models]
    struct_order = [struct_rank.index(m) for m in models]
    tau, pval = kendalltau(base_order, struct_order)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, m in enumerate(base_rank):
        base_pos = i
        struct_pos = struct_rank.index(m)
        color = '#2ca02c' if struct_pos <= base_pos else '#d62728'
        if struct_pos == base_pos:
            color = '#7f7f7f'
        ax.plot([0, 1], [base_pos, struct_pos], 'o-', color=color,
               markersize=8, linewidth=2, alpha=0.7)
        ax.text(-0.05, base_pos, f'{i+1}. {MODEL_LABELS.get(m, m)}',
               ha='right', va='center', fontsize=9)
        ax.text(1.05, struct_pos, f'{struct_pos+1}. {MODEL_LABELS.get(m, m)}',
               ha='left', va='center', fontsize=9)

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Minimal Prompt', 'Structured Prompt'], fontsize=11)
    ax.set_yticks([])
    ax.set_title(f'Rank Stability Across Prompts (Kendall τ = {tau:.3f}, p = {pval:.3f})',
                fontsize=12)
    ax.axvline(x=0, color='gray', alpha=0.3)
    ax.axvline(x=1, color='gray', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rank_stability.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'rank_stability.png'}")
    return tau, pval


def plot_task_sensitivity_heatmap(baseline_tm, structured_tm):
    """Heatmap of (structured - baseline) score per task × model."""
    models = [m for m in MODEL_ORDER
              if any((t, m) in baseline_tm for t in TASK_ORDER)
              and any((t, m) in structured_tm for t in TASK_ORDER)]
    tasks = [t for t in TASK_ORDER
             if any((t, m) in baseline_tm for m in models)]

    diff_matrix = np.zeros((len(models), len(tasks)))
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            base = np.mean(baseline_tm.get((task, model), [0]))
            struct = np.mean(structured_tm.get((task, model), [0]))
            diff_matrix[i, j] = struct - base

    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()), 0.3)
    im = ax.imshow(diff_matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])

    for i in range(len(models)):
        for j in range(len(tasks)):
            val = diff_matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Score Change (structured - baseline)")
    ax.set_title("Prompt Sensitivity by Task × Model\n(blue = structured helps, red = structured hurts)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task_sensitivity_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'task_sensitivity_heatmap.png'}")


def compute_rank_flip_significance(baseline_scores, structured_scores, n_perms=10000):
    """Bootstrap permutation test: is the rank change statistically significant?

    Null hypothesis: the two prompt conditions produce the same ranking.
    Test statistic: number of pairwise rank inversions between conditions.
    """
    models = [m for m in MODEL_ORDER if m in baseline_scores and m in structured_scores]
    rng = np.random.default_rng(42)

    base_means = np.array([np.mean(baseline_scores[m]) for m in models])
    struct_means = np.array([np.mean(structured_scores[m]) for m in models])

    # Observed inversions
    base_rank = np.argsort(-base_means)
    struct_rank = np.argsort(-struct_means)
    observed_inversions = sum(
        1 for i in range(len(models)) for j in range(i+1, len(models))
        if (base_rank[i] - base_rank[j]) * (struct_rank[i] - struct_rank[j]) < 0
    )

    # Permutation: pool all scores per model, randomly split into two conditions
    null_inversions = []
    for _ in range(n_perms):
        perm_base = []
        perm_struct = []
        for m in models:
            pooled = baseline_scores[m] + structured_scores[m]
            perm = rng.permutation(len(pooled))
            half = len(baseline_scores[m])
            perm_base.append(np.mean([pooled[i] for i in perm[:half]]))
            perm_struct.append(np.mean([pooled[i] for i in perm[half:]]))
        pb_rank = np.argsort(-np.array(perm_base))
        ps_rank = np.argsort(-np.array(perm_struct))
        inv = sum(
            1 for i in range(len(models)) for j in range(i+1, len(models))
            if (pb_rank[i] - pb_rank[j]) * (ps_rank[i] - ps_rank[j]) < 0
        )
        null_inversions.append(inv)

    p_value = np.mean(np.array(null_inversions) >= observed_inversions)
    return observed_inversions, p_value


def print_summary(baseline_scores, structured_scores, tau, pval, inv, inv_pval):
    """Print a comprehensive text summary."""
    models = [m for m in MODEL_ORDER if m in baseline_scores and m in structured_scores]

    print("\n" + "="*80)
    print("PROMPT SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nMinimal: task description + 'use Qiskit 2.x' (no API migration details)")
    print(f"Structured: version-specific Qiskit 2.x guidance + output format")
    print(f"\n{'Model':<20} {'Minimal':>10} {'Structured':>12} {'Δ':>8} {'Rank (M→S)':>12}")
    print("-"*65)

    base_ranked = sorted(models, key=lambda m: np.mean(baseline_scores[m]), reverse=True)
    struct_ranked = sorted(models, key=lambda m: np.mean(structured_scores[m]), reverse=True)

    for m in models:
        bm = np.mean(baseline_scores[m])
        sm = np.mean(structured_scores[m])
        br = base_ranked.index(m) + 1
        sr = struct_ranked.index(m) + 1
        delta = sm - bm
        arrow = "↑" if sr < br else ("↓" if sr > br else "=")
        print(f"{MODEL_LABELS.get(m,m):<20} {bm:>10.3f} {sm:>12.3f} {delta:>+8.3f} {br}→{sr} {arrow}")

    print(f"\n{'Ranking stability:':<25} Kendall τ = {tau:.3f} (p = {pval:.3f})")
    print(f"{'Rank inversions:':<25} {inv} (permutation p = {inv_pval:.4f})")

    if inv_pval < 0.05:
        print("\n⚠ The ranking change IS statistically significant (p < 0.05).")
        print("  The benchmark's model ordering depends on the system prompt.")
    else:
        print("\n✓ The ranking change is NOT statistically significant (p ≥ 0.05).")
        print("  The benchmark's model ordering is robust to this prompt variation.")

    # Key findings
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    biggest_gain = max(models, key=lambda m: np.mean(structured_scores[m]) - np.mean(baseline_scores[m]))
    biggest_loss = min(models, key=lambda m: np.mean(structured_scores[m]) - np.mean(baseline_scores[m]))
    gain_val = np.mean(structured_scores[biggest_gain]) - np.mean(baseline_scores[biggest_gain])
    loss_val = np.mean(structured_scores[biggest_loss]) - np.mean(baseline_scores[biggest_loss])
    print(f"  Biggest gain from structured prompt: {MODEL_LABELS[biggest_gain]} ({gain_val:+.3f})")
    print(f"  Biggest loss from structured prompt: {MODEL_LABELS[biggest_loss]} ({loss_val:+.3f})")
    print("="*80)


def main():
    print("Loading minimal-prompt results...")
    baseline_all = load_results(MINIMAL_FILE)
    print(f"  {len(baseline_all)} records")

    print("Loading structured-prompt results (full_benchmark_final)...")
    structured_all = load_results(STRUCTURED_FILE)
    print(f"  {len(structured_all)} records")

    if len(baseline_all) == 0:
        print("\n⚠ No minimal prompt results found yet. Run run_sensitivity_minimal.py first.")
        return

    # Filter to matched pairs only (fair comparison)
    baseline, structured, n_pairs = filter_to_matched_pairs(baseline_all, structured_all)
    print(f"\n  Matched pairs: {n_pairs} (comparing only identical task/model/level/seed)")
    n_tasks = len(set(r["task_name"] for r in structured))
    n_levels = len(set(r["level"] for r in structured))
    print(f"  Coverage: {n_tasks} tasks, {n_levels} levels")

    baseline_scores = compute_model_scores(baseline)
    structured_scores = compute_model_scores(structured)
    baseline_tm = compute_task_model_scores(baseline)
    structured_tm = compute_task_model_scores(structured)

    print("\nGenerating plots...")
    plot_paired_bars(baseline_scores, structured_scores)
    tau, pval = plot_rank_stability(baseline_scores, structured_scores)
    plot_task_sensitivity_heatmap(baseline_tm, structured_tm)

    print("\nComputing statistical significance...")
    inv, inv_pval = compute_rank_flip_significance(baseline_scores, structured_scores)

    print_summary(baseline_scores, structured_scores, tau, pval, inv, inv_pval)


if __name__ == "__main__":
    main()
