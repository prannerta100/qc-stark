"""QC-Stark final benchmark numbers — the single analysis script.

Reads results/full_benchmark_final.jsonl and prints every number the paper
reports from the main benchmark: the per-model x per-task accuracy matrix
(with the LaTeX body for Table 1), task means, the error taxonomy, the
difficulty-level breakdown, and the overall-vs-per-task rank inversions.

Conventions, matched to the paper:
  * accuracy is binarized `correct`, not the continuous `score`
  * sigma is the POPULATION std dev (ddof=0) across the 5 seeds, where each
    seed's value is its mean over the 5 difficulty levels
"""
import json, math, statistics, collections
from scipy.stats import spearmanr

RESULTS = "results/full_benchmark_final.jsonl"
TASKS = [("T1","A_stateprep"),("T2","G1_trotter"),("T3","H1_oracle"),("T4","B1_debugging"),
         ("T5","B3_noise_logic"),("T6","J1_reverse"),("T7","F1_equivalence"),("T8","C1_routing"),
         ("T9","I1_noise"),("T10","E2_vqe"),("T11","D1_qec")]
MODELS = [("Sonnet~5","claude-sonnet-5"),("Gemini~3.5F","gemini-3.5-flash"),("o4-mini","o4-mini"),
          ("GPT-5.4","gpt-5.4"),("Gemma-4","gemma-4-31b"),("GPT-4.1m","gpt-4.1-mini"),
          ("Opus~4.1","claude-opus"),("Gemini~FL","gemini-flash-lite"),
          ("Mistral~L3","mistral-large-3"),("LLaMA-70B","llama-70b")]


def load():
    return [json.loads(l) for l in open(RESULTS) if l.strip()]


def acc(r):
    return 1.0 if str(r["correct"]).lower() == "true" else 0.0


def seed_sigma(rows, model, task=None):
    per = collections.defaultdict(list)
    for r in rows:
        if r["model"] != model or (task and r["task_name"] != task):
            continue
        per[str(r["seed"])].append(acc(r))
    return statistics.pstdev([statistics.mean(v) for v in per.values()])


def main():
    rows = load()
    cell = collections.defaultdict(list)
    for r in rows:
        cell[(r["task_name"], r["model"])].append(acc(r))

    print(f"records: {len(rows)}   unique keys: "
          f"{len({(r['task_name'],r['model'],str(r['level']),str(r['seed'])) for r in rows})}")

    print("\n--- Table 1 body (LaTeX) ---")
    for disp, m in MODELS:
        cells = " & ".join(f"{statistics.mean(cell[(t,m)]):.2f}".lstrip("0") for _, t in TASKS)
        ov = statistics.mean([acc(r) for r in rows if r["model"] == m])
        sd = seed_sigma(rows, m)
        print(f"{disp:13} & {cells} & {ov:.3f}".replace("& 0.", "& .")
              + f"$\\pm${sd:.3f}".replace("$\\pm$0.", "$\\pm$.") + " \\\\")
    tm = " & ".join(f"{statistics.mean([acc(r) for r in rows if r['task_name']==t]):.2f}".lstrip("0")
                    for _, t in TASKS)
    print(f"{'Task mean':13} & {tm} & "
          f"{statistics.mean([acc(r) for r in rows]):.3f}".replace("& 0.", "& .") + " \\\\")

    print("\n--- error taxonomy ---")
    det = lambda r: r["details"] if isinstance(r.get("details"), dict) else {}
    corr = int(sum(acc(r) for r in rows))
    to = sum(1 for r in rows if det(r).get("verifier") == "timeout")
    tr = sum(1 for r in rows if r.get("finish_reason") == "length")
    ae = sum(1 for r in rows if "api_error" in det(r))
    for label, n in (("Correct answer", corr), ("Wrong answer / code crash", len(rows)-corr-to-tr-ae),
                     ("Verifier timeout (>300 s)", to), ("Truncated at model cap", tr),
                     ("API error", ae)):
        print(f"  {label:28} {n:5}  {n/len(rows)*100:5.1f}%")

    print("\n--- difficulty ---")
    lv = [statistics.mean([acc(r) for r in rows if int(r["level"]) == l]) for l in range(1, 6)]
    print("  " + "  ".join(f"L{i+1}={v:.3f}" for i, v in enumerate(lv)))
    print(f"  L1->L5 drop: {(lv[0]-lv[4])/lv[0]*100:.0f}%")

    print("\n--- overall vs per-task rank correlation (n=10 models) ---")
    overall = {m: statistics.mean([acc(r) for r in rows if r["model"] == m]) for _, m in MODELS}
    order = sorted(overall, key=lambda m: -overall[m])
    ov = [overall[m] for m in order]
    ns = 0
    for code, t in TASKS:
        rho, p = spearmanr(ov, [statistics.mean(cell[(t, m)]) for m in order])
        sig = "" if p < 0.05 else "  non-significant"
        ns += p >= 0.05
        print(f"  {code:4} {t:16} rho={rho:+.3f}  p={p:.4f}{sig}")
    print(f"  -> {ns} of {len(TASKS)} non-significant at alpha=0.05 (uncorrected)")

    best = max(overall, key=overall.get)
    tmeans = {t: statistics.mean([acc(r) for r in rows if r["task_name"] == t]) for _, t in TASKS}
    hi, lo = max(tmeans, key=tmeans.get), min(tmeans, key=tmeans.get)
    print(f"\n--- headline ---")
    print(f"  best model: {best} {overall[best]:.3f}")
    print(f"  task span : {hi} {tmeans[hi]:.3f} / {lo} {tmeans[lo]:.3f} = {tmeans[hi]/tmeans[lo]:.1f}x")
    print(f"  overall   : {statistics.mean([acc(r) for r in rows]):.4f}")


if __name__ == "__main__":
    main()
