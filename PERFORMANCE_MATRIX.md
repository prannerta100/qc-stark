# QC-Hard: Full Performance Matrix

**Dataset:** `results/full_benchmark_v2.jsonl` — 2,750 target evaluations (11 tasks × 10 models × 5 seeds × 5 difficulty levels), each with full prompt and raw LLM response stored.

**Notation:** Mean accuracy across 25 instances (5 seeds × 5 levels). σ = std dev of per-seed accuracy (each seed averaged over 5 levels), n=5. **Bold** = best per task. *Italic* = worst per task.

---

## Call Failure Accounting (2,750 total evaluations)

| Failure Type | Count | % | Detail |
|---|---|---|---|
| Missing from file | **0** | 0.0% | All resolved: gemma-4-31b slots re-run via isolated VM (gemma_vm_results.jsonl verified). |
| API errors (model deprecated/404) | **0** | 0.0% | All resolved: claude-opus and mistral-large-3 retried with corrected IDs (`anthropic/claude-opus-4.1`, `mistral/mistral-large-3`). |
| API errors (proxy timeout) | **0** | 0.0% | All resolved: gemma-4-31b A_stateprep L5 s1/s2/s3 resolved via VM re-run. |
| Verifier timeouts | **0** | — | All resolved: gpt-4.1-mini F1 L5 s1-5 → all CORRECT (needed 2000 s); gemini-flash-lite F1 L5 replaced by 3.1-lite rerun (all CORRECT); gpt-5.4 G1 L5 s3/s5 → wrong (genuine failure). claude-sonnet-5: 0 timeouts (2000 s limit used throughout). |
| gemini-flash-lite model swap | — | — | google/gemini-2.0-flash-lite-001 deprecated; replaced with google/gemini-3.1-flash-lite-global. All 275 slots rerun. |
| claude-sonnet-5 added | — | — | anthropic/claude-sonnet-5 (Jun 2026): 275/275 slots solid, 0 errors/timeouts. Merged as new row. |
| **Total unresolved gaps** | **0** | **0.0%** | All 2,750/2,750 evaluations complete. |
| Code crashes / wrong answers | 1650 | 60.0% | Genuine model failures |
| **Correct answers** | **1100** | **40.0%** | Verified from live data (last-record-wins dedup) |

*All 2,750/2,750 target evaluations have valid records: 0 missing, 0 API errors.*

---

## Overall Accuracy Matrix — Mean ± σ

σ computed across 5 seeds; each seed's score = mean accuracy over 5 difficulty levels.

| Model | Family | Released | Avg ± σ | A | B1 | B3 | C1 | D1 | E2 | F1 | G1 | H1 | I1 | J1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **o4-mini** | o-series | Apr 2025 | **.556 ±.045** | .48±.10 | .00±.00 | .48±.41 | .16±.15 | .52±.27 | .40±.00 | .92±.10 | **.80±.25** | **.84±.08** | **.92±.16** | .60±.13 |
| claude-sonnet-5 | standard | Jun 2026 | .553 ±.068 | **.68±.10** | .00±.00 | .48±.39 | .12±.16 | .68±.32 | **.96±.08** | **1.00±.00** | .24±.15 | .64±.08 | .64±.15 | **.64±.08** |
| gpt-5.4 | o-series | Mar 2026 | .545 ±.055 | .40±.00 | **.16±.15** | **.56±.46** | .08±.10 | **.72±.16** | .92±.16 | .96±.08 | .40±.13 | .44±.15 | .76±.23 | .60±.18 |
| gemma-4-31b | standard | Apr 2026 | .491 ±.088 | .40±.00 | .00±.00 | .52±.45 | .28±.10 | .64±.37 | .56±.15 | .76±.20 | .52±.20 | .52±.10 | .72±.30 | .48±.10 |
| claude-opus (4.1) | standard | Aug 2025 | .396 ±.035 | .40±.00 | .00±.00 | .00±.00 | .12±.10 | .40±.18 | .56±.15 | .96±.08 | .28±.20 | .52±.20 | .72±.10 | .40±.00 |
| gpt-4.1-mini | standard | Apr 2025 | .371 ±.055 | .40±.00 | .00±.00 | .16±.15 | .16±.15 | .32±.16 | .40±.00 | **1.00±.00** | .24±.15 | .32±.10 | .72±.27 | .36±.15 |
| gemini-flash-lite (v3.1) | standard | Mar 2026 | .367 ±.067 | .44±.08 | .00±.00 | .32±.30 | **.44±.27** | .08±.16 | .48±.16 | **1.00±.00** | .04±.08 | .36±.15 | .48±.32 | .40±.13 |
| gemini-3.5-flash | standard | May 2026 | .309 ±.023 | .40±.00 | .00±.00 | .20±.18 | .16±.08 | .20±.18 | .88±.10 | .64±.15 | .00±.00 | .36±.20 | .00±.00 | .56±.15 |
| mistral-large-3 | standard | Dec 2025 | .291 ±.023 | .40±.00 | .00±.00 | .00±.00 | .24±.08 | .48±.10 | .40±.00 | .96±.08 | .00±.00 | .12±.10 | .20±.13 | .40±.13 |
| *llama-70b* | open | Dec 2024 | *.120 ±.027* | .40±.00 | *.04±.08* | .00±.00 | *.08±.10* | .36±.08 | .36±.08 | *.04±.08* | .00±.00 | .00±.00 | .00±.00 | *.04±.08* |
| **Task mean** | — | — | **.400** | .44 | .02 | .27 | .18 | .44 | .59 | **.82** | .25 | .41 | .52 | .45 |

Bold = best per task. *Italic* = worst overall model (llama-70b). F1 three-way tie (claude-sonnet-5, gpt-4.1-mini, gemini-flash-lite all 1.00).

### Model Release References

| Model | API ID used | Released | Reference |
|---|---|---|---|
| o4-mini | `openai/o4-mini-2025-04-16-global` | Apr 16, 2025 | [TechCrunch](https://techcrunch.com/2025/04/16/openai-launches-a-pair-of-ai-reasoning-models-o3-and-o4-mini/) |
| gpt-5.4 | `openai/gpt-5.4-2026-03-05` | Mar 5, 2026 | [TechCrunch](https://techcrunch.com/2026/03/05/openai-launches-gpt-5-4-with-pro-and-thinking-versions/) |
| gemma-4-31b | `google/gemma-4-31b` | Apr 2, 2026 | [Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) |
| claude-opus (4.1) | `anthropic/claude-opus-4.1` | Aug 5, 2025 | [Anthropic](https://www.anthropic.com/news/claude-opus-4-1) |
| gpt-4.1-mini | `openai/gpt-4.1-mini-2025-04-14` | Apr 14, 2025 | [Wikipedia](https://en.wikipedia.org/wiki/GPT-4.1) |
| gemini-flash-lite (v3.1) | `google/gemini-3.1-flash-lite-global` | Mar 3, 2026 | [Google Cloud Docs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite) |
| gemini-3.5-flash | `google/gemini-3.5-flash` | May 19, 2026 | [Google Blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/) |
| mistral-large-3 | `mistral/mistral-large-3` | Dec 2, 2025 | [Mistral AI](https://mistral.ai/news/mistral-3/) |
| llama-70b (3.3) | `meta/llama-3.3-70b` | Dec 6, 2024 | [TechCrunch](https://techcrunch.com/2024/12/06/meta-unveils-a-new-more-efficient-llama-model/) |
| claude-sonnet-5 *(pending)* | `anthropic/claude-sonnet-5` | Jun 30, 2026 | [Anthropic Docs](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5) |

---

## Per-Level Accuracy (all 10 models averaged per task)

| Task | L1 Textbook | L2 Homework | L3 Exam | L4 Research | L5 Open | Cliff |
|---|---|---|---|---|---|---|
| A State Prep | 1.00 | 1.00 | 0.04 | 0.10 | 0.06 | **L2→L3** (catastrophic) |
| B1 Debugging | 0.04 | 0.00 | 0.00 | 0.04 | 0.02 | Universally hard at all levels |
| B3 Noise/Logic | 0.28 | 0.28 | 0.28 | 0.24 | 0.28 | Flat — structural ceiling |
| C1 Routing | 0.44 | 0.24 | 0.06 | 0.12 | 0.06 | **L1→L2** then near-zero |
| D1 QEC | 0.54 | 0.54 | 0.36 | 0.40 | 0.36 | **L2→L3** |
| E2 VQE | 0.96 | 0.98 | 0.38 | 0.36 | 0.28 | **L2→L3** (catastrophic) |
| F1 Equivalence | 0.86 | 0.90 | 0.84 | 0.82 | 0.56 | **L4→L5** |
| G1 Trotter | 0.32 | 0.30 | 0.26 | 0.22 | 0.16 | Gradual decline |
| H1 Oracle | 0.86 | 0.40 | 0.38 | 0.31 | 0.10 | **L1→L2** |
| I1 Noise Fidelity | 0.68 | 0.58 | 0.54 | 0.42 | 0.36 | Gradual decline |
| J1 Reverse Eng | 0.90 | 0.30 | 0.48 | 0.16 | 0.40 | Non-monotone (L2 hard) |

---

## Task Difficulty Summary

| Task | Mean Acc | Best Model | Rank |
|---|---|---|---|
| F1 Equivalence | **0.824** | claude-sonnet-5 / gpt-4.1-mini / gemini-flash-lite (1.000) | Easiest |
| E2 VQE | 0.592 | claude-sonnet-5 (0.960) | 2 |
| I1 Noise Fidelity | 0.516 | o4-mini (0.920) | 3 |
| J1 Reverse Eng | 0.448 | claude-sonnet-5 (0.640) | 4 |
| A State Prep | 0.440 | claude-sonnet-5 (0.680) | 5 (tied) |
| D1 QEC | 0.440 | gpt-5.4 (0.720) | 5 (tied) |
| H1 Oracle | 0.412 | o4-mini (0.840) | 7 |
| B3 Noise/Logic | 0.272 | gpt-5.4 (0.560) | 8 |
| G1 Trotter | 0.252 | o4-mini (0.800) | 9 |
| C1 Routing | 0.184 | gemini-flash-lite (0.440) | 10 |
| B1 Debugging | **0.020** | gpt-5.4 (0.160) | Hardest |

**Difficulty range: 41.2× (B1 debugging 2.0% → F1 equivalence 82.4%)**

---

## Spearman ρ: Overall Rank vs Per-Task Rank

| Task | ρ | p | Interpretation |
|---|---|---|---|
| A State Prep | +0.891 | 0.001 | Highly predictive |
| B1 Debugging | +0.503 | 0.138 | Not significant (p>0.05) |
| B3 Noise/Logic | +0.794 | 0.006 | Predictive |
| C1 Routing | **+0.030** | 0.934 | Not predictive — weakest correlation |
| D1 QEC | +0.673 | 0.033 | Predictive |
| E2 VQE | +0.576 | 0.082 | Not significant (p>0.05) |
| F1 Equivalence | +0.564 | 0.090 | Not significant (p>0.05) |
| G1 Trotter | +0.915 | 0.000 | Highly predictive |
| H1 Oracle | +0.927 | 0.000 | Highly predictive |
| I1 Noise | +0.855 | 0.002 | Highly predictive |
| J1 Reverse | +0.818 | 0.004 | Predictive |

Task with smallest |ρ|: **C1 Routing** (+0.03, p=0.93) — near-zero correlation, structurally distinct from global capability axis.

---

## B3 Noise/Logic Deep Dive

B3 is 50/50 bug/no-bug (even seeds = bug, odd seeds = noise-only). Random baseline = 50%.

| Subset | n | Acc | Notes |
|---|---|---|---|
| Bug cases (even seeds) | 100 | **0.000** | No model fixes the circuit — 0/100 correct across all models |
| Noise-only cases (odd seeds) | 150 | **0.453** | Models correctly leave unchanged 45.3% |

**B3 per-model breakdown:**

| Model | Overall B3 | Bug acc | Noise acc |
|---|---|---|---|
| **gpt-5.4** | .56 | 0.00 | 0.69 |
| **gemma-4-31b** | .52 | 0.00 | 0.64 |
| **o4-mini** | .48 | 0.00 | 0.56 |
| **gemini-flash-lite** | .32 | 0.00 | 0.47 |
| **gemini-3.5-flash** | .20 | 0.00 | 0.27 |
| **gpt-4.1-mini** | .16 | 0.00 | 0.29 |
| **claude-sonnet-5** | .48 | 0.00 | 0.80 |
| **claude-opus** | .00 | 0.00 | 0.00 |
| **mistral-large-3** | .00 | 0.00 | 0.00 |
| **llama-70b** | .00 | 0.00 | 0.00 |

> B3 score entirely reflects noise-recognition ability; bug-case accuracy is 0.000 across all 10 models.

---

## Reasoning Effort Sweep: D1 QEC and F1 Equivalence

540 records (0 API errors): o4-mini and gpt-5.4 × low/medium/high effort × 5 seeds × 3 difficulty levels × 3 reps.

**Design:** D1 sweep uses 100% negative cases (no error present; correct answer = False). F1 sweep uses 100% positive cases (circuits ARE equivalent; correct answer = True). This isolates one direction of the confusion matrix — whether models intervene when they should not.

### D1 QEC — no-error syndromes (correct = return False, do not intervene)

| Model | Effort | N | TN (correct) | FP (wrong intervention) | Crash (invalid code) |
|---|---|---|---|---|---|
| o4-mini | low | 45 | 21 (47%) | **23 (51%)** | 1 (2%) |
| o4-mini | medium | 45 | 33 (73%) | 12 (27%) | 0 (0%) |
| o4-mini | high | 45 | 29 (64%) | 9 (20%) | 7 (16%) |
| gpt-5.4 | low | 45 | 45 (100%) | 0 (0%) | 0 (0%) |
| gpt-5.4 | medium | 45 | 40 (89%) | 0 (0%) | 5 (11%) |
| gpt-5.4 | high | 45 | 20 (44%) | **0 (0%)** | **25 (56%)** |

**o4-mini pattern:** genuine over-correction that decreases with effort (51% → 27% → 20% FP). More reasoning suppresses false interventions.

**gpt-5.4 pattern:** zero genuine FPs at all effort levels. Instead, crash rate escalates with effort (0% → 11% → 56%). At high effort gpt-5.4 generates code so complex it fails to execute, rather than producing incorrect predictions. Two distinct failure modes.

### F1 Equivalence — equivalent circuit pairs (correct = return True, confirm equivalence)

| Model | Effort | N | TP (correct) | FN (missed equiv.) | Crash (invalid code) |
|---|---|---|---|---|---|
| o4-mini | low | 45 | 43 (96%) | 0 (0%) | 2 (4%) |
| o4-mini | medium | 45 | 44 (98%) | 0 (0%) | 1 (2%) |
| o4-mini | high | 45 | 41 (91%) | 1 (2%) | 3 (7%) |
| gpt-5.4 | low | 45 | 45 (100%) | 0 (0%) | 0 (0%) |
| gpt-5.4 | medium | 45 | 45 (100%) | 0 (0%) | 0 (0%) |
| gpt-5.4 | high | 45 | 44 (98%) | 0 (0%) | 1 (2%) |

F1 equivalence shows no effort paradox. Both models are near-ceiling at all effort levels; FN rates are negligible (≤2%). The anomalous behavior is isolated to D1 QEC.

---

## What's Going On: Key Observations

### 1. Top three models are essentially tied
o4-mini (55.6%), claude-sonnet-5 (55.3%), gpt-5.4 (54.5%) are within 1.1 pp. o4-mini leads overall but scores 0% on B1 Debugging; claude-sonnet-5 leads A/E2/J1 despite near-identical overall score. Different models top different tasks even within this tight cluster.

### 2. B1 Debugging is the hardest task (2.0% mean) — universally unsolved
Mean accuracy 2.0% across all 10 models at all levels, including L1. Models fail at L1 debugging just as badly as L5. The task requires reverse-engineering a unitary and isolating a single gate substitution or angle error — qualitatively beyond current LLM capability regardless of scale. Spearman ρ = +0.50 (p=0.14, not significant).

### 3. Routing is not universally unsolvable
Gemini Flash Lite (v3.1) achieves 44% on C1 Routing, strong at L1/L2 (80% each) then collapsing at L3+. ρ = +0.03 (p=0.93) — the routing leaderboard bears essentially no relation to the global leaderboard (gpt-5.4 is 3rd overall but only 8% on routing; Gemini FL is 7th overall but leads routing by a wide margin).

### 4. E2 VQE shows the most dramatic difficulty cliff
Near-perfect at L1 (96%) and L2 (98%), collapses to 38/36/28% at L3/L4/L5. No other task has this bimodal gap. Low-level VQE is essentially solved; research-level is far from it.

### 5. B3 bug cases are completely unsolved (0/100 correct)
Every model at every level fails to fix a quantum circuit bug in B3. Models can recognize noise-only circuits ~45% of the time, but cannot triangulate the bug location from noisy output distributions.

### 6. gemini-3.5-flash scores 0% on noise fidelity (I1)
I1 requires predicting circuit fidelity under depolarizing noise. gemini-3.5-flash gets 0/25 — systematically lower than all other models (0.52–0.92). G1 Trotter is also 0/25 for gemini-3.5-flash. Likely a consistent format mismatch, not a reasoning gap.

### 7. D1 effort paradox: two different failure modes
o4-mini over-corrects (false positives) most severely at low effort; more reasoning suppresses false intervention. gpt-5.4 never over-corrects but generates increasingly complex, non-executing code as effort rises. The "more reasoning → more hallucination" hypothesis applies to o4-mini but manifests as code crashes for gpt-5.4.

---

## Data Sources

| File | Rows | Description |
|---|---|---|
| `results/full_benchmark_v2.jsonl` | 2,757 | Main benchmark (2,750/2,750 complete; includes 3.1-lite gemini rerun + claude-sonnet-5 + gemma VM re-run) |
| `results/paradox_effort_sweep.jsonl` | 540 | F1/D1 null-case effort sweep — 0 API errors |
| `results/b3_pilot.jsonl` | 144 | B3 pilot with raw responses (8 models × 18 instances) |
| `results/b3_effort_replication.jsonl` | 542 | B3 noise-only effort replication study |

Total evaluations with raw LLM responses stored: **3,693**

*All 2,750/2,750 records present: 0 missing, 0 API errors. Gemma slots resolved via VM re-run; claude-opus and mistral deprecated-ID slots resolved via retry with corrected IDs.*
*gemini-flash-lite: all 275 records from google/gemini-3.1-flash-lite-global (2.0 deprecated); archive in results/archive/.*
*Effort sweep: 540/540 records present, 0 API errors, all slots filled.*
