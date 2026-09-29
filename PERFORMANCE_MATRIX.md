# QC-Stark: Full Performance Matrix

**Dataset:** `results/full_benchmark_final.jsonl` — 2,750 evaluations (11 tasks × 10 models × 5 seeds × 5 difficulty levels), each with full prompt and raw LLM response stored. All models run at maximum supported token budgets (Aug 2026 rerun).

**Notation:** Mean accuracy across 25 instances (5 seeds × 5 levels). σ = std dev of per-seed accuracy (each seed averaged over 5 levels), n=5. **Bold** = best per task. *Italic* = worst per task.

---

## Call Failure Accounting (2,750 total evaluations)

| Failure Type | Count | % | Detail |
|---|---|---|---|
| Missing from file | **0** | 0.0% | All resolved: gemma-4-31b slots re-run via isolated VM (gemma_vm_results.jsonl verified). |
| API errors (model deprecated/404) | **0** | 0.0% | All resolved: claude-opus and mistral-large-3 retried with corrected IDs (`anthropic/claude-opus-4.1`, `mistral/mistral-large-3`). |
| API errors (proxy timeout) | **0** | 0.0% | All resolved: gemma-4-31b A_stateprep L5 s1/s2/s3 resolved via VM re-run. |
| Verifier timeouts | **46** | 1.7% | Concentrated in VQE and Trotterization tasks producing computationally expensive circuits. |
| Truncated at hard model caps | **11** | 0.4% | 5 Gemini 3.5-Flash (B1 L5), 2 GPT-4.1-mini, 4 LLaMA-70B. |
| gemini-flash-lite model swap | — | — | google/gemini-2.0-flash-lite-001 deprecated; replaced with google/gemini-3.1-flash-lite-global. All 275 slots rerun. |
| claude-sonnet-5 added | — | — | anthropic/claude-sonnet-5 (Jun 2026): 275/275 slots solid, 0 errors/timeouts. Merged as new row. |
| **Total unresolved gaps** | **0** | **0.0%** | All 2,750/2,750 evaluations complete. |
| Code crashes / wrong answers | 1,525 | 55.5% | Genuine model failures |
| **Correct answers** | **1,225** | **44.5%** | Verified from live data; includes 1 record recovered by the corrected code-extraction routine |

*All 2,750/2,750 target evaluations have valid records: 0 missing, 0 API errors.*

---

## Overall Accuracy Matrix — Mean ± σ

σ computed across 5 seeds; each seed's score = mean accuracy over 5 difficulty levels.

| Model | Family | Released | Avg ± σ | A | B1 | B3 | C1 | D1 | E2 | F1 | G1 | H1 | I1 | J1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **claude-sonnet-5** | standard | Jun 2026 | **.662 ±.047** | .68±.10 | **.60±.22** | **.64±.45** | .20±.18 | **.68±.27** | .60±.00 | **1.00±.00** | .24±.23 | **1.00±.00** | **1.00±.00** | .64±.08 |
| gemini-3.5-flash | standard | May 2026 | .651 ±.059 | **.92±.10** | .48±.20 | .60±.49 | .20±.00 | .60±.22 | .80±.00 | .84±.23 | .00±.00 | .96±.08 | **1.00±.00** | **.76±.08** |
| o4-mini | o-series | Apr 2025 | .593 ±.060 | .52±.10 | .00±.00 | .56±.46 | .32±.27 | **.68±.27** | .40±.00 | .88±.10 | **.64±.27** | .84±.08 | **1.00±.00** | .68±.10 |
| gpt-5.4 | o-series | Mar 2026 | .535 ±.083 | .40±.00 | .20±.13 | .60±.49 | .04±.08 | .64±.27 | **.88±.16** | .84±.08 | .36±.29 | .44±.15 | .80±.13 | .68±.10 |
| gemma-4-31b | standard | Apr 2026 | .491 ±.088 | .40±.00 | .00±.00 | .52±.45 | .28±.10 | .64±.37 | .56±.15 | .76±.20 | .52±.20 | .52±.10 | .72±.30 | .48±.10 |
| gpt-4.1-mini | standard | Apr 2025 | .393 ±.034 | .40±.00 | .00±.00 | .16±.15 | .20±.13 | .60±.28 | .40±.00 | .80±.00 | .20±.22 | .24±.08 | .84±.08 | .48±.10 |
| claude-opus (4.1) | standard | Aug 2025 | .382 ±.041 | .40±.00 | .00±.00 | .00±.00 | .16±.20 | .40±.13 | .56±.08 | .88±.10 | .24±.15 | .48±.10 | .76±.20 | .32±.10 |
| gemini-flash-lite (v3.1) | standard | Mar 2026 | .367 ±.067 | .44±.08 | .00±.00 | .32±.30 | **.44±.27** | .08±.16 | .48±.16 | **1.00±.00** | .04±.08 | .36±.15 | .48±.32 | .40±.13 |
| mistral-large-3 | standard | Dec 2025 | .262 ±.015 | .40±.00 | .00±.00 | .00±.00 | .12±.10 | .40±.13 | .40±.00 | .80±.00 | .00±.00 | .16±.08 | .04±.08 | .56±.15 |
| *llama-70b* | open | Dec 2024 | *.120 ±.009* | .40±.00 | *.04±.08* | .00±.00 | *.12±.10* | *.36±.08* | *.36±.08* | *.00±.00* | *.00±.00* | *.00±.00* | *.00±.00* | *.04±.08* |
| **Task mean** | — | — | **.445** | .50 | .13 | .34 | .21 | .51 | .54 | **.78** | .22 | .50 | .66 | .50 |

Bold = best per task. *Italic* = worst overall model (llama-70b). F1 two-way tie (claude-sonnet-5 and gemini-flash-lite both 1.00). I1 three-way tie (claude-sonnet-5, gemini-3.5-flash, o4-mini all 1.00). D1 two-way tie (claude-sonnet-5 and o4-mini both 0.68).

### Model Release References

| Model | API ID used | Released | Reference |
|---|---|---|---|
| claude-sonnet-5 | `anthropic/claude-sonnet-5` | Jun 30, 2026 | [Anthropic Docs](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5) |
| gemini-3.5-flash | `google/gemini-3.5-flash` | May 19, 2026 | [Google Blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/) |
| o4-mini | `openai/o4-mini-2025-04-16-global` | Apr 16, 2025 | [TechCrunch](https://techcrunch.com/2025/04/16/openai-launches-a-pair-of-ai-reasoning-models-o3-and-o4-mini/) |
| gpt-5.4 | `openai/gpt-5.4-2026-03-05` | Mar 5, 2026 | [TechCrunch](https://techcrunch.com/2026/03/05/openai-launches-gpt-5-4-with-pro-and-thinking-versions/) |
| gemma-4-31b | `google/gemma-4-31b` | Apr 2, 2026 | [Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) |
| gpt-4.1-mini | `openai/gpt-4.1-mini-2025-04-14` | Apr 14, 2025 | [Wikipedia](https://en.wikipedia.org/wiki/GPT-4.1) |
| claude-opus (4.1) | `anthropic/claude-opus-4.1` | Aug 5, 2025 | [Anthropic](https://www.anthropic.com/news/claude-opus-4-1) |
| gemini-flash-lite (v3.1) | `google/gemini-3.1-flash-lite-global` | Mar 3, 2026 | [Google Cloud Docs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite) |
| mistral-large-3 | `mistral/mistral-large-3` | Dec 2, 2025 | [Mistral AI](https://mistral.ai/news/mistral-3/) |
| llama-70b (3.3) | `meta/llama-3.3-70b` | Dec 6, 2024 | [TechCrunch](https://techcrunch.com/2024/12/06/meta-unveils-a-new-more-efficient-llama-model/) |

---

## Per-Level Accuracy (all 10 models averaged per task)

| Task | L1 Textbook | L2 Homework | L3 Exam | L4 Research | L5 Open | Cliff |
|---|---|---|---|---|---|---|
| A State Prep | 1.00 | 1.00 | 0.20 | 0.14 | 0.14 | **L2→L3** (catastrophic) |
| B1 Debugging | 0.16 | 0.16 | 0.16 | 0.16 | 0.02 | Gradual decline |
| B3 Noise/Logic | 0.28 | 0.36 | 0.38 | 0.34 | 0.34 | Flat |
| C1 Routing | 0.50 | 0.30 | 0.06 | 0.10 | 0.08 | **L2→L3** |
| D1 QEC | 0.64 | 0.62 | 0.60 | 0.34 | 0.34 | **L3→L4** |
| E2 VQE | 0.96 | 0.98 | 0.40 | 0.26 | 0.12 | **L2→L3** (catastrophic) |
| F1 Equivalence | 0.86 | 0.88 | 0.88 | 0.84 | 0.44 | **L4→L5** |
| G1 Trotter | 0.34 | 0.28 | 0.20 | 0.22 | 0.08 | Gradual decline |
| H1 Oracle | 0.88 | 0.46 | 0.44 | 0.46 | 0.24 | **L1→L2** (catastrophic) |
| I1 Noise Fidelity | 0.70 | 0.70 | 0.66 | 0.60 | 0.66 | Gradual decline |
| J1 Reverse Eng | 0.88 | 0.34 | 0.58 | 0.24 | 0.48 | **L1→L2** (catastrophic) |

---

## Task Difficulty Summary

| Task | Name | Mean Acc | Best Model | Rank |
|---|---|---|---|---|
| F1 | Equivalence | **0.780** | claude-sonnet-5 / gemini-flash-lite (1.000) | Easiest |
| I1 | Noise Fidelity | 0.664 | claude-sonnet-5 / gemini-3.5-flash / o4-mini (1.000) | 2 |
| E2 | VQE | 0.544 | gpt-5.4 (0.880) | 3 |
| D1 | QEC Decode | 0.508 | claude-sonnet-5 / o4-mini (0.680) | 4 |
| J1 | Reverse Eng | 0.504 | gemini-3.5-flash (0.760) | 5 |
| A | State Prep | 0.496 | gemini-3.5-flash (0.920) | 6 (tied) |
| H1 | Oracle Synth | 0.496 | claude-sonnet-5 (1.000) | 6 (tied) |
| B3 | Noise/Logic | 0.340 | claude-sonnet-5 (0.640) | 8 |
| G1 | Trotter | 0.224 | o4-mini (0.640) | 9 |
| C1 | Routing | 0.208 | gemini-flash-lite (0.440) | 10 |
| B1 | Debugging | **0.132** | claude-sonnet-5 (0.600) | Hardest |

**Difficulty range: 5.9× (B1 Debugging 13.2% → F1 Equivalence 78.0%)**

---

## Spearman ρ: Overall Rank vs Per-Task Rank

| Task | ρ | p | Interpretation |
|---|---|---|---|
| A State Prep | +0.683 | 0.030 | Predictive |
| B1 Debugging | +0.539 | 0.108 | Not significant (p>0.05) |
| B3 Noise/Logic | +0.917 | 0.000 | Highly predictive |
| C1 Routing | **+0.271** | 0.449 | Not predictive — weakest correlation |
| D1 QEC | +0.847 | 0.002 | Highly predictive |
| E2 VQE | +0.671 | 0.034 | Predictive |
| F1 Equivalence | +0.448 | 0.194 | Not significant (p>0.05) |
| G1 Trotter | +0.486 | 0.154 | Not significant (p>0.05) |
| H1 Oracle | +0.936 | 0.000 | Highly predictive |
| I1 Noise Fidelity | +0.926 | 0.000 | Highly predictive |
| J1 Reverse Eng | +0.787 | 0.007 | Highly predictive |

Task with smallest |ρ|: **C1 Routing** (+0.27, p=0.45) — near-zero correlation, structurally distinct from global capability axis.

---

## B3 Noise/Logic Deep Dive

B3 is 50/50 bug/no-bug (even seeds = bug, odd seeds = noise-only). Random baseline = 50%.

| Subset | n per model | Detail |
|---|---|---|
| Bug cases (even seeds) | 10 | Only claude-sonnet-5 fixes 1/10; all others 0/10 |
| Noise-only cases (odd seeds) | 15 | Top models correctly leave unchanged 87–100% |

**B3 per-model breakdown:**

| Model | Overall B3 | Bug acc | Noise acc |
|---|---|---|---|
| **claude-sonnet-5** | .64 | 0.10 | 1.00 |
| **gpt-5.4** | .60 | 0.00 | 1.00 |
| **gemini-3.5-flash** | .60 | 0.00 | 1.00 |
| **o4-mini** | .56 | 0.00 | 0.93 |
| **gemma-4-31b** | .52 | 0.00 | 0.87 |
| **gemini-flash-lite** | .32 | 0.00 | 0.53 |
| **gpt-4.1-mini** | .16 | 0.00 | 0.27 |
| **claude-opus** | .00 | 0.00 | 0.00 |
| **mistral-large-3** | .00 | 0.00 | 0.00 |
| **llama-70b** | .00 | 0.00 | 0.00 |

> B3 score primarily reflects noise-recognition ability; bug-case accuracy is 0.000 for 9/10 models. Claude Sonnet 5 is the sole model to fix any bug case (1/10).

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

**gpt-5.4 pattern:** zero genuine FPs at all effort levels. Instead, crash rate escalates with effort (0% → 11% → 56%). At high effort gpt-5.4 generates code so complex it fails to execute, rather than producing incorrect predictions. Two distinct structural failure modes.

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

### 1. Claude Sonnet 5 leads overall, Gemini 3.5 Flash close behind
Claude Sonnet 5 (66.2%) leads overall; Gemini 3.5 Flash (65.1%) is the biggest budget-fix beneficiary — jumped from 30.9% (#8 pre-fix) to 65.1% (#2). o4-mini (59.3%) rounds out the top 3. Different models top different tasks even within this tight cluster.

### 2. B1 Debugging is the hardest task (13.2% mean)
6 of 10 models score 0% on B1 Debugging even at maximum token budgets. Claude Sonnet 5 leads at 60%, followed by Gemini 3.5 Flash (48%), GPT-5.4 (20%), and LLaMA-70B (4%). The task requires reverse-engineering a unitary and isolating a single gate substitution or angle error. Spearman ρ = +0.54 (p=0.11, not significant).

### 3. Routing is not universally unsolvable
Gemini Flash Lite (v3.1) achieves 44% on C1 Routing, strong at L1 (80%) then collapsing. ρ = +0.27 (p=0.45) — the routing leaderboard bears essentially no relation to the global leaderboard (gpt-5.4 is 4th overall but only 4% on routing; Gemini FL is 8th overall but leads routing by a wide margin).

### 4. E2 VQE shows the most dramatic difficulty cliff
Near-perfect at L1 (96%) and L2 (98%), collapses to 40/26/12% at L3/L4/L5. No other task has this bimodal gap. Low-level VQE is essentially solved; research-level is far from it.

### 5. B3 bug cases are nearly universally unsolved
9 of 10 models score 0/10 on bug cases. Only Claude Sonnet 5 fixes 1/10. Models can recognize noise-only circuits 87–100% of the time (top 4 models), but cannot triangulate the bug location from noisy output distributions.

### 6. Gemini 3.5 Flash: biggest budget-fix beneficiary
Gemini 3.5 Flash was severely truncated at the original budget. After rerun at 65K tokens: I1 Noise Fidelity jumped from 0% to 100%, State Prep from 40% to 92%, overall from 30.9% to 65.1%.

### 7. D1 effort paradox: two different failure modes
o4-mini over-corrects (false positives) most severely at low effort; more reasoning suppresses false intervention. gpt-5.4 never over-corrects but generates increasingly complex, non-executing code as effort rises. The "more reasoning → more hallucination" hypothesis applies to o4-mini but manifests as code crashes for gpt-5.4.

---

## Data Sources

| File | Rows | Description |
|---|---|---|
| `results/full_benchmark_final.jsonl` | 2,750 | Main benchmark, structured (canonical) prompt. 2,750/2,750 complete, no duplicate keys, 0 API errors. Backs Table 1, the error taxonomy, the difficulty breakdown, and the IRT analysis. |
| `results/sensitivity_minimal_final.jsonl` | 2,750 | Minimal-prompt condition, same grid. Paired against the file above for the prompt-sensitivity analysis (2,745 usable pairs). Raw responses are not retained for this condition. |

Total evaluations in the released results: **5,500** (2,750 per prompt condition).

*Supporting data held outside the release (`results/archive/`, not published):
`paradox_effort_sweep.jsonl` (540 clean records, F1/D1 null-case effort sweep) and
`debugging_paradox_deep/` (710 records across 9 experiment files: o3, o3-mini,
GPT-5.5, GPT-5.6-luna, Gemini 2.5-Pro, Sonnet 5). The latter is the only backing
for the debugging-paradox figures quoted in the main text, so it must be
published alongside the paper or those numbers are unverifiable by a reader.*

*Archived (see `archive_code/stale_results/`): `b3_pilot.jsonl` (144) and `b3_effort_replication.jsonl` (542). Both were generated under prompt variants that differ from the canonical `SYSTEM_PROMPT` — and from each other — so they are not directly comparable to the main benchmark. No claim in this document depends on them; the B3 bug-case result above is computed from `full_benchmark_final.jsonl`.*

*All 2,750/2,750 records present: 0 missing, 0 API errors. All models run at maximum supported token budgets.*
*Effort sweep: 540/540 records present, 0 API errors, all slots filled.*
