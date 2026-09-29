# QC-Stark: A Multi-Task Benchmark for LLM Evaluation on Quantum Computing

QC-Stark is a benchmark that evaluates large language models on 11 quantum computing tasks spanning the full practitioner workflow: circuit construction, debugging, compilation, verification, simulation, and error correction.

All numbers below are computed from `results/full_benchmark_final.jsonl` (2,750 evaluations; 10 models x 11 tasks x 5 levels x 5 seeds), with every model run at its maximum supported token budget.

## Key Findings (2,750 evaluations across 10 models)

- **Capability dissociation is real and large.** o4-mini ranks 3rd overall (59.3%) yet scores **0%** on circuit debugging. Six of ten models score exactly 0% on debugging; last-ranked LLaMA-70B (12.0% overall) outscores all six of them on that task.
- **Task difficulty spans 5.9x** from Equivalence Checking (78.0%) to Debugging (13.2%). Performance drops 59% from Level 1 (65.5%) to Level 5 (26.7%).
- **Hardware routing remains unsolved** at research difficulty (mean 0.080 at Level 3+). Gemini Flash Lite (8th overall) leads the task at 44%.
- **Psychometrically validated.** A 2PL IRT model treating each of the 55 (task, level) cells as an item -- 120 parameters (10 abilities, 55 difficulties, 55 discriminations) over 2,750 responses -- places model ability and item difficulty on one logit scale. Gemini 3.5 Flash leads on IRT ability (theta = 1.122) while Claude Sonnet 5 leads on mean accuracy (0.662). 15 of the 53 identified items sit above the ablest model's ability, so the benchmark retains headroom.
- **Rankings are robust to prompt wording.** Against a minimal prompt over the same grid (2,745 paired evaluations), Spearman rho = 0.915 and Kendall tau = 0.778. The top-4 grouping is unchanged; the bottom-4 is not.

## Tasks

| # | Task | Workflow Stage | Mean Accuracy |
|---|---|---|---|
| T1 | State Preparation | Construction | 0.496 |
| T2 | Trotterization | Construction | 0.224 |
| T3 | Oracle Synthesis | Construction | 0.500 |
| T4 | Debugging | Understanding | 0.132 |
| T5 | Noise Discrimination | Understanding | 0.340 |
| T6 | Reverse Engineering | Understanding | 0.504 |
| T7 | Equivalence Checking | Verification | 0.780 |
| T8 | Hardware Routing | Compilation | 0.208 |
| T9 | Noise Fidelity | Simulation | 0.664 |
| T10 | VQE | Simulation | 0.544 |
| T11 | QEC Decoding | Error Correction | 0.508 |

## Repository Contents

- `qc_stark_dataset.jsonl` -- 275 benchmark instances (11 tasks x 5 difficulty levels x 5 seeds). Each record contains: task ID, task name, workflow stage, difficulty level, seed, and the full evaluation prompt. Models must return a `solve()` function in Qiskit 2.x.
- `PERFORMANCE_MATRIX.md` -- Full results matrix for all 10 models across all 11 tasks, including per-seed standard deviations, error taxonomy, IRT analysis, and difficulty-level breakdowns.
- `qc_hard/prompts.py` -- **Authoritative system prompts.** `SYSTEM_PROMPT` is the canonical string that produced every headline number; do not redefine it elsewhere.

## Dataset Schema

Each line of `qc_stark_dataset.jsonl` is a JSON object with:

```
{
  "task_id":        "T1",                  // T1-T11
  "task_code":      "A_stateprep",         // internal code
  "task_name":      "State Preparation",   // human-readable name
  "workflow_stage": "Construct",           // Construct | Understand | Verify | Compile | Simulate | Error Correction
  "level":          1,                     // 1-5 (Textbook -> Open)
  "difficulty":     "Textbook",            // Textbook | Homework | Exam | Research | Open
  "seed":           1,                     // 1-5
  "prompt":         "Write a Qiskit ..."   // full evaluation prompt
}
```

## Evaluation Protocol

1. Each instance is deterministically generated from a (task, level, seed) triple.
2. Models receive the canonical system prompt (`SYSTEM_PROMPT` in `qc_hard/prompts.py`), which carries Qiskit 2.x API guidance, plus the task-specific prompt.
3. Models must return executable Python code defining a `solve()` function.
4. Verification executes the code and checks output against ground truth (state fidelity, functional equivalence, correct identification). No human judgment required.

## Models Evaluated

| Rank | Model | Overall Accuracy | IRT Ability |
|---|---|---|---|
| 1 | Claude Sonnet 5 | 0.662 | +1.090 |
| 2 | Gemini 3.5 Flash | 0.651 | +1.122 |
| 3 | o4-mini | 0.593 | +0.669 |
| 4 | GPT-5.4 | 0.535 | +0.432 |
| 5 | Gemma-4 31B | 0.491 | +0.251 |
| 6 | GPT-4.1-mini | 0.393 | -0.087 |
| 7 | Claude Opus 4.1 | 0.382 | -0.021 |
| 8 | Gemini Flash Lite | 0.367 | -0.141 |
| 9 | Mistral Large 3 | 0.262 | -0.843 |
| 10 | LLaMA-3.3 70B | 0.120 | -2.473 |

Gemini 3.5 Flash leads on IRT ability but ranks 2nd on accuracy. The orderings differ because IRT weights each item by how sharply it separates models, rather than averaging all items equally. The two are within 0.032 logits of each other, well inside their confidence intervals, so the top-2 order is not resolved by this data.

## Citation

If you use QC-Stark in your research, please cite:

```
@inproceedings{gupta2026qcstark,
  title={QC-Stark: A Multi-Task Benchmark Revealing Capability Dissociations in LLMs for Quantum Computing},
  author={Gupta, Pranav},
  booktitle={Proceedings of qnlp.ai 2026},
  year={2026}
}
```

## License

This benchmark is released for research purposes.
