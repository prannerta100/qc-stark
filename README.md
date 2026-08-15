# QC-Stark: A Multi-Task Benchmark for LLM Evaluation on Quantum Computing

QC-Stark is a benchmark that evaluates large language models on 11 quantum computing tasks spanning the full practitioner workflow: circuit construction, debugging, compilation, verification, simulation, and error correction.

## Key Findings (2,750 evaluations across 10 models)

- **Capability dissociation is real and large.** The best overall model (o4-mini, 55.6%) scores 0% on circuit debugging. Four of five top-ranked models score 0% on debugging, while last-ranked LLaMA-70B outperforms them.
- **Task difficulty spans 41.2x** from Equivalence Checking (82.4%) to Debugging (2.0%). Performance drops 62% from Level 1 to Level 5.
- **Hardware routing remains unsolved** at research difficulty (mean 0.08 at Level 3+). Gemini Flash Lite (7th overall) leads at 44%.
- **Psychometrically validated.** IRT marginal reliability = 0.984. Rankings robust to prompt format (Spearman rho = 0.95).

## Tasks

| # | Task | Workflow Stage | Mean Accuracy |
|---|---|---|---|
| T1 | State Preparation | Construction | 0.440 |
| T2 | Trotterization | Construction | 0.252 |
| T3 | Oracle Synthesis | Construction | 0.412 |
| T4 | Debugging | Understanding | 0.020 |
| T5 | Noise Discrimination | Understanding | 0.272 |
| T6 | Reverse Engineering | Understanding | 0.448 |
| T7 | Equivalence Checking | Verification | 0.824 |
| T8 | Hardware Routing | Compilation | 0.184 |
| T9 | Noise Fidelity | Simulation | 0.516 |
| T10 | VQE | Simulation | 0.592 |
| T11 | QEC Decoding | Error Correction | 0.440 |

## Repository Contents

- `qc_stark_dataset.jsonl` -- 275 benchmark instances (11 tasks x 5 difficulty levels x 5 seeds). Each record contains: task ID, task name, workflow stage, difficulty level, seed, and the full evaluation prompt. Models must return a `solve()` function in Qiskit 2.x.
- `PERFORMANCE_MATRIX.md` -- Full results matrix for all 10 models across all 11 tasks, including per-seed standard deviations, error taxonomy, IRT analysis, and difficulty-level breakdowns.
- `paper/poster.pdf` -- Conference poster with visual summary of results.

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
2. Models receive a system prompt with Qiskit 2.x API guidance plus the task-specific prompt.
3. Models must return executable Python code defining a `solve()` function.
4. Verification executes the code and checks output against ground truth (state fidelity, functional equivalence, correct identification). No human judgment required.

## Models Evaluated

| Rank | Model | Overall Accuracy | IRT Ability |
|---|---|---|---|
| 1 | o4-mini | 0.556 | +0.649 |
| 2 | Claude Sonnet 5 | 0.553 | +0.410 |
| 3 | GPT-5.4 | 0.545 | +0.842 |
| 4 | Gemma-4 31B | 0.491 | +0.304 |
| 5 | Claude Opus 4.1 | 0.396 | -0.455 |
| 6 | GPT-4.1-mini | 0.371 | -0.362 |
| 7 | Gemini Flash Lite | 0.367 | -0.454 |
| 8 | Gemini 3.5 Flash | 0.309 | -1.025 |
| 9 | Mistral Large 3 | 0.291 | -0.976 |
| 10 | LLaMA-3.3 70B | 0.120 | -2.148 |

GPT-5.4 leads in IRT ability but ranks 3rd in accuracy -- IRT binarizes at score > 0.5, revealing different ranking under dichotomous scoring.

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
