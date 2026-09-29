"""
Prepare and upload QC-Stark benchmark dataset to HuggingFace.
275 unique instances: 11 tasks × 5 levels × 5 seeds.
Usage: python3 prep_hf_dataset.py --token hf_xxxx [--repo-owner username]
"""
import json
import argparse
import tempfile
import os
from pathlib import Path

# Bypass corporate proxy SSL interception (Cisco CA not in Python's bundle).
# Must patch HTTPAdapter.send before huggingface_hub is imported so its session
# inherits verify=False — env-var approaches don't reach urllib3's SSL stack.
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from requests.adapters import HTTPAdapter as _HTTPAdapter
_orig_send = _HTTPAdapter.send
def _unverified_send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    return _orig_send(self, request, stream=stream, timeout=timeout, verify=False, cert=cert, proxies=proxies)
_HTTPAdapter.send = _unverified_send  # type: ignore[method-assign]

TASK_META = {
    "A_stateprep":  ("T1",  "State Preparation",    "Construct"),
    "G1_trotter":   ("T2",  "Trotterization",        "Construct"),
    "H1_oracle":    ("T3",  "Oracle Synthesis",      "Construct"),
    "B1_debugging": ("T4",  "Debugging",             "Understand"),
    "B3_noise_logic":("T5", "Noise Discrimination",  "Understand"),
    "J1_reverse":   ("T6",  "Reverse Engineering",   "Understand"),
    "F1_equivalence":("T7", "Equivalence Checking",  "Verify"),
    "C1_routing":   ("T8",  "Hardware Routing",      "Compile"),
    "I1_noise":     ("T9",  "Noise Fidelity",        "Simulate"),
    "E2_vqe":       ("T10", "VQE",                   "Simulate"),
    "D1_qec":       ("T11", "QEC Decoding",          "Error Correction"),
}

DIFFICULTY = {1: "Textbook", 2: "Homework", 3: "Exam", 4: "Research", 5: "Open"}

DATASET_CARD = """---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - quantum-computing
  - benchmarking
  - code-generation
  - qiskit
  - llm-evaluation
size_categories:
  - n<1K
---

# QC-Stark Benchmark

**QC-Stark** is a multi-task benchmark for evaluating large language models on
quantum computing (QC) tasks. It covers the full practitioner workflow—circuit
construction, debugging, compilation, verification, simulation, and error
correction—with systematic difficulty scaling and contamination resistance via
procedural generation.

## Key facts

| Property | Value |
|----------|-------|
| Tasks | 11 |
| Difficulty levels per task | 5 (Textbook → Homework → Exam → Research → Open) |
| Seeds per (task, level) | 5 |
| **Total instances** | **275** |
| Auto-verification | Yes (Qiskit execution) |
| Contamination resistance | Seed-based procedural generation |

## Task list

| Task ID | Name | Workflow Stage | Mean accuracy (10 models) |
|---------|------|----------------|--------------------------|
| T1 | State Preparation | Construct | 0.440 |
| T2 | Trotterization | Construct | 0.252 |
| T3 | Oracle Synthesis | Construct | 0.412 |
| T4 | Debugging | Understand | 0.020 |
| T5 | Noise Discrimination | Understand | 0.272 |
| T6 | Reverse Engineering | Understand | 0.448 |
| T7 | Equivalence Checking | Verify | 0.824 |
| T8 | Hardware Routing | Compile | 0.184 |
| T9 | Noise Fidelity | Simulate | 0.516 |
| T10 | VQE | Simulate | 0.592 |
| T11 | QEC Decoding | Error Correction | 0.440 |

## Schema

Each instance has the following fields:

- `task_id` (string): T1–T11
- `task_code` (string): internal code (e.g. `B1_debugging`)
- `task_name` (string): human-readable task name
- `workflow_stage` (string): QC workflow stage
- `level` (int): difficulty level 1–5
- `difficulty` (string): Textbook / Homework / Exam / Research / Open
- `seed` (int): procedural generation seed 1–5
- `prompt` (string): the full problem statement given to the model

## Evaluation

Responses are verified by executing model-generated Qiskit code against
auto-generated test cases. No human grading. Verification metrics vary
by task (state fidelity, functional equivalence, syndrome correctness, etc.).

## Citation

```bibtex
@misc{gupta2026qcstark,
  title={QC-Stark: A Multi-Task Benchmark Revealing Capability Dissociations
         in LLMs for Quantum Computing Tasks},
  author={Gupta, Pranav},
  year={2026},
  institution={Cisco Collaboration AI}
}
```
"""


def build_instances(jsonl_path: str) -> list[dict]:
    """Extract 275 unique (task, level, seed) instances — last-record-wins."""
    seen = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["task_name"], r["level"], r["seed"])
            seen[key] = r["prompt"]

    instances = []
    for (task_code, level, seed), prompt in sorted(seen.items()):
        task_id, task_name, stage = TASK_META[task_code]
        instances.append({
            "task_id":        task_id,
            "task_code":      task_code,
            "task_name":      task_name,
            "workflow_stage": stage,
            "level":          level,
            "difficulty":     DIFFICULTY[level],
            "seed":           seed,
            "prompt":         prompt,
        })

    instances.sort(key=lambda x: (int(x["task_id"][1:]), x["level"], x["seed"]))
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    parser.add_argument("--repo-owner", default=None,
                        help="HF username (defaults to token owner)")
    parser.add_argument("--repo-name", default="qc-stark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build dataset locally, skip upload")
    args = parser.parse_args()

    from huggingface_hub import HfApi, login

    login(token=args.token, add_to_git_credential=False)
    api = HfApi()

    if args.repo_owner is None:
        user_info = api.whoami()
        args.repo_owner = user_info["name"]
        print(f"Uploading as: {args.repo_owner}")

    repo_id = f"{args.repo_owner}/{args.repo_name}"

    # Build dataset
    src = Path(__file__).parent / "results" / "full_benchmark_final.jsonl"
    instances = build_instances(str(src))
    print(f"Built {len(instances)} instances")
    assert len(instances) == 275, f"Expected 275, got {len(instances)}"

    if args.dry_run:
        out = Path("qc_stark_dataset_preview.jsonl")
        with open(out, "w") as f:
            for inst in instances[:10]:
                f.write(json.dumps(inst) + "\n")
        print(f"Dry run: wrote 10-row preview to {out}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Write JSONL (train split — this is a benchmark, but HF convention)
        data_file = tmp / "qc_stark.jsonl"
        with open(data_file, "w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")

        # Dataset card
        readme = tmp / "README.md"
        readme.write_text(DATASET_CARD)

        # Create repo
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")

        # Upload files
        api.upload_file(
            path_or_fileobj=str(data_file),
            path_in_repo="qc_stark.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add 275-instance benchmark dataset (11 tasks × 5 levels × 5 seeds)",
        )
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    print(f"\nDone! Dataset live at: https://huggingface.co/datasets/{repo_id}")
    print(f"Load with: load_dataset('{repo_id}', data_files='qc_stark.jsonl')")


if __name__ == "__main__":
    main()
