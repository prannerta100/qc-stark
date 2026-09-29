"""QC-Stark benchmark runner — the single data-generation script.

Runs all 11 tasks x 10 models x 5 levels x 5 seeds under one prompt condition.
Stores the full prompt and raw model response for every record, and is
resumable: (task, model, level, seed) already present in the output file are
skipped.

    python run_benchmark.py                  # structured (canonical, main results)
    python run_benchmark.py --condition minimal

structured -> results/full_benchmark_final.jsonl      (SYSTEM_PROMPT)
minimal    -> results/sensitivity_minimal_final.jsonl (MINIMAL_PROMPT)

Token budgets per model match the configuration that produced the published
results; see qc_hard/prompts.py for the authoritative prompt strings.
"""
import argparse

import re
import os, subprocess, json, time, threading, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

from qc_hard.types import DifficultyLevel, ModelResponse

from qc_hard.generators.cat_a_synthesis       import StatePrepGenerator
from qc_hard.verifiers.cat_a_synthesis        import StatePrepVerifier
from qc_hard.generators.cat_b_debugging       import SingleBugGenerator
from qc_hard.verifiers.cat_b_debugging        import SingleBugVerifier
from qc_hard.generators.cat_b3_noise_logic    import NoiseBugDiscriminationGenerator
from qc_hard.verifiers.cat_b3_noise_logic     import NoiseBugDiscriminationVerifier
from qc_hard.generators.cat_c_compilation     import RoutingGenerator
from qc_hard.verifiers.cat_c_compilation      import RoutingVerifier
from qc_hard.generators.cat_d_qec            import SyndromeDecodingGenerator
from qc_hard.verifiers.cat_d_qec             import SyndromeDecodingVerifier
from qc_hard.generators.cat_e2_vqe           import VQEGenerator
from qc_hard.verifiers.cat_e2_vqe            import VQEVerifier
from qc_hard.generators.cat_f_equivalence    import EquivalenceGenerator
from qc_hard.verifiers.cat_f_equivalence     import EquivalenceVerifier
from qc_hard.generators.cat_g_trotter        import TrotterGenerator
from qc_hard.verifiers.cat_g_trotter         import TrotterVerifier
from qc_hard.generators.cat_h_oracle         import OracleGenerator
from qc_hard.verifiers.cat_h_oracle          import OracleVerifier
from qc_hard.generators.cat_i_noise          import NoiseFidelityGenerator
from qc_hard.verifiers.cat_i_noise           import NoiseFidelityVerifier
from qc_hard.generators.cat_j_reverse        import ReverseEngineeringGenerator
from qc_hard.verifiers.cat_j_reverse         import ReverseEngineeringVerifier

TASKS = {
    "A_stateprep":    (StatePrepGenerator,              StatePrepVerifier),
    "B1_debugging":   (SingleBugGenerator,              SingleBugVerifier),
    "B3_noise_logic": (NoiseBugDiscriminationGenerator, NoiseBugDiscriminationVerifier),
    "C1_routing":     (RoutingGenerator,                RoutingVerifier),
    "D1_qec":         (SyndromeDecodingGenerator,       SyndromeDecodingVerifier),
    "E2_vqe":         (VQEGenerator,                    VQEVerifier),
    "F1_equivalence": (EquivalenceGenerator,            EquivalenceVerifier),
    "G1_trotter":     (TrotterGenerator,                TrotterVerifier),
    "H1_oracle":      (OracleGenerator,                 OracleVerifier),
    "I1_noise":       (NoiseFidelityGenerator,          NoiseFidelityVerifier),
    "J1_reverse":     (ReverseEngineeringGenerator,     ReverseEngineeringVerifier),
}

MODELS = {
    # budget = max_completion_tokens; matches the configuration that produced
    # results/full_benchmark_final.jsonl (see archive_code/rerun_budget_fix.py).
    "o4-mini":           {"id": "openai/o4-mini-2025-04-16-global",     "type": "o-series", "budget": 65536},
    "gpt-5.4":           {"id": "openai/gpt-5.4-2026-03-05",           "type": "o-series", "budget": 65536},
    "gpt-4.1-mini":      {"id": "openai/gpt-4.1-mini-2025-04-14",      "type": "standard", "budget": 32768},
    "claude-sonnet-5":   {"id": "anthropic/claude-sonnet-5",            "type": "claude",   "budget": 65536},
    "claude-opus":       {"id": "anthropic/claude-opus-4.1",            "type": "claude",   "budget": 32000},
    "gemini-3.5-flash":  {"id": "google/gemini-3.5-flash-global",       "type": "standard", "budget": 65536},
    "gemini-flash-lite": {"id": "google/gemini-3.1-flash-lite-global",  "type": "standard", "budget": 4096},
    "gemma-4-31b":       {"id": "google/gemma-4-31b-it",                "type": "standard", "budget": 4096},
    "mistral-large-3":   {"id": "mistral/mistral-large-3",              "type": "standard", "budget": 65536},
    "llama-70b":         {"id": "meta/llama-3.3-70b",                   "type": "standard", "budget": 8192},
}

from qc_hard.prompts import SYSTEM_PROMPT, MINIMAL_PROMPT

N_SEEDS            = 5
LEVELS             = [DifficultyLevel.TEXTBOOK, DifficultyLevel.HOMEWORK, DifficultyLevel.EXAM,
                      DifficultyLevel.RESEARCH, DifficultyLevel.OPEN]
VERIFICATION_TIMEOUT = 45
CONDITION = "structured"
CONDITIONS = {
    "structured": ("results/full_benchmark_final.jsonl",      SYSTEM_PROMPT),
    "minimal":    ("results/sensitivity_minimal_final.jsonl", MINIMAL_PROMPT),
}
OUTPUT_FILE = CONDITIONS["structured"][0]
ACTIVE_PROMPT = CONDITIONS["structured"][1]
MAX_WORKERS        = 4

BASE_URL = os.environ.get("QCSTARK_LLM_BASE_URL")
API_KEY  = os.environ.get("QCSTARK_LLM_API_KEY")
if not (BASE_URL and API_KEY):
    raise SystemExit(
        "Set QCSTARK_LLM_BASE_URL and QCSTARK_LLM_API_KEY to an OpenAI-compatible\n"
        "endpoint and credential before running, e.g.\n"
        "  export QCSTARK_LLM_BASE_URL=https://api.openai.com/v1\n"
        "  export QCSTARK_LLM_API_KEY=$(your-credential-command)"
    )
_HEADERS = {}
if os.environ.get("QCSTARK_LLM_APP_HEADER"):
    _HEADERS["x-app"] = os.environ["QCSTARK_LLM_APP_HEADER"]

from openai import OpenAI
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, default_headers=_HEADERS or None)

write_lock = Lock()
counter    = {"done": 0, "correct": 0, "api_err": 0}


_SOLVE_RE = re.compile(r"def\s+solve\s*\(")
_CODE_START_RE = re.compile(r"^(?:import |from |def |class |@)", re.M)
_PY_OPEN_RE = re.compile(r"```[ \t]*(?:python|py)[ \t]*\r?\n")
_ANY_OPEN_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n")
_BLOCK_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n.*?```", re.S)


def _legacy_extract(text):
    for fence in ("```python", "```"):
        if fence in text:
            parts = text.split(fence)
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
    return text.strip()


def _closed_body(text, m):
    rest = text[m.end():]
    if "```" not in rest:
        return None
    return rest.split("```")[0]


def extract_code(text):
    """Return the Python the model actually intended as its answer.

    Prefers a closed ```python block defining solve(), then any closed block
    defining solve(), then unfenced code when solve() lives outside every
    closed block (orphan trailing fence, or an answer written after an inline
    snippet). Falls back to the legacy first-block behaviour.
    """
    for m in _PY_OPEN_RE.finditer(text):
        b = _closed_body(text, m)
        if b and _SOLVE_RE.search(b):
            return b.strip()
    for m in _ANY_OPEN_RE.finditer(text):
        b = _closed_body(text, m)
        if b and _SOLVE_RE.search(b):
            return b.strip()
    if _SOLVE_RE.search(text):
        rest = _BLOCK_RE.sub("", text)
        rest = re.sub(r"^[ \t]*```.*$", "", rest, flags=re.M)
        m = _CODE_START_RE.search(rest)
        if m:
            cand = rest[m.start():].strip()
            if _SOLVE_RE.search(cand):
                return cand
        return rest.strip()
    return _legacy_extract(text)



def safe_details(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def run_single(task_name, gen_cls, ver_cls, model_name, model_cfg, level, seed):
    try:
        task = gen_cls().generate(seed=seed, level=level)
        prompt_text = task.prompt

        kwargs = {
            "model": model_cfg["id"],
            "messages": [
                {"role": "system", "content": ACTIVE_PROMPT},
                {"role": "user",   "content": prompt_text},
            ],
        }
        kwargs["max_completion_tokens"] = model_cfg["budget"]
        if model_cfg["type"] == "standard":
            kwargs["temperature"] = 0.0

        t0 = time.time()
        resp_api = client.chat.completions.create(**kwargs)
        latency  = (time.time() - t0) * 1000
        content  = resp_api.choices[0].message.content or ""
        tokens   = resp_api.usage.total_tokens if resp_api.usage else 0

        resp = ModelResponse(
            task_id=task.task_id, model_name=model_name,
            raw_response=content, parsed_code=extract_code(content),
            latency_ms=latency, token_count=tokens,
        )

        result_holder, error_holder = [None], [None]
        def _verify():
            try:    result_holder[0] = ver_cls().verify(task, resp)
            except Exception as e: error_holder[0] = e

        t = threading.Thread(target=_verify)
        t.start(); t.join(timeout=VERIFICATION_TIMEOUT)

        if t.is_alive():
            return _rec(task_name, model_name, level, seed, prompt_text, content,
                        False, 0.0, {"verifier": "timeout"}, tokens, latency)
        if error_holder[0]:
            # API call succeeded — preserve content even though verifier crashed
            return _rec(task_name, model_name, level, seed, prompt_text, content,
                        False, 0.0, {"verifier_error": str(error_holder[0])[:300]},
                        tokens, latency)

        res = result_holder[0]
        det = safe_details(res.details or {})

        # Extra B3 metadata for paradox analysis
        if task_name == "B3_noise_logic":
            det["has_bug"] = bool(task.metadata.get("has_bug", False))

        return _rec(task_name, model_name, level, seed, prompt_text, content,
                    bool(res.correct), float(res.score), det, tokens, latency)

    except Exception as e:
        return _rec(task_name, model_name, level, seed, "", "",
                    False, 0.0, {"api_error": str(e)[:300]}, 0, 0)


def _rec(task, model, level, seed, prompt, raw, correct, score, details, tokens, latency):
    return {
        "task_name":   task,
        "model":       model,
        "level":       level.value,
        "seed":        seed,
        "correct":     correct,
        "score":       score,
        "details":     details,
        "latency_ms":  latency,
        "token_count": tokens,
        "prompt":      prompt,
        "raw_response": raw,
        "prompt_condition": CONDITION,
    }


def load_completed():
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["task_name"], r["model"], r["level"], r["seed"]))
                except Exception:
                    pass
    return done


def print_summary():
    if not os.path.exists(OUTPUT_FILE):
        return
    data = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            try: data.append(json.loads(line))
            except: pass

    by_task = defaultdict(list)
    for r in data: by_task[r["task_name"]].append(r)

    print(f"\n{'='*70}")
    print(f"{'task':<18} {'n':>5}  {'acc':>6}  {'models':>7}  {'has_prompt':>10}  {'has_raw':>8}")
    print(f"{'='*70}")
    for t in sorted(by_task.keys()):
        rows = by_task[t]
        acc  = sum(r["correct"] for r in rows) / len(rows)
        mods = len(set(r["model"] for r in rows))
        hp   = sum(1 for r in rows if r.get("prompt"))
        hr   = sum(1 for r in rows if r.get("raw_response"))
        print(f"{t:<18} {len(rows):>5}  {acc:>6.3f}  {mods:>7}  {hp:>10}  {hr:>8}")
    print(f"{'='*70}")
    print(f"Total: {len(data)} records")


def main():
    global OUTPUT_FILE, ACTIVE_PROMPT, CONDITION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="structured")
    args = ap.parse_args()
    CONDITION = args.condition
    OUTPUT_FILE, ACTIVE_PROMPT = CONDITIONS[CONDITION]
    print(f"condition: {CONDITION}  ->  {OUTPUT_FILE}")

    os.makedirs("results", exist_ok=True)
    completed = load_completed()

    jobs = [
        (task_name, gen_cls, ver_cls, model_name, model_cfg, level, seed)
        for task_name, (gen_cls, ver_cls) in TASKS.items()
        for level in LEVELS
        for seed in range(1, N_SEEDS + 1)
        for model_name, model_cfg in MODELS.items()
        if (task_name, model_name, level.value, seed) not in completed
    ]

    total_target = len(TASKS) * len(MODELS) * len(LEVELS) * N_SEEDS
    print(f"Full benchmark v2 — all 11 tasks with prompt + raw_response storage")
    print(f"Tasks: {list(TASKS.keys())}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Target: {total_target}  Done: {len(completed)}  Remaining: {len(jobs)}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    if not jobs:
        print_summary()
        return

    global_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, *job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            with write_lock:
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(record) + "\n")
                counter["done"] += 1
                if record["correct"]:       counter["correct"] += 1
                if "api_error" in record.get("details", {}): counter["api_err"] += 1
                if counter["done"] % 50 == 0:
                    elapsed = time.time() - global_start
                    rate    = counter["done"] / elapsed * 60
                    eta     = (len(jobs) - counter["done"]) / (rate / 60) if rate > 0 else 0
                    print(f"  [{counter['done']}/{len(jobs)}]  "
                          f"acc={counter['correct']/counter['done']:.2f}  "
                          f"api_err={counter['api_err']}  "
                          f"rate={rate:.0f}/min  ETA={eta/60:.1f}min")

    print(f"\nDone in {(time.time()-global_start)/60:.1f} min")
    print_summary()


if __name__ == "__main__":
    main()
