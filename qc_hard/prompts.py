"""Canonical system prompts for the QC-Stark benchmark.

SINGLE SOURCE OF TRUTH. Do not redefine any prompt string elsewhere in the
repo. Two prompts are live:

  SYSTEM_PROMPT   canonical. Produced results/full_benchmark_final.jsonl (the
                  dataset behind every headline number) and
                  results/paradox_effort_sweep.jsonl. Any run reported as a
                  main result MUST use it verbatim.

  MINIMAL_PROMPT  reduced-guidance contrast condition, used ONLY for the
                  prompt-sensitivity analysis
                  (results/sensitivity_minimal_final.jsonl). Never a
                  main-results condition.

HISTORICAL_PROMPTS below records prompt strings that produced already-published
data but are not used by any live script. They are kept for provenance and
reproducibility auditing, NOT for reuse. Do not import them into new runs.
Every entry differs from SYSTEM_PROMPT only in the ways noted.
"""

SYSTEM_PROMPT = 'You are a quantum computing expert writing Python code.\n\nEnvironment: Python 3.13, Qiskit 2.x, numpy, scipy.\nKey Qiskit 2.x notes:\n- QuantumCircuit.qasm() removed → use qiskit.qasm2.dumps(circuit) / loads(qasm_str)\n- execute() removed → use Statevector or StatevectorSimulator\n- qiskit.opflow removed → use qiskit.quantum_info (Operator, Statevector)\n\nOutput requirements:\n- Respond with ONLY executable Python code. No markdown fences, no explanations.\n- Define a function called `solve()` that returns the answer.\n'

MINIMAL_PROMPT = 'You are a quantum computing expert. Respond with ONLY executable Python code using Qiskit (version 2.x). No explanations, no markdown fences, just the raw code. The code must define a function called `solve()` that returns the answer. Do NOT use deprecated APIs like .qasm() — use qiskit.qasm2.dumps() if needed.'


# ---------------------------------------------------------------------------
# Provenance only. Not for reuse.
# ---------------------------------------------------------------------------
HISTORICAL_PROMPTS = {
    # Same as SYSTEM_PROMPT except ASCII "->" instead of U+2192 in three
    # bullets. Glyph-only difference. Used for the 550 gemma-4-31b /
    # gemini-flash-lite rows of full_benchmark_final.jsonl that were re-run on
    # an isolated VM; the other 2,200 rows used SYSTEM_PROMPT.
    "full_benchmark_final.jsonl:gemma_vm_rows": 'You are a quantum computing expert writing Python code.\n\nEnvironment: Python 3.13, Qiskit 2.x, numpy, scipy.\nKey Qiskit 2.x notes:\n- QuantumCircuit.qasm() removed -> use qiskit.qasm2.dumps(circuit) / loads(qasm_str)\n- execute() removed -> use Statevector or StatevectorSimulator\n- qiskit.opflow removed -> use qiskit.quantum_info (Operator, Statevector)\n\nOutput requirements:\n- Respond with ONLY executable Python code. No markdown fences, no explanations.\n- Define a function called `solve()` that returns the answer.\n',

    # SYSTEM_PROMPT with solve() asked to return a QuantumCircuit rather than a
    # generic answer, plus an "Available imports" line. B3 noise-discrimination
    # tasks require a circuit.
    "b3_pilot.jsonl": "You are a quantum computing expert writing Python code.\n\nEnvironment: Python 3.13, Qiskit 2.x, numpy, scipy.\nKey Qiskit 2.x notes:\n- QuantumCircuit.qasm() removed → use qiskit.qasm2.dumps(circuit) / loads(qasm_str)\n- execute() removed → use Statevector or StatevectorSimulator\n- qiskit.opflow removed → use qiskit.quantum_info (Operator, Statevector)\n\nOutput requirements:\n- Respond with ONLY executable Python code. No markdown fences, no explanations.\n- Define a function called `solve()` that returns a QuantumCircuit.\n- Available imports: qiskit, numpy, scipy. Do not use packages that aren't installed.\n",

    # As above but WITHOUT the "Available imports" line. Note this differs from
    # the b3_pilot.jsonl prompt: the two published B3 datasets were not
    # generated under identical prompts.
    "b3_effort_replication.jsonl": 'You are a quantum computing expert writing Python code.\n\nEnvironment: Python 3.13, Qiskit 2.x, numpy, scipy.\nKey Qiskit 2.x notes:\n- QuantumCircuit.qasm() removed → use qiskit.qasm2.dumps(circuit) / loads(qasm_str)\n- execute() removed → use Statevector or StatevectorSimulator\n- qiskit.opflow removed → use qiskit.quantum_info (Operator, Statevector)\n\nOutput requirements:\n- Respond with ONLY executable Python code. No markdown fences, no explanations.\n- Define a function called `solve()` that returns a QuantumCircuit.\n',
}
