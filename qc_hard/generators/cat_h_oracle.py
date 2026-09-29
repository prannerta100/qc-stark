import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    DifficultyLevel.TEXTBOOK: {"n_input": 2, "max_ancillae": None},
    DifficultyLevel.HOMEWORK: {"n_input": 3, "max_ancillae": None},
    DifficultyLevel.EXAM: {"n_input": 3, "max_ancillae": 2},
    DifficultyLevel.RESEARCH: {"n_input": 4, "max_ancillae": 2},
    DifficultyLevel.OPEN: {"n_input": 4, "max_ancillae": 1},
}


class OracleGenerator(BaseGenerator):
    category = Category.H_ORACLE
    subtask = "T6_oracle"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_input = params["n_input"]
        max_ancillae = params["max_ancillae"]

        # Generate random boolean function as truth table
        n_outputs = 2 ** n_input
        truth_table = [int(x) for x in rng.integers(0, 2, size=n_outputs).tolist()]

        # Build truth table display
        tt_lines = []
        for i in range(n_outputs):
            input_bits = f"{i:0{n_input}b}"
            tt_lines.append(f"  f({input_bits}) = {truth_table[i]}")
        tt_str = "\n".join(tt_lines)

        # Determine total qubits
        if max_ancillae is None:
            ancilla_str = "unlimited"
            # Suggest a reasonable number for the circuit size
            suggested_total = n_input + n_input + 1  # input + ancillae + output
        else:
            ancilla_str = str(max_ancillae)
            suggested_total = n_input + max_ancillae + 1

        prompt = self._build_prompt(tt_str, n_input, max_ancillae, ancilla_str)

        return TaskInstance(
            task_id=f"T6_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "n_input": n_input,
                "truth_table": truth_table,
                "max_ancillae": max_ancillae,
            },
        )

    def _build_prompt(self, tt_str: str, n_input: int, max_ancillae, ancilla_str: str) -> str:
        ancilla_constraint = ""
        if max_ancillae is not None:
            ancilla_constraint = (
                f"\nYou may use at most {max_ancillae} ancilla qubit(s). "
                f"All ancillae must be returned to |0> (clean computation).\n"
            )
        else:
            ancilla_constraint = (
                "\nYou may use as many ancilla qubits as needed. "
                "All ancillae must be returned to |0> (clean computation).\n"
            )

        total_qubits_hint = f"n_input={n_input}, plus ancillae, plus 1 output qubit"

        qubit_labels = ", ".join(f"q[{i}]=x{i}" for i in range(n_input))
        return (
            f"Implement a quantum oracle U_f for the following boolean function "
            f"f: {{0,1}}^{n_input} -> {{0,1}}:\n\n"
            f"Truth table:\n{tt_str}\n\n"
            f"The oracle should act as U_f|x>|y> = |x>|y XOR f(x)>.\n"
            f"{ancilla_constraint}\n"
            f"Write a function `solve()` that returns a QuantumCircuit where:\n"
            f"  - Qubits 0 to {n_input - 1} are the input register ({qubit_labels})\n"
            f"  - The last qubit (highest index) is the output (target) qubit\n"
            f"  - Any qubits in between are ancillae\n"
            f"  - IMPORTANT: In the truth table, the input string is read as "
            f"q[{n_input-1}]...q[1]q[0] (q[0] is the LEAST significant bit).\n"
            f"    Example: f(01) means q[1]=0, q[0]=1.\n\n"
            f"Use only cx, ccx, and x gates.\n"
        )
