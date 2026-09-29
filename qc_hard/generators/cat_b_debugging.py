import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.quantum_info import Operator
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "gates_range": (4, 8)},
    DifficultyLevel.HOMEWORK: {"n_qubits": 3, "gates_range": (10, 20)},
    DifficultyLevel.EXAM: {"n_qubits": 3, "gates_range": (20, 35)},
    DifficultyLevel.RESEARCH: {"n_qubits": 4, "gates_range": (35, 55)},
    DifficultyLevel.OPEN: {"n_qubits": 5, "gates_range": (55, 80)},
}

SINGLE_QUBIT_GATES = ["h", "x", "y", "z", "s", "t"]
ROTATION_GATES = ["rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx", "cz"]


class SingleBugGenerator(BaseGenerator):
    category = Category.B_DEBUGGING
    subtask = "B1_single_bug"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_qubits = params["n_qubits"]
        min_gates, max_gates = params["gates_range"]
        n_gates = int(rng.integers(min_gates, max_gates + 1))

        correct_circuit = self._random_circuit(rng, n_qubits, n_gates)
        correct_qasm = qasm_dumps(correct_circuit)
        op = Operator(correct_circuit)

        buggy_circuit, mutation_info = self._inject_bug(rng, correct_circuit)
        buggy_qasm = qasm_dumps(buggy_circuit)

        u_data = op.data
        prompt = self._build_prompt(buggy_qasm, u_data, n_qubits)

        return TaskInstance(
            task_id=f"B1_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "buggy_circuit_qasm": buggy_qasm,
                "correct_circuit_qasm": correct_qasm,
                "mutation": mutation_info,
                "n_qubits": n_qubits,
                "correct_unitary_real": [[float(c.real) for c in row] for row in u_data],
                "correct_unitary_imag": [[float(c.imag) for c in row] for row in u_data],
            },
        )

    def _random_circuit(self, rng, n_qubits: int, n_gates: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for _ in range(n_gates):
            if n_qubits > 1 and rng.random() < 0.3:
                gate = str(rng.choice(TWO_QUBIT_GATES))
                qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
                if gate == "cx":
                    qc.cx(int(qubits[0]), int(qubits[1]))
                else:
                    qc.cz(int(qubits[0]), int(qubits[1]))
            else:
                use_rotation = rng.random() < 0.3
                if use_rotation:
                    gate = str(rng.choice(ROTATION_GATES))
                    qubit = int(rng.integers(0, n_qubits))
                    angle = float(rng.uniform(0.1, 2 * np.pi))
                    getattr(qc, gate)(angle, qubit)
                else:
                    gate = str(rng.choice(SINGLE_QUBIT_GATES))
                    qubit = int(rng.integers(0, n_qubits))
                    getattr(qc, gate)(qubit)
        return qc

    def _inject_bug(self, rng, circuit: QuantumCircuit):
        qc = circuit.copy()
        data = list(qc.data)
        if not data:
            return qc, {"type": "none"}

        idx = int(rng.integers(0, len(data)))
        inst = data[idx]
        orig_name = inst.operation.name

        # Try gate substitution first
        if inst.operation.num_qubits == 1 and orig_name in SINGLE_QUBIT_GATES:
            candidates = [g for g in SINGLE_QUBIT_GATES if g != orig_name]
            new_gate = str(rng.choice(candidates))
            new_qc = QuantumCircuit(qc.num_qubits)
            for i, d in enumerate(data):
                if i == idx:
                    getattr(new_qc, new_gate)(d.qubits[0]._index)
                else:
                    new_qc.append(d)
            return new_qc, {"type": "gate_sub", "index": idx, "original": orig_name, "replacement": new_gate}

        # Fallback: gate deletion
        new_qc = QuantumCircuit(qc.num_qubits)
        for i, d in enumerate(data):
            if i != idx:
                new_qc.append(d)
        return new_qc, {"type": "gate_delete", "index": idx, "deleted_gate": orig_name}

    def _build_prompt(self, buggy_qasm: str, target_unitary, n_qubits: int) -> str:
        import numpy as np
        dim = 2**n_qubits
        # Show input-output pairs on basis states
        io_lines = []
        for col_idx in range(dim):
            input_label = f"|{col_idx:0{n_qubits}b}>"
            output_parts = []
            for row_idx in range(dim):
                amp = target_unitary[row_idx, col_idx]
                if abs(amp) > 0.01:
                    output_parts.append(f"({amp.real:.4f}{amp.imag:+.4f}i)|{row_idx:0{n_qubits}b}>")
            io_lines.append(f"  {input_label} -> {' + '.join(output_parts)}")

        io_str = "\n".join(io_lines)
        return (
            f"The following {n_qubits}-qubit quantum circuit has exactly ONE bug "
            f"(a wrong gate, swapped qubits, or missing gate).\n\n"
            f"Buggy circuit (OpenQASM 2.0):\n```\n{buggy_qasm}\n```\n\n"
            f"The INTENDED unitary transformation maps basis states as follows:\n"
            f"{io_str}\n\n"
            f"Identify the bug and write a Qiskit function `solve()` that returns the CORRECTED QuantumCircuit.\n"
        )
