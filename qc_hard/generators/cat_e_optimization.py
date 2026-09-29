import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.quantum_info import Operator
from qiskit.compiler import transpile
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    # Keep n_qubits and base circuit CONSTANT (3q, 10 gates) — vary ONLY redundancy type
    # This isolates the difficulty variable: how hard are the redundancies to spot?
    DifficultyLevel.TEXTBOOK: {"n_qubits": 3, "base_gates": 10, "redundancies": 3},
    DifficultyLevel.HOMEWORK: {"n_qubits": 3, "base_gates": 10, "redundancies": 3},
    DifficultyLevel.EXAM: {"n_qubits": 3, "base_gates": 10, "redundancies": 3},
    DifficultyLevel.RESEARCH: {"n_qubits": 4, "base_gates": 14, "redundancies": 4},
    DifficultyLevel.OPEN: {"n_qubits": 4, "base_gates": 14, "redundancies": 5},
}

SINGLE_QUBIT_GATES = ["h", "x", "y", "z", "s", "t"]
ROTATION_GATES = ["rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx", "cz"]


class OptimizationGenerator(BaseGenerator):
    category = Category.E_OPTIMIZATION
    subtask = "T3_optimization"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_qubits = params["n_qubits"]
        n_gates = params["base_gates"]
        n_redundancies = params["redundancies"]

        # Build a random base circuit
        base_circuit = self._random_circuit(rng, n_qubits, n_gates)

        # Transpile to get an optimized version (this is the "optimal" target)
        # Use only gates that are valid in OpenQASM 2.0 qelib1.inc
        optimized = transpile(base_circuit, basis_gates=["cx", "h", "rz", "x", "ry", "rx"],
                             optimization_level=3, seed_transpiler=int(rng.integers(0, 2**31)))
        original_gate_count = optimized.size()
        original_unitary = Operator(optimized).data

        # Inflate the circuit with redundancies
        inflated = self._inflate_circuit(rng, optimized, n_redundancies, level)
        inflated_gate_count = inflated.size()
        inflated_qasm = qasm_dumps(inflated)

        prompt = self._build_prompt(inflated_qasm, inflated_gate_count, n_qubits)

        return TaskInstance(
            task_id=f"T3_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "inflated_qasm": inflated_qasm,
                "inflated_gate_count": inflated_gate_count,
                "original_gate_count": original_gate_count,
                "n_qubits": n_qubits,
                "original_unitary_real": [[float(c.real) for c in row] for row in original_unitary],
                "original_unitary_imag": [[float(c.imag) for c in row] for row in original_unitary],
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

    def _inflate_circuit(self, rng, circuit: QuantumCircuit, n_redundancies: int,
                         level: DifficultyLevel) -> QuantumCircuit:
        """Insert redundant gate patterns at random positions within the circuit."""
        n_qubits = circuit.num_qubits
        data = list(circuit.data)

        for _ in range(n_redundancies):
            method = self._pick_inflation_method(rng, level)
            # Build redundancy gates in a temp circuit
            temp = QuantumCircuit(n_qubits)
            temp = method(rng, temp, n_qubits)
            redundancy_gates = list(temp.data)

            # Insert at a random position within the existing gate list
            insert_pos = int(rng.integers(0, max(1, len(data) + 1)))
            data = data[:insert_pos] + redundancy_gates + data[insert_pos:]

        # Reconstruct circuit
        qc = QuantumCircuit(n_qubits)
        for inst in data:
            qc.append(inst.operation, [q._index for q in inst.qubits])
        return qc

    def _pick_inflation_method(self, rng, level: DifficultyLevel):
        if level == DifficultyLevel.TEXTBOOK:
            return self._insert_self_inverse_pair
        elif level == DifficultyLevel.HOMEWORK:
            return self._split_rotation
        elif level == DifficultyLevel.EXAM:
            return self._insert_3gate_sequence
        elif level == DifficultyLevel.RESEARCH:
            return self._insert_identity_subcircuit
        else:  # OPEN
            methods = [self._insert_self_inverse_pair, self._split_rotation,
                       self._insert_identity_subcircuit, self._insert_3gate_sequence]
            return methods[int(rng.integers(0, len(methods)))]

    def _insert_self_inverse_pair(self, rng, qc: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
        """Insert HH, XX, or CX*CX pairs."""
        new_qc = qc.copy()
        qubit = int(rng.integers(0, n_qubits))
        choice = int(rng.integers(0, 3))
        if choice == 0:
            new_qc.h(qubit)
            new_qc.h(qubit)
        elif choice == 1:
            new_qc.x(qubit)
            new_qc.x(qubit)
        else:
            if n_qubits > 1:
                qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
                new_qc.cx(int(qubits[0]), int(qubits[1]))
                new_qc.cx(int(qubits[0]), int(qubits[1]))
            else:
                new_qc.h(qubit)
                new_qc.h(qubit)
        return new_qc

    def _split_rotation(self, rng, qc: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
        """Insert Rz(a/2)*Rz(a/2) which equals Rz(a) for a random a."""
        new_qc = qc.copy()
        qubit = int(rng.integers(0, n_qubits))
        angle = float(rng.uniform(0.1, 2 * np.pi))
        # Insert Rz(angle/2) twice (net: Rz(angle)) then cancel with Rz(-angle)
        # Actually, just add rz(a/2)*rz(a/2)*rz(-a) = identity
        new_qc.rz(angle / 2, qubit)
        new_qc.rz(angle / 2, qubit)
        new_qc.rz(-angle, qubit)
        return new_qc

    def _insert_identity_subcircuit(self, rng, qc: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
        """Insert H*X*H*Z = identity on a qubit."""
        new_qc = qc.copy()
        qubit = int(rng.integers(0, n_qubits))
        # H X H Z = Z Z = I  (since HXH = Z, then Z*Z = I)
        new_qc.h(qubit)
        new_qc.x(qubit)
        new_qc.h(qubit)
        new_qc.z(qubit)
        return new_qc

    def _insert_3gate_sequence(self, rng, qc: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
        """Insert S*S*Z = I (since S*S = Z and Z*Z = I)."""
        new_qc = qc.copy()
        qubit = int(rng.integers(0, n_qubits))
        new_qc.s(qubit)
        new_qc.s(qubit)
        new_qc.z(qubit)
        return new_qc

    def _build_prompt(self, inflated_qasm: str, inflated_gate_count: int, n_qubits: int) -> str:
        return (
            f"Optimize the following {n_qubits}-qubit quantum circuit. Reduce the gate count "
            f"(currently {inflated_gate_count} gates) while preserving the unitary.\n\n"
            f"Circuit (OpenQASM 2.0):\n```\n{inflated_qasm}\n```\n\n"
            f"Write a function `solve()` that returns an optimized QuantumCircuit implementing "
            f"the same unitary with fewer gates. Do NOT use qiskit.transpile().\n"
        )
