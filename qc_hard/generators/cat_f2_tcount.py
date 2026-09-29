"""F2: T-gate count minimization in Clifford+T circuits."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

CLIFFORD_T_BASIS = ["h", "cx", "s", "sdg", "t", "tdg", "x", "y", "z"]
CLIFFORD_BASIS   = ["h", "cx", "s", "sdg", "x", "y", "z"]

LEVEL_CONFIG = {
    # Redundancy type determines difficulty; circuit size is secondary.
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "n_base_gates": 6,  "pattern": "adjacent"},
    DifficultyLevel.HOMEWORK: {"n_qubits": 2, "n_base_gates": 8,  "pattern": "fourfold"},
    DifficultyLevel.EXAM:     {"n_qubits": 3, "n_base_gates": 12, "pattern": "mixed"},
    DifficultyLevel.RESEARCH: {"n_qubits": 3, "n_base_gates": 16, "pattern": "separated"},
    DifficultyLevel.OPEN:     {"n_qubits": 4, "n_base_gates": 20, "pattern": "separated"},
}


class TCountGenerator(BaseGenerator):
    category = Category.F_EQUIVALENCE
    subtask = "F2_tcount"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level.value * 10000)
        cfg = LEVEL_CONFIG[level]
        n_qubits = cfg["n_qubits"]

        base = self._random_clifford_t(rng, n_qubits, cfg["n_base_gates"])
        optimal_t = self._count_t(base)
        target_unitary = Operator(base).data

        inflated = self._inflate(rng, base, cfg["pattern"], n_qubits)
        inflated_t = self._count_t(inflated)
        inflated_qasm = qasm_dumps(inflated)

        prompt = self._build_prompt(inflated_qasm, inflated_t, optimal_t, n_qubits)

        return TaskInstance(
            task_id=f"F2_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "inflated_qasm": inflated_qasm,
                "inflated_t_count": inflated_t,
                "optimal_t_count": optimal_t,
                "n_qubits": n_qubits,
                "pattern": cfg["pattern"],
                "unitary_real": [[float(v.real) for v in row] for row in target_unitary],
                "unitary_imag": [[float(v.imag) for v in row] for row in target_unitary],
            },
        )

    def _random_clifford_t(self, rng, n_qubits: int, n_gates: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        clifford_1q = ["h", "s", "sdg", "x", "y", "z"]
        n_t = max(2, n_gates // 4)
        # Build a base of Clifford gates, scatter a few T/Tdg gates
        for _ in range(n_gates - n_t):
            q = int(rng.integers(0, n_qubits))
            if n_qubits > 1 and rng.random() < 0.25:
                q2 = (q + 1) % n_qubits
                qc.cx(q, q2)
            else:
                getattr(qc, str(rng.choice(clifford_1q)))(q)
        # Add the core T gates
        for _ in range(n_t):
            q = int(rng.integers(0, n_qubits))
            qc.t(q) if rng.random() < 0.5 else qc.tdg(q)
        return qc

    def _count_t(self, qc: QuantumCircuit) -> int:
        return sum(1 for inst in qc.data if inst.operation.name in ("t", "tdg"))

    def _inflate(self, rng, qc: QuantumCircuit, pattern: str, n_qubits: int) -> QuantumCircuit:
        data = list(qc.data)

        def insert(pos, gates):
            nonlocal data
            data = data[:pos] + gates + data[pos:]

        def make_adjacent_pair(q):
            # T · T† = I — insert both adjacent
            tmp = QuantumCircuit(n_qubits)
            tmp.t(q)
            tmp.tdg(q)
            return list(tmp.data)

        def make_fourfold(q):
            # T⁴ = I
            tmp = QuantumCircuit(n_qubits)
            for _ in range(4):
                tmp.t(q)
            return list(tmp.data)

        def make_separated(q):
            # T · H · H · T† = T · T† = I (since H²=I)
            tmp = QuantumCircuit(n_qubits)
            tmp.t(q)
            tmp.h(q)
            tmp.h(q)
            tmp.tdg(q)
            return list(tmp.data)

        if pattern == "adjacent":
            for _ in range(2):
                q = int(rng.integers(0, n_qubits))
                pos = int(rng.integers(0, max(1, len(data) + 1)))
                insert(pos, make_adjacent_pair(q))

        elif pattern == "fourfold":
            q = int(rng.integers(0, n_qubits))
            pos = int(rng.integers(0, max(1, len(data) + 1)))
            insert(pos, make_fourfold(q))

        elif pattern == "mixed":
            q = int(rng.integers(0, n_qubits))
            pos = int(rng.integers(0, max(1, len(data) + 1)))
            insert(pos, make_adjacent_pair(q))
            pos2 = int(rng.integers(0, max(1, len(data) + 1)))
            q2 = int(rng.integers(0, n_qubits))
            insert(pos2, make_fourfold(q2))

        elif pattern == "separated":
            for _ in range(2):
                q = int(rng.integers(0, n_qubits))
                pos = int(rng.integers(0, max(1, len(data) + 1)))
                insert(pos, make_separated(q))

        new_qc = QuantumCircuit(n_qubits)
        for inst in data:
            new_qc.append(inst.operation, [q._index for q in inst.qubits])
        return new_qc

    def _build_prompt(self, qasm: str, inflated_t: int, optimal_t: int, n_qubits: int) -> str:
        return (
            f"Minimize the T-gate count of the following {n_qubits}-qubit Clifford+T circuit.\n"
            f"The circuit currently uses {inflated_t} T/T† gates. Reduce this while preserving "
            f"the unitary exactly.\n\n"
            f"Useful identities:\n"
            f"  T · T† = I\n"
            f"  T⁴ = I  (so T³ = T†, T² = S)\n"
            f"  H · T† · H · T · H = Rx(π/4) (up to phase)\n\n"
            f"Circuit (OpenQASM 2.0):\n```\n{qasm}\n```\n\n"
            f"Write a function `solve()` that returns an optimized QuantumCircuit "
            f"with fewer T/T† gates (target: ≤ {optimal_t}) implementing the same unitary.\n"
            f"Do NOT use qiskit.transpile() or any PassManager.\n"
        )
