import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.compiler import transpile
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 4, "n_gates": 10},
    DifficultyLevel.HOMEWORK: {"n_qubits": 6, "n_gates": 18},
    DifficultyLevel.EXAM: {"n_qubits": 10, "n_gates": 30},
    DifficultyLevel.RESEARCH: {"n_qubits": 12, "n_gates": 40},
    DifficultyLevel.OPEN: {"n_qubits": 14, "n_gates": 55},
}

SINGLE_QUBIT_GATES = ["h", "x", "y", "z", "s", "t"]
ROTATION_GATES = ["rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx", "cz"]


class EquivalenceGenerator(BaseGenerator):
    category = Category.F_EQUIVALENCE
    subtask = "T4_equivalence"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_qubits = params["n_qubits"]
        n_gates = params["n_gates"]

        # Build base circuit
        circuit_a = self._random_circuit(rng, n_qubits, n_gates)
        qasm_a = qasm_dumps(circuit_a)

        # Determine if equivalent or not (50/50 based on seed parity)
        is_equivalent = (seed % 2 == 0)

        use_hard = level in (DifficultyLevel.RESEARCH, DifficultyLevel.OPEN)
        if is_equivalent:
            if use_hard:
                circuit_b = self._make_equivalent_variant_hard(rng, circuit_a, n_qubits)
            else:
                circuit_b = self._make_equivalent_variant(rng, circuit_a, n_qubits)
        else:
            if use_hard:
                circuit_b = self._make_non_equivalent_variant_hard(rng, circuit_a, n_qubits)
            else:
                circuit_b = self._make_non_equivalent_variant(rng, circuit_a, n_qubits)

        qasm_b = qasm_dumps(circuit_b)

        # Ground truth is known by construction:
        # - equivalent variants use only identity-preserving rewrites
        # - non-equivalent variants inject a guaranteed mutation
        ground_truth = is_equivalent

        prompt = self._build_prompt(qasm_a, qasm_b, n_qubits)

        return TaskInstance(
            task_id=f"T4_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "circuit_a_qasm": qasm_a,
                "circuit_b_qasm": qasm_b,
                "ground_truth": ground_truth,
                "n_qubits": n_qubits,
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

    def _make_equivalent_variant(self, rng, circuit: QuantumCircuit,
                                  n_qubits: int) -> QuantumCircuit:
        """Create an equivalent circuit via identity-preserving rewrites."""
        qc = circuit.copy()

        # Apply multiple rewrites
        n_rewrites = int(rng.integers(1, 4))
        for _ in range(n_rewrites):
            method = int(rng.integers(0, 3))
            if method == 0:
                # Insert and cancel: add HH pair
                qubit = int(rng.integers(0, n_qubits))
                qc.h(qubit)
                qc.h(qubit)
            elif method == 1:
                # Insert XX pair
                qubit = int(rng.integers(0, n_qubits))
                qc.x(qubit)
                qc.x(qubit)
            else:
                # Insert S S Z = I
                qubit = int(rng.integers(0, n_qubits))
                qc.s(qubit)
                qc.s(qubit)
                qc.z(qubit)

        return qc

    def _make_equivalent_variant_hard(self, rng, circuit: QuantumCircuit,
                                       n_qubits: int) -> QuantumCircuit:
        """Create structurally different but equivalent circuit via re-synthesis."""
        seed_val = int(rng.integers(0, 2**31))
        resynthesized = transpile(
            circuit,
            basis_gates=["cx", "h", "rz", "x", "ry", "rx"],
            optimization_level=3,
            seed_transpiler=seed_val,
        )
        # Interleave identity insertions at random positions within the circuit
        data = list(resynthesized.data)
        n_insertions = int(rng.integers(2, 5))
        for _ in range(n_insertions):
            pos = int(rng.integers(0, max(1, len(data) + 1)))
            qubit = int(rng.integers(0, n_qubits))
            temp = QuantumCircuit(n_qubits)
            method = int(rng.integers(0, 3))
            if method == 0:
                temp.h(qubit)
                temp.h(qubit)
            elif method == 1:
                temp.x(qubit)
                temp.x(qubit)
            else:
                temp.s(qubit)
                temp.s(qubit)
                temp.z(qubit)
            data = data[:pos] + list(temp.data) + data[pos:]

        new_qc = QuantumCircuit(n_qubits)
        for inst in data:
            new_qc.append(inst.operation, [q._index for q in inst.qubits])
        return new_qc

    def _make_non_equivalent_variant(self, rng, circuit: QuantumCircuit,
                                      n_qubits: int) -> QuantumCircuit:
        """Create a non-equivalent circuit via subtle mutation."""
        qc = circuit.copy()
        data = list(qc.data)

        if not data:
            new_qc = QuantumCircuit(n_qubits)
            new_qc.x(0)
            return new_qc

        # Pick a random instruction to mutate
        idx = int(rng.integers(0, len(data)))
        inst = data[idx]

        new_qc = QuantumCircuit(n_qubits)
        for i, d in enumerate(data):
            if i == idx:
                if inst.operation.name in ROTATION_GATES and inst.operation.params:
                    new_angle = inst.operation.params[0] + float(rng.uniform(0.3, 1.0))
                    qubit_idx = d.qubits[0]._index
                    getattr(new_qc, inst.operation.name)(new_angle, qubit_idx)
                elif inst.operation.name in SINGLE_QUBIT_GATES:
                    candidates = [g for g in SINGLE_QUBIT_GATES if g != inst.operation.name]
                    new_gate = str(rng.choice(candidates))
                    qubit_idx = d.qubits[0]._index
                    getattr(new_qc, new_gate)(qubit_idx)
                else:
                    if inst.operation.name == "cx":
                        new_qc.cx(d.qubits[1]._index, d.qubits[0]._index)
                    elif inst.operation.name == "cz":
                        new_qc.append(d)
                        new_qc.rz(float(rng.uniform(0.3, 1.0)), d.qubits[0]._index)
                    else:
                        new_qc.append(d)
            else:
                new_qc.append(d)

        return new_qc

    def _make_non_equivalent_variant_hard(self, rng, circuit: QuantumCircuit,
                                           n_qubits: int) -> QuantumCircuit:
        """Create a structurally different non-equivalent circuit.

        Re-synthesize first (like the equivalent variant), then apply a subtle
        rotation perturbation so the circuits look completely different AND
        aren't equivalent.
        """
        seed_val = int(rng.integers(0, 2**31))
        resynthesized = transpile(
            circuit,
            basis_gates=["cx", "h", "rz", "x", "ry", "rx"],
            optimization_level=3,
            seed_transpiler=seed_val,
        )
        # Now mutate: pick a rotation gate and shift its angle
        data = list(resynthesized.data)
        rotation_indices = [i for i, d in enumerate(data)
                           if d.operation.name in ROTATION_GATES and d.operation.params]
        new_qc = QuantumCircuit(n_qubits)
        if rotation_indices:
            # Mutate multiple rotation gates to ensure fidelity breaks clearly
            n_mutations = min(3, len(rotation_indices))
            mutate_set = set(rng.choice(rotation_indices, size=n_mutations, replace=False).tolist())
            for i, d in enumerate(data):
                if i in mutate_set:
                    new_angle = d.operation.params[0] + float(rng.uniform(0.3, 1.0))
                    qubit_idx = d.qubits[0]._index
                    getattr(new_qc, d.operation.name)(new_angle, qubit_idx)
                else:
                    new_qc.append(d.operation, [q._index for q in d.qubits])
        else:
            for d in data:
                new_qc.append(d.operation, [q._index for q in d.qubits])
            qubit = int(rng.integers(0, n_qubits))
            new_qc.rz(float(rng.uniform(0.5, 1.5)), qubit)
        return new_qc

    def _build_prompt(self, qasm_a: str, qasm_b: str, n_qubits: int) -> str:
        return (
            f"Are these two {n_qubits}-qubit circuits equivalent "
            f"(do they implement the same unitary up to global phase)?\n\n"
            f"Circuit A (OpenQASM 2.0):\n```\n{qasm_a}\n```\n\n"
            f"Circuit B (OpenQASM 2.0):\n```\n{qasm_b}\n```\n\n"
            f"Determine equivalence. You may use any approach: gate identity reasoning, "
            f"ZX-calculus, simulation, or any other method.\n\n"
            f"Write a function `solve()` that returns True if the circuits are equivalent "
            f"(same unitary up to global phase), or False if they are not.\n"
        )
