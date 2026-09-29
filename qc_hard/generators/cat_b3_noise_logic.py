"""B3: Noise vs. Logic Bug Discrimination.

Given a circuit and two probability distributions (noiseless expected vs noisy observed),
determine whether the deviation is from a logic bug or hardware noise only.
If bug: return the fixed circuit. If noise only: return the original circuit unchanged.

This is the task that directly tests the debugging paradox: reasoning models over-intervene
on noise-only cases, modifying circuits that should not be touched.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.quantum_info import Operator, Statevector
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

# Noise parameters: fixed across levels so the model can reason about them.
# Varied difficulty comes from circuit size and bug subtlety.
P_1Q = 0.02   # depolarizing probability after each 1-qubit gate
P_2Q = 0.05   # depolarizing probability after each 2-qubit gate

LEVEL_CONFIG = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "gates_range": (4,  7),  "bug_type": "gate_sub"},
    DifficultyLevel.HOMEWORK: {"n_qubits": 3, "gates_range": (8,  14), "bug_type": "gate_sub"},
    DifficultyLevel.EXAM:     {"n_qubits": 3, "gates_range": (14, 22), "bug_type": "angle"},
    DifficultyLevel.RESEARCH: {"n_qubits": 4, "gates_range": (20, 30), "bug_type": "angle"},
    DifficultyLevel.OPEN:     {"n_qubits": 4, "gates_range": (28, 40), "bug_type": "angle"},
}

SINGLE_QUBIT_GATES = ["h", "x", "y", "z", "s", "t"]
ROTATION_GATES     = ["rx", "ry", "rz"]
TWO_QUBIT_GATES    = ["cx", "cz"]


class NoiseBugDiscriminationGenerator(BaseGenerator):
    category = Category.B_DEBUGGING
    subtask   = "B3_noise_logic"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level.value * 10000)
        cfg = LEVEL_CONFIG[level]
        n_qubits = cfg["n_qubits"]
        n_gates  = int(rng.integers(*cfg["gates_range"]))

        correct_circuit = self._random_circuit(rng, n_qubits, n_gates)
        correct_unitary = np.array(Operator(correct_circuit).data)

        # 50/50 split: even seeds have a bug, odd seeds are noise-only.
        has_bug = (seed % 2 == 0)

        if has_bug:
            shown_circuit, mutation = self._inject_bug(rng, correct_circuit, cfg["bug_type"])
        else:
            shown_circuit = correct_circuit.copy()
            mutation = {"type": "none"}

        shown_unitary = np.array(Operator(shown_circuit).data)

        # Noiseless expected probabilities: from the CORRECT (intended) circuit
        noiseless_probs = self._noiseless_probs(correct_circuit, n_qubits)

        # Noisy observed probabilities: from the SHOWN circuit (correct or buggy) + noise
        noisy_probs = self._noisy_probs(shown_circuit, n_qubits)

        shown_qasm = qasm_dumps(shown_circuit)
        prompt = self._build_prompt(shown_qasm, noiseless_probs, noisy_probs, n_qubits)

        return TaskInstance(
            task_id=f"B3_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "has_bug":         has_bug,
                "mutation":        mutation,
                "n_qubits":        n_qubits,
                "shown_qasm":      shown_qasm,
                "correct_unitary_real": [[float(v.real) for v in row] for row in correct_unitary],
                "correct_unitary_imag": [[float(v.imag) for v in row] for row in correct_unitary],
                "shown_unitary_real":   [[float(v.real) for v in row] for row in shown_unitary],
                "shown_unitary_imag":   [[float(v.imag) for v in row] for row in shown_unitary],
                "p_1q": P_1Q,
                "p_2q": P_2Q,
            },
        )

    # ------------------------------------------------------------------ helpers

    def _random_circuit(self, rng, n_qubits: int, n_gates: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for _ in range(n_gates):
            if n_qubits > 1 and rng.random() < 0.30:
                gate   = str(rng.choice(TWO_QUBIT_GATES))
                qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
                qc.cx(int(qubits[0]), int(qubits[1])) if gate == "cx" else qc.cz(int(qubits[0]), int(qubits[1]))
            elif rng.random() < 0.30:
                gate   = str(rng.choice(ROTATION_GATES))
                qubit  = int(rng.integers(0, n_qubits))
                angle  = float(rng.uniform(0.2, 2 * np.pi))
                getattr(qc, gate)(angle, qubit)
            else:
                gate  = str(rng.choice(SINGLE_QUBIT_GATES))
                qubit = int(rng.integers(0, n_qubits))
                getattr(qc, gate)(qubit)
        return qc

    def _inject_bug(self, rng, circuit: QuantumCircuit, bug_type: str):
        data = list(circuit.data)
        if not data:
            return circuit.copy(), {"type": "none"}

        if bug_type == "gate_sub":
            # Pick a random single-qubit gate and substitute
            candidates = [i for i, d in enumerate(data) if d.operation.name in SINGLE_QUBIT_GATES]
            if not candidates:
                return circuit.copy(), {"type": "none"}
            idx = candidates[int(rng.integers(0, len(candidates)))]
            orig_name = data[idx].operation.name
            new_gate  = str(rng.choice([g for g in SINGLE_QUBIT_GATES if g != orig_name]))
            new_qc = QuantumCircuit(circuit.num_qubits)
            for i, d in enumerate(data):
                if i == idx:
                    getattr(new_qc, new_gate)(d.qubits[0]._index)
                else:
                    new_qc.append(d)
            return new_qc, {"type": "gate_sub", "index": idx, "original": orig_name, "replacement": new_gate}

        else:  # angle perturbation
            candidates = [i for i, d in enumerate(data) if d.operation.name in ROTATION_GATES and d.operation.params]
            if not candidates:
                return self._inject_bug(rng, circuit, "gate_sub")
            idx   = candidates[int(rng.integers(0, len(candidates)))]
            d     = data[idx]
            delta = float(rng.choice([0.4, 0.5, 0.6, 0.7, 0.8]) * rng.choice([-1, 1]))
            new_angle = float(d.operation.params[0]) + delta
            new_qc = QuantumCircuit(circuit.num_qubits)
            for i, dd in enumerate(data):
                if i == idx:
                    getattr(new_qc, dd.operation.name)(new_angle, dd.qubits[0]._index)
                else:
                    new_qc.append(dd)
            return new_qc, {"type": "angle", "index": idx, "gate": d.operation.name,
                            "original_angle": float(d.operation.params[0]), "delta": delta}

    def _noiseless_probs(self, circuit: QuantumCircuit, n_qubits: int) -> dict:
        sv = Statevector.from_label("0" * n_qubits).evolve(circuit)
        return {f"|{i:0{n_qubits}b}>": float(abs(a) ** 2) for i, a in enumerate(sv.data)}

    def _noisy_probs(self, circuit: QuantumCircuit, n_qubits: int) -> dict:
        """Depolarizing noise via density matrix simulation (reused from cat_i_noise)."""
        dim = 2 ** n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0

        I2 = np.eye(2, dtype=complex)
        X  = np.array([[0, 1], [1, 0]], dtype=complex)
        Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z  = np.array([[1, 0], [0, -1]], dtype=complex)

        def embed(mat_2x2, q):
            ops = [I2.copy() for _ in range(n_qubits)]
            ops[q] = mat_2x2
            res = ops[n_qubits - 1]
            for k in range(n_qubits - 2, -1, -1):
                res = np.kron(res, ops[k])
            return res

        for inst in circuit.data:
            qubits = [q._index for q in inst.qubits]
            sub_qc = QuantumCircuit(n_qubits)
            sub_qc.append(inst.operation, qubits)
            U = np.array(Operator(sub_qc).data)
            rho = U @ rho @ U.conj().T

            if len(qubits) == 1:
                q, p = qubits[0], P_1Q
                rho = (1 - p) * rho + (p / 3) * sum(
                    embed(P, q) @ rho @ embed(P, q).conj().T for P in [X, Y, Z]
                )
            elif len(qubits) == 2:
                p = P_2Q
                depol = (1 - p) * rho
                paulis = [I2, X, Y, Z]
                for pi in paulis:
                    for pj in paulis:
                        if np.allclose(pi, I2) and np.allclose(pj, I2):
                            continue
                        P = embed(pi, qubits[0]) @ embed(pj, qubits[1])
                        depol += (p / 15) * (P @ rho @ P.conj().T)
                rho = depol

        return {f"|{i:0{n_qubits}b}>": float(rho[i, i].real) for i in range(dim)}

    def _build_prompt(self, qasm: str, noiseless: dict, noisy: dict, n_qubits: int) -> str:
        # Show top states sorted by ideal probability (non-trivial states first)
        states = sorted(noiseless.keys(), key=lambda s: -noiseless[s])

        noiseless_str = "\n".join(
            f"  {s}: {noiseless[s]:.4f}" for s in states if noiseless[s] > 0.001 or noisy[s] > 0.001
        )
        noisy_str = "\n".join(
            f"  {s}: {noisy[s]:.4f}" for s in states if noiseless[s] > 0.001 or noisy[s] > 0.001
        )

        tvd = 0.5 * sum(abs(noisy[s] - noiseless[s]) for s in noiseless)

        return (
            f"A {n_qubits}-qubit circuit was run on noisy hardware. "
            f"Determine whether the output deviation is from a LOGIC BUG in the circuit "
            f"or from HARDWARE NOISE ONLY.\n\n"
            f"Circuit (OpenQASM 2.0):\n```\n{qasm}\n```\n\n"
            f"Noise model (depolarizing, applied after each gate):\n"
            f"  After each 1-qubit gate: p = {P_1Q} (D(ρ) = (1-p)ρ + p/3·(XρX + YρY + ZρZ))\n"
            f"  After each 2-qubit gate: p = {P_2Q} (D(ρ) = (1-p)ρ + p/15·Σ_{{P≠II}} PρP)\n\n"
            f"Expected noiseless output (from the INTENDED correct circuit):\n"
            f"{noiseless_str}\n\n"
            f"Observed noisy output (from running THIS circuit on noisy hardware):\n"
            f"{noisy_str}\n\n"
            f"Total variation distance between expected and observed: {tvd:.4f}\n\n"
            f"IMPORTANT: The noise model above would cause a TVD of roughly "
            f"{P_1Q:.2f}–{P_2Q:.2f} per gate on a typical circuit. "
            f"Simulate this circuit with the given noise model to decide.\n\n"
            f"If there IS a logic bug: identify it, write a Qiskit function `solve()` that "
            f"returns the CORRECTED QuantumCircuit.\n"
            f"If there is NO logic bug (noise only): write `solve()` that returns the "
            f"ORIGINAL circuit UNCHANGED.\n\n"
            f"Only return the code for `solve()`. No explanation.\n"
        )
