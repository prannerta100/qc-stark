import warnings
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.quantum_info import Statevector, Operator
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "n_gates": 4, "p_1q": 0.01, "p_2q": 0.05},
    DifficultyLevel.HOMEWORK: {"n_qubits": 3, "n_gates": 8, "p_1q": 0.005, "p_2q": 0.02},
    DifficultyLevel.EXAM: {"n_qubits": 4, "n_gates": 15, "p_1q": 0.003, "p_2q": 0.015},
    DifficultyLevel.RESEARCH: {"n_qubits": 5, "n_gates": 20, "p_1q": 0.002, "p_2q": 0.01},
    DifficultyLevel.OPEN: {"n_qubits": 5, "n_gates": 30, "p_1q": 0.001, "p_2q": 0.008},
}

SINGLE_QUBIT_GATES = ["h", "x", "y", "z", "s", "t"]
ROTATION_GATES = ["rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx"]


class NoiseFidelityGenerator(BaseGenerator):
    category = Category.I_NOISE
    subtask = "T7_noise_fidelity"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_qubits = params["n_qubits"]
        n_gates = params["n_gates"]
        p_1q = params["p_1q"]
        p_2q = params["p_2q"]

        # Generate random circuit
        circuit = self._random_circuit(rng, n_qubits, n_gates)
        circuit_qasm = qasm_dumps(circuit)

        # Compute ideal state
        ideal_sv = Statevector.from_label("0" * n_qubits).evolve(circuit)

        # Compute noisy density matrix using depolarizing noise from first principles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            noisy_rho = self._simulate_noisy(circuit, n_qubits, p_1q, p_2q)

        # Compute fidelity: <psi_ideal|rho_noisy|psi_ideal>
        ideal_state = np.array(ideal_sv.data)
        ground_truth_fidelity = float(
            np.real(ideal_state.conj() @ noisy_rho @ ideal_state)
        )

        prompt = self._build_prompt(circuit_qasm, n_qubits, p_1q, p_2q, level)

        return TaskInstance(
            task_id=f"T7_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "circuit_qasm": circuit_qasm,
                "n_qubits": n_qubits,
                "p_1q": p_1q,
                "p_2q": p_2q,
                "ground_truth_fidelity": ground_truth_fidelity,
                "n_gates": n_gates,
            },
        )

    def _random_circuit(self, rng, n_qubits: int, n_gates: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for _ in range(n_gates):
            if n_qubits > 1 and rng.random() < 0.3:
                gate = str(rng.choice(TWO_QUBIT_GATES))
                qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
                qc.cx(int(qubits[0]), int(qubits[1]))
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

    def _simulate_noisy(self, circuit: QuantumCircuit, n_qubits: int,
                        p_1q: float, p_2q: float) -> np.ndarray:
        """Simulate noisy circuit using density matrix with depolarizing noise."""
        dim = 2 ** n_qubits
        # Start with |0...0><0...0|
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0

        for inst in circuit.data:
            qubits = [q._index for q in inst.qubits]
            n_gate_qubits = len(qubits)

            # Build a sub-circuit with just this gate to get the full-space unitary
            sub_qc = QuantumCircuit(n_qubits)
            sub_qc.append(inst.operation, qubits)
            full_unitary = np.array(Operator(sub_qc).data)

            # Apply ideal gate: rho -> U * rho * U^dag
            rho = full_unitary @ rho @ full_unitary.conj().T

            # Apply depolarizing noise on the affected qubits
            if n_gate_qubits == 1:
                rho = self._apply_depolarizing_1q(rho, qubits[0], n_qubits, p_1q)
            elif n_gate_qubits == 2:
                rho = self._apply_depolarizing_2q(rho, qubits, n_qubits, p_2q)

        return rho

    def _single_qubit_op_in_full_space(self, mat_2x2: np.ndarray, qubit: int,
                                        n_qubits: int) -> np.ndarray:
        """Embed a 2x2 matrix acting on `qubit` into the full 2^n Hilbert space."""
        # Qiskit convention: qubit 0 is least significant (rightmost in kron)
        ops = [np.eye(2, dtype=complex)] * n_qubits
        ops[qubit] = mat_2x2
        # Kron order: qubit n-1 (leftmost) ... qubit 0 (rightmost)
        result = ops[n_qubits - 1]
        for i in range(n_qubits - 2, -1, -1):
            result = np.kron(result, ops[i])
        return result

    def _apply_depolarizing_1q(self, rho: np.ndarray, qubit: int,
                                n_qubits: int, p: float) -> np.ndarray:
        """Apply single-qubit depolarizing channel: (1-p)*rho + p/3*(X rho X + Y rho Y + Z rho Z)."""
        if p < 1e-15:
            return rho

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        result = (1 - p) * rho
        for pauli in [X, Y, Z]:
            P = self._single_qubit_op_in_full_space(pauli, qubit, n_qubits)
            result += (p / 3) * (P @ rho @ P.conj().T)

        return result

    def _apply_depolarizing_2q(self, rho: np.ndarray, qubits: list[int],
                                n_qubits: int, p: float) -> np.ndarray:
        """Apply two-qubit depolarizing channel: (1-p)*rho + p/15 * sum_{P!=II} P rho P."""
        if p < 1e-15:
            return rho

        I2 = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        paulis_1q = [I2, X, Y, Z]

        result = (1 - p) * rho
        for i, p1 in enumerate(paulis_1q):
            for j, p2 in enumerate(paulis_1q):
                if i == 0 and j == 0:
                    continue  # skip II
                # Apply p1 on qubits[0] and p2 on qubits[1]
                P = self._single_qubit_op_in_full_space(p1, qubits[0], n_qubits) @ \
                    self._single_qubit_op_in_full_space(p2, qubits[1], n_qubits)
                result += (p / 15) * (P @ rho @ P.conj().T)

        return result

    def _gate_list_from_circuit(self, circuit: QuantumCircuit) -> list[dict]:
        """Convert circuit to a plain gate list with explicit matrix representations."""
        import json
        gates = []
        for inst in circuit.data:
            qubits = [q._index for q in inst.qubits]
            name = inst.operation.name
            params = [float(p) for p in inst.operation.params] if inst.operation.params else []
            gates.append({"gate": name, "qubits": qubits, "params": params})
        return gates

    def _build_prompt(self, circuit_qasm: str, n_qubits: int,
                      p_1q: float, p_2q: float, level: DifficultyLevel) -> str:
        from qiskit.qasm2 import loads as qasm_loads
        circuit = qasm_loads(circuit_qasm)
        gate_list = self._gate_list_from_circuit(circuit)

        gate_lines = []
        for g in gate_list:
            if g["params"]:
                param_str = ", ".join(f"{p:.6f}" for p in g["params"])
                gate_lines.append(f"  {g['gate']}({param_str}) on qubits {g['qubits']}")
            else:
                gate_lines.append(f"  {g['gate']} on qubits {g['qubits']}")
        gate_str = "\n".join(gate_lines)

        return (
            f"Given this {n_qubits}-qubit quantum circuit and noise model, predict the "
            f"output state fidelity F = <psi_ideal|rho_noisy|psi_ideal>.\n\n"
            f"The initial state is |{'0' * n_qubits}>.\n\n"
            f"Circuit ({len(gate_list)} gates applied in order):\n{gate_str}\n\n"
            f"Noise model: depolarizing noise applied AFTER each gate.\n"
            f"  - After each 1-qubit gate: depolarizing channel with probability p = {p_1q}\n"
            f"    D_1(rho) = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)\n"
            f"  - After each 2-qubit gate: depolarizing channel with probability p = {p_2q}\n"
            f"    D_2(rho) = (1-p)*rho + (p/15)*sum_{{P in Paulis\\II}} P*rho*P\n"
            f"    (sum over all 15 non-identity two-qubit Pauli operators)\n\n"
            f"Standard gate definitions:\n"
            f"  h: (1/sqrt(2))*[[1,1],[1,-1]]\n"
            f"  x: [[0,1],[1,0]]\n"
            f"  y: [[0,-i],[i,0]]\n"
            f"  z: [[1,0],[0,-1]]\n"
            f"  s: [[1,0],[0,i]]\n"
            f"  t: [[1,0],[0,exp(i*pi/4)]]\n"
            f"  rx(theta): [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]\n"
            f"  ry(theta): [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]\n"
            f"  rz(theta): [[exp(-i*t/2), 0], [0, exp(i*t/2)]]\n"
            f"  cx: CNOT (control, target) in computational basis\n\n"
            f"Multi-qubit operators use Qiskit convention: qubit 0 is LEAST significant "
            f"(rightmost in tensor product). E.g., for gate on qubit k in an n-qubit system, "
            f"the full operator is I_{{n-1}} ⊗ ... ⊗ Gate_k ⊗ ... ⊗ I_0.\n\n"
            f"Write a function `solve()` that returns a float (the predicted fidelity).\n"
            f"You may use numpy. Do NOT use qiskit, qiskit_aer, or any quantum simulation library.\n"
            f"Implement density matrix propagation from scratch.\n"
        )
