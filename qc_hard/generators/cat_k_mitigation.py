"""T10: Error Mitigation — implement ZNE or similar to improve noisy estimates."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.quantum_info import Operator, Statevector, SparsePauliOp
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel


LEVEL_CONFIG = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "n_gates": 4, "noise_p": 0.05, "method": "zne_linear"},
    DifficultyLevel.HOMEWORK: {"n_qubits": 2, "n_gates": 6, "noise_p": 0.04, "method": "zne_linear"},
    DifficultyLevel.EXAM: {"n_qubits": 3, "n_gates": 8, "noise_p": 0.03, "method": "zne_quadratic"},
    DifficultyLevel.RESEARCH: {"n_qubits": 3, "n_gates": 10, "noise_p": 0.02, "method": "zne_exponential"},
    DifficultyLevel.OPEN: {"n_qubits": 4, "n_gates": 12, "noise_p": 0.02, "method": "zne_exponential"},
}

SINGLE_GATES = ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz']
TWO_GATES = ['cx', 'cz']


def _random_circuit(n_qubits, n_gates, rng):
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.3:
            gate = rng.choice(TWO_GATES)
            qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
            if gate == 'cx':
                qc.cx(int(qubits[0]), int(qubits[1]))
            else:
                qc.cz(int(qubits[0]), int(qubits[1]))
        else:
            gate = rng.choice(SINGLE_GATES)
            qubit = int(rng.integers(0, n_qubits))
            if gate in ('rx', 'ry', 'rz'):
                angle = float(rng.uniform(0.1, np.pi))
                getattr(qc, gate)(angle, qubit)
            else:
                getattr(qc, gate)(qubit)
    return qc


def _compute_ideal_expectation(qc, observable):
    sv = Statevector.from_instruction(qc)
    return float(np.real(sv.expectation_value(observable)))


def _compute_noisy_expectation(qc, observable, noise_p, n_qubits):
    """Compute expectation under depolarizing noise (density matrix simulation)."""
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0

    paulis_1q = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]])
    ]

    for inst in qc.data:
        gate_unitary = Operator(inst.operation).data
        qubit_indices = [q._index for q in inst.qubits]
        n_gate_qubits = len(qubit_indices)

        full_unitary = _embed_gate(gate_unitary, qubit_indices, n_qubits)
        rho = full_unitary @ rho @ full_unitary.conj().T

        if n_gate_qubits == 1:
            p = noise_p
            depol = (1 - p) * rho
            for pauli in paulis_1q[1:]:
                full_p = _embed_gate(pauli, qubit_indices, n_qubits)
                depol += (p / 3) * (full_p @ rho @ full_p.conj().T)
            rho = depol
        elif n_gate_qubits == 2:
            p = noise_p * 2
            depol = (1 - p) * rho
            for i in range(4):
                for j in range(4):
                    if i == 0 and j == 0:
                        continue
                    pp = np.kron(paulis_1q[i], paulis_1q[j])
                    full_p = _embed_gate(pp, qubit_indices, n_qubits)
                    depol += (p / 15) * (full_p @ rho @ full_p.conj().T)
            rho = depol

    obs_matrix = SparsePauliOp.from_list([(observable, 1.0)]).to_matrix()
    return float(np.real(np.trace(obs_matrix @ rho)))


def _embed_gate(gate_matrix, qubit_indices, n_qubits):
    dim = 2 ** n_qubits
    n_gate = len(qubit_indices)
    full = np.eye(dim, dtype=complex)

    for row in range(dim):
        for col in range(dim):
            row_bits = [(row >> (n_qubits - 1 - q)) & 1 for q in qubit_indices]
            col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in qubit_indices]

            other_row_bits = [(row >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits) if q not in qubit_indices]
            other_col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits) if q not in qubit_indices]

            if other_row_bits != other_col_bits:
                full[row, col] = 0
            else:
                gate_row = sum(b << (n_gate - 1 - i) for i, b in enumerate(row_bits))
                gate_col = sum(b << (n_gate - 1 - i) for i, b in enumerate(col_bits))
                full[row, col] = gate_matrix[gate_row, gate_col]

    return full


def _choose_observable(n_qubits, rng):
    paulis = ['I', 'X', 'Y', 'Z']
    while True:
        obs = ''.join(rng.choice(paulis) for _ in range(n_qubits))
        if obs != 'I' * n_qubits:
            return obs


class ErrorMitigationGenerator(BaseGenerator):
    category = Category.K_MITIGATION
    subtask = "T10_mitigation"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level.value * 10000)
        config = LEVEL_CONFIG[level]

        n_qubits = config["n_qubits"]
        n_gates = config["n_gates"]
        noise_p = config["noise_p"]
        method = config["method"]

        qc = _random_circuit(n_qubits, n_gates, rng)
        observable = _choose_observable(n_qubits, rng)

        ideal_val = _compute_ideal_expectation(qc, SparsePauliOp.from_list([(observable, 1.0)]))
        noisy_val = _compute_noisy_expectation(qc, observable, noise_p, n_qubits)

        qasm_str = qasm_dumps(qc)

        if "linear" in method:
            method_desc = "zero-noise extrapolation (ZNE) with linear fit"
            scale_factors = "1, 2, 3"
        elif "quadratic" in method:
            method_desc = "zero-noise extrapolation (ZNE) with quadratic fit"
            scale_factors = "1, 2, 3, 4"
        else:
            method_desc = "zero-noise extrapolation (ZNE) with exponential fit"
            scale_factors = "1, 2, 3, 4, 5"

        prompt = (
            f"Given the following {n_qubits}-qubit quantum circuit under depolarizing noise, "
            f"implement {method_desc} to estimate the noiseless expectation value "
            f"of the observable {observable}.\n\n"
            f"Circuit (OpenQASM 2.0):\n```\n{qasm_str}\n```\n\n"
            f"Noise model:\n"
            f"  - After each 1-qubit gate: depolarizing channel with p = {noise_p}\n"
            f"  - After each 2-qubit gate: depolarizing channel with p = {noise_p * 2}\n\n"
            f"Observable to measure: {observable}\n\n"
            f"Instructions:\n"
            f"  - Implement ZNE by computing the noisy expectation value at noise scale "
            f"factors [{scale_factors}] (multiply base noise probability by each factor).\n"
            f"  - Extrapolate to the zero-noise limit using {'linear' if 'linear' in method else 'quadratic' if 'quadratic' in method else 'exponential'} fitting.\n"
            f"  - You must implement the noisy simulation yourself (density matrix evolution).\n"
            f"  - Do NOT use qiskit_aer or any noise simulation library.\n\n"
            f"Write a function `solve()` that returns the mitigated (zero-noise extrapolated) "
            f"expectation value as a float.\n"
        )

        return TaskInstance(
            task_id=f"T10_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "n_qubits": n_qubits,
                "n_gates": n_gates,
                "noise_p": noise_p,
                "observable": observable,
                "ideal_value": ideal_val,
                "noisy_value": noisy_val,
                "method": method,
                "circuit_qasm": qasm_str,
            },
        )
