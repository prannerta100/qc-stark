"""E2: VQE ground state energy estimation."""
import numpy as np
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

LEVEL_CONFIG = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "n_terms": 3, "any_method": True,  "give_matrix": True},
    DifficultyLevel.HOMEWORK: {"n_qubits": 2, "n_terms": 4, "any_method": True,  "give_matrix": True},
    DifficultyLevel.EXAM:     {"n_qubits": 3, "n_terms": 5, "any_method": False, "give_matrix": False},
    DifficultyLevel.RESEARCH: {"n_qubits": 3, "n_terms": 6, "any_method": False, "give_matrix": False},
    DifficultyLevel.OPEN:     {"n_qubits": 4, "n_terms": 8, "any_method": False, "give_matrix": False},
}


def _pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    # Qiskit/SparsePauliOp convention: rightmost character acts on qubit 0.
    # "XY" → X on qubit 1, Y on qubit 0 → kron(X, Y)
    result = PAULI_MATS[pauli_str[-1]]
    for p in reversed(pauli_str[:-1]):
        result = np.kron(PAULI_MATS[p], result)
    return result


class VQEGenerator(BaseGenerator):
    category = Category.E_OPTIMIZATION
    subtask = "E2_vqe"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level.value * 10000)
        cfg = LEVEL_CONFIG[level]
        n_qubits = cfg["n_qubits"]
        n_terms = cfg["n_terms"]
        paulis = ["I", "X", "Y", "Z"]

        terms, coefficients = [], []
        while len(terms) < n_terms:
            p = "".join(str(rng.choice(paulis)) for _ in range(n_qubits))
            if p == "I" * n_qubits or p in terms:
                continue
            terms.append(p)
            coefficients.append(float(rng.uniform(-1.5, 1.5)))

        H = sum(c * _pauli_string_to_matrix(t) for c, t in zip(coefficients, terms))
        ground_energy = float(np.linalg.eigvalsh(H)[0])

        prompt = self._build_prompt(terms, coefficients, n_qubits, cfg,
                                    H if cfg["give_matrix"] else None)

        return TaskInstance(
            task_id=f"E2_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "n_qubits": n_qubits,
                "terms": terms,
                "coefficients": coefficients,
                "ground_energy": ground_energy,
                "hamiltonian_real": [[float(v.real) for v in row] for row in H],
                "hamiltonian_imag": [[float(v.imag) for v in row] for row in H],
                "any_method": cfg["any_method"],
            },
        )

    def _build_prompt(self, terms, coefficients, n_qubits, cfg, H_matrix) -> str:
        pauli_lines = "\n".join(f"  {c:+.6f} * {p}" for c, p in zip(coefficients, terms))

        if H_matrix is not None:
            rows = []
            for row in H_matrix:
                row_str = "  [" + ", ".join(
                    f"{v.real:+.4f}{'+' if v.imag >= 0 else ''}{v.imag:.4f}j"
                    for v in row
                ) + "]"
                rows.append(row_str)
            matrix_section = "\nAs a matrix (rows/cols ordered |0...0> to |1...1>):\n" + "\n".join(rows) + "\n"
        else:
            matrix_section = ""

        if cfg["any_method"]:
            method = (
                "You may use any method including exact diagonalization (numpy.linalg.eigvalsh).\n"
            )
        else:
            method = (
                "Use a variational quantum eigensolver (VQE) approach:\n"
                "  1. Define a parametrized ansatz circuit.\n"
                "  2. Compute E(θ) = <ψ(θ)|H|ψ(θ)> using Qiskit's Estimator or by hand.\n"
                "  3. Minimize E(θ) over θ with scipy.optimize.minimize.\n"
                "Do NOT use numpy.linalg.eig, scipy.linalg.eigh, or any exact diagonalization.\n"
            )

        return (
            f"Find the ground state energy (lowest eigenvalue) of the following "
            f"{n_qubits}-qubit Hamiltonian:\n\n"
            f"H =\n{pauli_lines}\n"
            f"{matrix_section}\n"
            f"Pauli string convention: rightmost character acts on qubit 0.\n"
            f"Example: 'ZI' = Z on qubit 1, I on qubit 0.\n\n"
            f"{method}\n"
            f"Write a function `solve()` that returns the ground state energy as a float.\n"
        )
