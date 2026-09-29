"""G2: First-order Trotter error estimation."""
import numpy as np
from scipy.linalg import expm
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

LEVEL_CONFIG = {
    # More terms + fewer steps + longer time = larger, harder-to-estimate error.
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "n_terms": 2, "t": 0.5, "r": 10},
    DifficultyLevel.HOMEWORK: {"n_qubits": 2, "n_terms": 3, "t": 0.8, "r": 5},
    DifficultyLevel.EXAM:     {"n_qubits": 3, "n_terms": 4, "t": 1.0, "r": 3},
    DifficultyLevel.RESEARCH: {"n_qubits": 3, "n_terms": 5, "t": 1.0, "r": 2},
    DifficultyLevel.OPEN:     {"n_qubits": 4, "n_terms": 6, "t": 1.2, "r": 1},
}


def _pauli_to_matrix(pauli_str: str) -> np.ndarray:
    # Qiskit convention: rightmost = qubit 0, so kron from left to right.
    result = PAULI_MATS[pauli_str[-1]]
    for p in reversed(pauli_str[:-1]):
        result = np.kron(PAULI_MATS[p], result)
    return result


def _first_order_trotter(terms, coefficients, t: float, r: int) -> np.ndarray:
    """Product formula: (∏_k exp(-i h_k P_k t/r))^r"""
    dim = _pauli_to_matrix(terms[0]).shape[0]
    step = np.eye(dim, dtype=complex)
    dt = t / r
    for coeff, pauli_str in zip(coefficients, terms):
        P = _pauli_to_matrix(pauli_str)
        # exp(-i * coeff * P * dt) = cos(c*dt) I - i sin(c*dt) P  (since P²=I for Pauli)
        theta = coeff * dt
        step = step @ (np.cos(theta) * np.eye(dim) - 1j * np.sin(theta) * P)
    U = np.linalg.matrix_power(step, r)
    return U


class TrotterErrorGenerator(BaseGenerator):
    category = Category.G_TROTTERIZATION
    subtask = "G2_trotter_error"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level.value * 10000)
        cfg = LEVEL_CONFIG[level]
        n_qubits = cfg["n_qubits"]
        n_terms = cfg["n_terms"]
        t = cfg["t"]
        r = cfg["r"]

        paulis = ["I", "X", "Y", "Z"]
        terms, coefficients = [], []
        while len(terms) < n_terms:
            p = "".join(str(rng.choice(paulis)) for _ in range(n_qubits))
            if p == "I" * n_qubits or p in terms:
                continue
            terms.append(p)
            coefficients.append(float(rng.uniform(-1.5, 1.5)))

        H = sum(c * _pauli_to_matrix(t_str) for c, t_str in zip(coefficients, terms))
        U_exact = expm(-1j * H * t)
        U_trotter = _first_order_trotter(terms, coefficients, t, r)

        diff = U_exact - U_trotter
        # Operator (spectral) norm = largest singular value
        ground_truth_error = float(np.linalg.norm(diff, ord=2))

        term_strs = [f"{c:+.4f} * {p}" for c, p in zip(coefficients, terms)]
        prompt = self._build_prompt(term_strs, t, r, n_qubits)

        return TaskInstance(
            task_id=f"G2_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "n_qubits": n_qubits,
                "terms": terms,
                "coefficients": coefficients,
                "t": t,
                "r": r,
                "ground_truth_error": ground_truth_error,
            },
        )

    def _build_prompt(self, term_strs: list, t: float, r: int, n_qubits: int) -> str:
        ham_str = "\n  ".join(term_strs)
        return (
            f"Compute the operator norm error of the first-order Trotter approximation "
            f"to e^{{-iHt}} for the following {n_qubits}-qubit Hamiltonian.\n\n"
            f"H =\n  {ham_str}\n\n"
            f"Parameters:\n"
            f"  t = {t}  (evolution time)\n"
            f"  r = {r}  (number of Trotter steps)\n\n"
            f"The first-order product formula for one step is:\n"
            f"  U_step = ∏_k exp(-i h_k P_k t/r)\n"
            f"  (product over terms in the order listed above)\n"
            f"The full approximation is U_trotter = U_step^r.\n\n"
            f"For a Pauli operator P (with P² = I):\n"
            f"  exp(-i θ P) = cos(θ) I - i sin(θ) P\n\n"
            f"Pauli string convention: rightmost character acts on qubit 0.\n\n"
            f"Compute the operator (spectral) norm of the error:\n"
            f"  ε = ‖ exp(-iHt) - U_trotter ‖₂\n"
            f"where ‖·‖₂ is the largest singular value (numpy.linalg.norm(M, ord=2)).\n\n"
            f"Write a function `solve()` that returns ε as a float.\n"
            f"You may use numpy and scipy. Do NOT import qiskit.\n"
        )
