import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    DifficultyLevel.TEXTBOOK: {"n_qubits": 2, "n_terms": 3, "t": 0.5, "commuting": False},
    DifficultyLevel.HOMEWORK: {"n_qubits": 2, "n_terms": 4, "t": 0.8, "commuting": False},
    DifficultyLevel.EXAM: {"n_qubits": 3, "n_terms": 5, "t": 1.0, "commuting": False},
    DifficultyLevel.RESEARCH: {"n_qubits": 4, "n_terms": 8, "t": 1.0, "commuting": False},
    DifficultyLevel.OPEN: {"n_qubits": 5, "n_terms": 12, "t": 1.5, "commuting": False},
}

# Pauli matrices
I_MAT = np.eye(2, dtype=complex)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_MATS = {"I": I_MAT, "X": X_MAT, "Y": Y_MAT, "Z": Z_MAT}
PAULI_LABELS = ["I", "X", "Y", "Z"]


class TrotterGenerator(BaseGenerator):
    category = Category.G_TROTTERIZATION
    subtask = "T5_trotter"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_qubits = params["n_qubits"]
        n_terms = params["n_terms"]
        t = params["t"]
        commuting = params["commuting"]

        # Generate Pauli terms
        terms = self._generate_terms(rng, n_qubits, n_terms, commuting)
        coefficients = [float(rng.uniform(-2.0, 2.0)) for _ in range(n_terms)]

        # Build Hamiltonian matrix
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for coeff, pauli_str in zip(coefficients, terms):
            H += coeff * self._pauli_string_to_matrix(pauli_str)

        # Compute exact time evolution
        exact_unitary = expm(-1j * H * t)

        # Build term descriptions for prompt
        term_strs = []
        for coeff, pauli_str in zip(coefficients, terms):
            term_strs.append(f"{coeff:+.4f} * {pauli_str}")
        hamiltonian_str = "\n  ".join(term_strs)

        prompt = self._build_prompt(hamiltonian_str, t, n_qubits)

        return TaskInstance(
            task_id=f"T5_seed{seed}_L{level.value}",
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
                "exact_unitary_real": [[float(c.real) for c in row] for row in exact_unitary],
                "exact_unitary_imag": [[float(c.imag) for c in row] for row in exact_unitary],
            },
        )

    def _generate_terms(self, rng, n_qubits: int, n_terms: int, commuting: bool) -> list[str]:
        """Generate random Pauli string terms."""
        terms = []
        if commuting:
            # Use same Pauli type on different qubit subsets (ZI, IZ commute; XI, IX commute)
            pauli_type = str(rng.choice(["X", "Y", "Z"]))
            for i in range(n_terms):
                pauli_str = ["I"] * n_qubits
                # Each term acts on a distinct qubit with the same Pauli
                qubit = i % n_qubits
                pauli_str[qubit] = pauli_type
                term = "".join(pauli_str)
                if term not in terms and term != "I" * n_qubits:
                    terms.append(term)
            # If we didn't get enough (n_terms > n_qubits), add multi-qubit terms
            # that still commute (same Pauli type on all non-I positions)
            while len(terms) < n_terms:
                pauli_str = ["I"] * n_qubits
                n_active = int(rng.integers(2, n_qubits + 1))
                active_qubits = rng.choice(n_qubits, size=n_active, replace=False)
                for q in active_qubits:
                    pauli_str[q] = pauli_type
                term = "".join(pauli_str)
                if term not in terms:
                    terms.append(term)
        else:
            # Non-commuting: random Pauli strings with at least 2 non-I positions
            attempts = 0
            while len(terms) < n_terms and attempts < n_terms * 20:
                attempts += 1
                pauli_str = [str(rng.choice(PAULI_LABELS)) for _ in range(n_qubits)]
                term = "".join(pauli_str)
                # Exclude all-identity
                if term == "I" * n_qubits:
                    continue
                if term not in terms:
                    terms.append(term)
        return terms

    def _pauli_string_to_matrix(self, pauli_str: str) -> np.ndarray:
        """Convert a Pauli string like 'XYZ' to its matrix representation.
        Uses Qiskit convention: qubit 0 is RIGHTMOST in tensor product.
        So 'YI' (Y on q0, I on q1) = kron(I, Y)."""
        # Reverse the string to match Qiskit's little-endian convention
        reversed_str = pauli_str[::-1]
        result = PAULI_MATS[reversed_str[0]]
        for p in reversed_str[1:]:
            result = np.kron(result, PAULI_MATS[p])
        return result

    def _build_prompt(self, hamiltonian_str: str, t: float, n_qubits: int) -> str:
        return (
            f"Construct a Trotter circuit for the time evolution operator e^{{-iHt}} "
            f"with t={t} for the following {n_qubits}-qubit Hamiltonian:\n\n"
            f"  H = {hamiltonian_str}\n\n"
            f"Write a function `solve()` that returns a QuantumCircuit on {n_qubits} qubits.\n"
            f"The circuit should achieve operator fidelity > 0.99 with the exact time evolution.\n"
            f"You may use multiple Trotter steps. Use only standard gates (rx, ry, rz, cx, h).\n"
        )
