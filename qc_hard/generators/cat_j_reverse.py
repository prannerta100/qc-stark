"""T11: Circuit Reverse Engineering — identify what algorithm a circuit implements."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qiskit.circuit.library import QFT
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel


ALGORITHMS = {
    "qft": "Quantum Fourier Transform",
    "inverse_qft": "Inverse Quantum Fourier Transform",
    "ghz": "GHZ State Preparation",
    "bell": "Bell State Preparation",
    "grover_diffusion": "Grover Diffusion Operator",
    "swap_test": "SWAP Test",
    "teleportation": "Quantum Teleportation",
    "phase_estimation_core": "Phase Estimation (core)",
    "bernstein_vazirani": "Bernstein-Vazirani Algorithm",
    "deutsch_jozsa": "Deutsch-Jozsa Algorithm",
}

LEVEL_CONFIG = {
    DifficultyLevel.TEXTBOOK: {
        "algorithms": ["bell", "ghz"],
        "n_qubits_range": (2, 3),
        "obfuscate": False,
    },
    DifficultyLevel.HOMEWORK: {
        "algorithms": ["qft", "inverse_qft", "ghz", "teleportation"],
        "n_qubits_range": (3, 4),
        "obfuscate": False,
    },
    DifficultyLevel.EXAM: {
        "algorithms": ["qft", "grover_diffusion", "swap_test", "bernstein_vazirani"],
        "n_qubits_range": (3, 5),
        "obfuscate": True,
    },
    DifficultyLevel.RESEARCH: {
        "algorithms": ["phase_estimation_core", "bernstein_vazirani", "deutsch_jozsa", "grover_diffusion"],
        "n_qubits_range": (4, 6),
        "obfuscate": True,
    },
    DifficultyLevel.OPEN: {
        "algorithms": ["phase_estimation_core", "deutsch_jozsa", "grover_diffusion", "bernstein_vazirani"],
        "n_qubits_range": (5, 7),
        "obfuscate": True,
    },
}


def _build_bell(n_qubits, rng):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc, "bell", "|00> + |11> (unnormalized)"


def _build_ghz(n_qubits, rng):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc, "ghz", f"|{'0'*n_qubits}> + |{'1'*n_qubits}> (unnormalized)"


def _build_qft(n_qubits, rng):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)
    return qc, "qft", "Applies QFT to the input register"


def _build_inverse_qft(n_qubits, rng):
    qc, _, _ = _build_qft(n_qubits, rng)
    qc = qc.inverse()
    return qc, "inverse_qft", "Applies inverse QFT to the input register"


def _build_grover_diffusion(n_qubits, rng):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc, "grover_diffusion", "Reflects about the uniform superposition state"


def _build_swap_test(n_qubits, rng):
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    return qc, "swap_test", "Measures overlap between states on qubits 1 and 2"


def _build_teleportation(n_qubits, rng):
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc, "teleportation", "Teleports qubit 0 state to qubit 2"


def _build_phase_estimation_core(n_qubits, rng):
    n_count = n_qubits - 1
    qc = QuantumCircuit(n_qubits)
    qc.x(n_qubits - 1)
    for i in range(n_count):
        qc.h(i)
    angle = rng.choice([np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3])
    for i in range(n_count):
        for _ in range(2 ** i):
            qc.cp(angle, i, n_qubits - 1)
    qft_inv = QFT(n_count, inverse=True, do_swaps=True)
    qc.compose(qft_inv, qubits=range(n_count), inplace=True)
    return qc, "phase_estimation_core", f"Estimates phase {angle/np.pi:.4f}*pi of a unitary"


def _build_bernstein_vazirani(n_qubits, rng):
    n_input = n_qubits - 1
    secret = rng.integers(1, 2**n_input)
    secret_bits = f"{secret:0{n_input}b}"

    qc = QuantumCircuit(n_qubits)
    qc.x(n_input)
    qc.h(range(n_qubits))
    for i, bit in enumerate(reversed(secret_bits)):
        if bit == '1':
            qc.cx(i, n_input)
    qc.h(range(n_input))
    return qc, "bernstein_vazirani", f"Finds secret string s={secret_bits}"


def _build_deutsch_jozsa(n_qubits, rng):
    n_input = n_qubits - 1
    is_constant = bool(rng.integers(0, 2))

    qc = QuantumCircuit(n_qubits)
    qc.x(n_input)
    qc.h(range(n_qubits))

    if is_constant:
        if rng.integers(0, 2):
            qc.x(n_input)
    else:
        targets = rng.choice(range(n_input), size=max(1, n_input // 2), replace=False)
        for t in targets:
            qc.cx(int(t), n_input)

    qc.h(range(n_input))
    fn_type = "constant" if is_constant else "balanced"
    return qc, "deutsch_jozsa", f"Determines function is {fn_type}"


BUILDERS = {
    "bell": _build_bell,
    "ghz": _build_ghz,
    "qft": _build_qft,
    "inverse_qft": _build_inverse_qft,
    "grover_diffusion": _build_grover_diffusion,
    "swap_test": _build_swap_test,
    "teleportation": _build_teleportation,
    "phase_estimation_core": _build_phase_estimation_core,
    "bernstein_vazirani": _build_bernstein_vazirani,
    "deutsch_jozsa": _build_deutsch_jozsa,
}


def _obfuscate_circuit(qc, rng):
    """Decompose to basic gates and optionally permute qubit labels."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=0, basis_gates=['cx', 'u3', 'u2', 'u1'])
    decomposed = pm.run(qc)
    return decomposed


class ReverseEngineeringGenerator(BaseGenerator):
    category = Category.J_REVERSE
    subtask = "T11_reverse"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level.value * 10000)
        config = LEVEL_CONFIG[level]

        algo_key = rng.choice(config["algorithms"])
        n_qubits = int(rng.integers(config["n_qubits_range"][0], config["n_qubits_range"][1] + 1))

        if algo_key in ("bell", "swap_test", "teleportation"):
            n_qubits = 3 if algo_key in ("swap_test", "teleportation") else 2

        builder = BUILDERS[algo_key]
        qc, algo_id, expected_output = builder(n_qubits, rng)

        if config["obfuscate"]:
            qc = _obfuscate_circuit(qc, rng)

        qasm_str = qasm_dumps(qc)

        prompt = (
            f"Analyze the following {qc.num_qubits}-qubit quantum circuit and identify "
            f"what algorithm or subroutine it implements.\n\n"
            f"Circuit (OpenQASM 2.0):\n```\n{qasm_str}\n```\n\n"
            f"Write a function `solve()` that returns a dictionary with:\n"
            f"  - 'algorithm': string name of the algorithm (e.g., 'qft', 'grover_diffusion', "
            f"'ghz', 'bell', 'teleportation', 'swap_test', 'bernstein_vazirani', "
            f"'deutsch_jozsa', 'phase_estimation_core', 'inverse_qft')\n"
            f"  - 'output': the expected output/result when run on the all-zeros input state "
            f"(describe as a string)\n\n"
            f"Valid algorithm names: {list(ALGORITHMS.keys())}\n"
        )

        return TaskInstance(
            task_id=f"T11_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "algorithm": algo_id,
                "algorithm_full_name": ALGORITHMS[algo_id],
                "n_qubits": qc.num_qubits,
                "expected_output": expected_output,
                "obfuscated": config["obfuscate"],
                "circuit_qasm": qasm_str,
            },
        )
