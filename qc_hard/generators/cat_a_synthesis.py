import numpy as np
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_QUBITS = {
    DifficultyLevel.TEXTBOOK: 2,
    DifficultyLevel.HOMEWORK: 3,
    DifficultyLevel.EXAM: 4,
    DifficultyLevel.RESEARCH: 5,
    DifficultyLevel.OPEN: 6,
}


class StatePrepGenerator(BaseGenerator):
    category = Category.A_SYNTHESIS
    subtask = "A1_state_prep"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        n_qubits = LEVEL_TO_QUBITS[level]
        dim = 2**n_qubits

        z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        u = q * ph[np.newaxis, :]
        target_state = u[:, 0]

        if level == DifficultyLevel.TEXTBOOK:
            target_state = target_state.real
            target_state = target_state / np.linalg.norm(target_state)

        state_serialized = [[float(a.real), float(a.imag)] for a in target_state]
        prompt = self._build_prompt(state_serialized, n_qubits, level)

        return TaskInstance(
            task_id=f"A1_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "target_state": state_serialized,
                "n_qubits": n_qubits,
            },
        )

    def _build_prompt(self, state: list, n_qubits: int, level: DifficultyLevel) -> str:
        state_str = "\n".join(
            f"  |{i:0{n_qubits}b}>: {s[0]:.15g} + {s[1]:.15g}i"
            for i, s in enumerate(state)
        )
        state_list_str = str([complex(s[0], s[1]) for s in state])

        # Level 1-2: allow initialize() — tests API knowledge (baseline)
        # Level 3+: disallow initialize() — tests decomposition reasoning
        if level.value <= 2:
            constraint = (
                "- You may use QuantumCircuit.initialize(statevector, qubits) or any standard gates.\n"
                "- Ensure the statevector is normalized (np.linalg.norm == 1) before passing to initialize().\n"
            )
        else:
            constraint = (
                "- Use ONLY elementary gates: rx, ry, rz, h, x, y, z, cx, cz, ccx, swap.\n"
                "- Do NOT use initialize(), prepare_state(), or isometry().\n"
                "- You must decompose the state preparation into a gate sequence yourself.\n"
            )

        return (
            f"Write a Qiskit function `solve()` that returns a QuantumCircuit on {n_qubits} qubits "
            f"which prepares the following quantum state from |{'0'*n_qubits}>:\n\n"
            f"{state_str}\n\n"
            f"As a Python list: {state_list_str}\n\n"
            f"Requirements:\n"
            f"{constraint}"
            f"- The circuit should achieve fidelity > 0.999 with the target state.\n"
            f"- Return ONLY the QuantumCircuit object from solve().\n"
        )
