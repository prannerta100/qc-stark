"""Verifier for B3: Noise vs. Logic Bug Discrimination.

Detailed metrics per result:
  has_bug            – ground truth
  code_crashed       – True if exec() or solve() raised an exception
  equiv_intended     – fidelity with correct (intended) unitary ≥ 0.999
  equiv_original     – fidelity with shown (original) unitary ≥ 0.999
  changed_circuit    – True if returned circuit ≠ original shown circuit
  fidelity_intended  – continuous fidelity with correct unitary
  fidelity_original  – continuous fidelity with shown (possibly buggy) unitary

Correct definition:
  - noise case (no bug): correct iff equiv_original (return it unchanged)
  - bug case:            correct iff equiv_intended (return the fixed version)
"""
import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

FIDELITY_THRESHOLD = 0.999


def _operator_fidelity(circuit, target: np.ndarray) -> float:
    from qiskit.quantum_info import Operator
    U   = np.array(Operator(circuit).data)
    dim = target.shape[0]
    return float(abs(np.trace(target.conj().T @ U)) / dim)


class NoiseBugDiscriminationVerifier(BaseVerifier):
    subtask = "B3_noise_logic"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code     = response.parsed_code or response.raw_response
        has_bug  = task.metadata["has_bug"]
        n_qubits = task.metadata["n_qubits"]

        correct_u = (np.array(task.metadata["correct_unitary_real"])
                     + 1j * np.array(task.metadata["correct_unitary_imag"]))
        shown_u   = (np.array(task.metadata["shown_unitary_real"])
                     + 1j * np.array(task.metadata["shown_unitary_imag"]))

        # ── try running the code ──────────────────────────────────────────────
        code_crashed       = False
        fidelity_intended  = 0.0
        fidelity_original  = 0.0
        equiv_intended     = False
        equiv_original     = False
        changed_circuit    = False
        crash_msg          = ""

        try:
            namespace = {}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            circuit = namespace["solve"]()

            fidelity_intended = _operator_fidelity(circuit, correct_u)
            fidelity_original = _operator_fidelity(circuit, shown_u)
            equiv_intended    = fidelity_intended >= FIDELITY_THRESHOLD
            equiv_original    = fidelity_original >= FIDELITY_THRESHOLD
            # "changed" means the model didn't just return the original circuit
            changed_circuit   = not equiv_original

        except Exception as e:
            code_crashed = True
            crash_msg    = str(e)[:500]

        # ── binary correct ────────────────────────────────────────────────────
        if code_crashed:
            correct = False
            score   = 0.0
        elif has_bug:
            # Bug case: must return the fixed (intended) circuit
            correct = equiv_intended
            score   = float(fidelity_intended)
        else:
            # Noise case: must return the original circuit UNCHANGED
            correct = equiv_original
            score   = float(fidelity_original)

        return VerificationResult(
            task_id    = task.task_id,
            model_name = response.model_name,
            correct    = correct,
            score      = score,
            details    = {
                "has_bug":           has_bug,
                "code_crashed":      code_crashed,
                "crash_msg":         crash_msg,
                "equiv_intended":    equiv_intended,
                "equiv_original":    equiv_original,
                "changed_circuit":   changed_circuit,
                "fidelity_intended": float(fidelity_intended),
                "fidelity_original": float(fidelity_original),
                "mutation_type":     task.metadata["mutation"]["type"],
            },
        )
