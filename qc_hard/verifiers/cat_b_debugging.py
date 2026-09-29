import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

OPERATOR_FIDELITY_THRESHOLD = 0.999


class SingleBugVerifier(BaseVerifier):
    subtask = "B1_single_bug"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        u_real = np.array(task.metadata["correct_unitary_real"])
        u_imag = np.array(task.metadata["correct_unitary_imag"])
        target_unitary = u_real + 1j * u_imag

        try:
            circuit = self._execute_code(code)
            fidelity = self._operator_fidelity(circuit, target_unitary)
            correct = fidelity >= OPERATOR_FIDELITY_THRESHOLD
            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=float(fidelity),
                details={"operator_fidelity": float(fidelity)},
            )
        except Exception as e:
            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=False,
                score=0.0,
                details={"error": str(e)[:500]},
            )

    def _execute_code(self, code: str):
        namespace = {}
        exec(code, namespace)
        if "solve" not in namespace:
            raise ValueError("Code must define a `solve()` function")
        return namespace["solve"]()

    def _operator_fidelity(self, circuit, target: np.ndarray) -> float:
        from qiskit.quantum_info import Operator
        op = Operator(circuit)
        U = np.array(op.data)
        dim = target.shape[0]
        fidelity = abs(np.trace(target.conj().T @ U)) / dim
        return float(fidelity)
