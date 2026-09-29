"""Verifier for F2: T-gate count minimization."""
import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

FIDELITY_THRESHOLD = 0.999


class TCountVerifier(BaseVerifier):
    subtask = "F2_tcount"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        u_real = np.array(task.metadata["unitary_real"])
        u_imag = np.array(task.metadata["unitary_imag"])
        target = u_real + 1j * u_imag
        inflated_t = task.metadata["inflated_t_count"]
        optimal_t = task.metadata["optimal_t_count"]

        try:
            namespace = {}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            circuit = namespace["solve"]()

            from qiskit.quantum_info import Operator
            U = np.array(Operator(circuit).data)
            dim = target.shape[0]

            fidelity = abs(np.trace(target.conj().T @ U)) / dim
            equivalent = fidelity >= FIDELITY_THRESHOLD

            output_t = sum(1 for inst in circuit.data if inst.operation.name in ("t", "tdg"))
            fewer = output_t < inflated_t
            correct = equivalent and fewer

            if inflated_t > optimal_t:
                raw = (inflated_t - output_t) / (inflated_t - optimal_t)
            else:
                raw = 1.0 if output_t <= optimal_t else 0.0
            score = float(np.clip(raw, 0.0, 1.0)) if equivalent else 0.0

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "fidelity": float(fidelity),
                    "equivalent": equivalent,
                    "inflated_t": inflated_t,
                    "output_t": output_t,
                    "optimal_t": optimal_t,
                },
            )
        except Exception as e:
            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=False,
                score=0.0,
                details={"error": str(e)[:500]},
            )
