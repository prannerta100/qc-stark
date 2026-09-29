import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

EQUIVALENCE_THRESHOLD = 0.999


class OptimizationVerifier(BaseVerifier):
    subtask = "T3_optimization"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        u_real = np.array(task.metadata["original_unitary_real"])
        u_imag = np.array(task.metadata["original_unitary_imag"])
        target_unitary = u_real + 1j * u_imag
        input_gates = task.metadata["inflated_gate_count"]
        optimal_gates = task.metadata["original_gate_count"]

        try:
            circuit = self._execute_code(code)
            from qiskit.quantum_info import Operator
            op = Operator(circuit)
            U = np.array(op.data)
            dim = target_unitary.shape[0]

            # Check unitary equivalence
            fidelity = abs(np.trace(target_unitary.conj().T @ U)) / dim
            equivalent = fidelity >= EQUIVALENCE_THRESHOLD

            # Check gate count reduction
            output_gates = circuit.size()
            fewer_gates = output_gates < input_gates

            correct = equivalent and fewer_gates

            # Score: reduction ratio
            if input_gates > optimal_gates:
                raw_score = (input_gates - output_gates) / (input_gates - optimal_gates)
            else:
                raw_score = 1.0 if output_gates <= optimal_gates else 0.0
            score = float(np.clip(raw_score, 0.0, 1.0))

            if not equivalent:
                score = 0.0

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "operator_fidelity": float(fidelity),
                    "equivalent": equivalent,
                    "input_gates": input_gates,
                    "output_gates": output_gates,
                    "optimal_gates": optimal_gates,
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

    def _execute_code(self, code: str):
        namespace = {}
        exec(code, namespace)
        if "solve" not in namespace:
            raise ValueError("Code must define a `solve()` function")
        return namespace["solve"]()
