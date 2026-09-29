import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

FIDELITY_THRESHOLD = 0.999


class StatePrepVerifier(BaseVerifier):
    subtask = "A1_state_prep"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        target = np.array([complex(r, i) for r, i in task.metadata["target_state"]])

        try:
            # At Level 3+, reject code that uses initialize/prepare_state
            if task.level.value >= 3:
                banned = ["initialize", "prepare_state", "isometry"]
                for b in banned:
                    if b in code:
                        return VerificationResult(
                            task_id=task.task_id,
                            model_name=response.model_name,
                            correct=False,
                            score=0.0,
                            details={"error": f"Used banned method '{b}' at Level {task.level.value}"},
                        )

            circuit = self._execute_code(code)
            fidelity = self._compute_fidelity(circuit, target)
            correct = fidelity >= FIDELITY_THRESHOLD

            details = {"fidelity": float(fidelity)}
            # Report gate count for efficiency analysis
            if circuit is not None:
                details["gate_count"] = len(circuit.data)
                details["depth"] = circuit.depth()

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=float(fidelity),
                details=details,
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

    def _compute_fidelity(self, circuit, target: np.ndarray) -> float:
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(circuit)
        state = np.array(sv.data)
        fidelity = abs(np.dot(np.conj(target), state)) ** 2
        return float(fidelity)
