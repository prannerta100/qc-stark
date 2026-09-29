"""Verifier for T10: Error Mitigation via ZNE."""
import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult


class ErrorMitigationVerifier(BaseVerifier):
    subtask = "T10_mitigation"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        ideal_val = task.metadata["ideal_value"]
        noisy_val = task.metadata["noisy_value"]

        try:
            namespace = {"np": np, "numpy": np}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            result = namespace["solve"]()

            if not isinstance(result, (int, float)):
                raise ValueError(f"solve() must return a float, got {type(result)}")

            mitigated_val = float(result)

            error_mitigated = abs(mitigated_val - ideal_val)
            error_noisy = abs(noisy_val - ideal_val)

            if error_noisy < 1e-10:
                improvement_ratio = 1.0 if error_mitigated < 1e-10 else 0.0
            else:
                improvement_ratio = max(0.0, 1.0 - error_mitigated / error_noisy)

            tolerance = 0.15
            correct = error_mitigated < tolerance

            score = 0.0
            if correct:
                score = min(1.0, 0.5 + 0.5 * improvement_ratio)
            else:
                score = max(0.0, improvement_ratio * 0.5)

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "mitigated_value": mitigated_val,
                    "ideal_value": ideal_val,
                    "noisy_value": noisy_val,
                    "error_mitigated": error_mitigated,
                    "error_noisy": error_noisy,
                    "improvement_ratio": improvement_ratio,
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
