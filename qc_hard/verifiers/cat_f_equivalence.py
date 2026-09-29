import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult


class EquivalenceVerifier(BaseVerifier):
    subtask = "T4_equivalence"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        ground_truth = task.metadata["ground_truth"]

        try:
            result_val = self._execute_code(code)
            if not isinstance(result_val, bool):
                result_val = bool(result_val)

            correct = result_val == ground_truth
            score = 1.0 if correct else 0.0

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "predicted": result_val,
                    "ground_truth": ground_truth,
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
