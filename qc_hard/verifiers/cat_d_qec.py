from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult


class SyndromeDecodingVerifier(BaseVerifier):
    subtask = "D1_syndrome_decoding"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        expected = task.metadata["observable_flip"]

        try:
            namespace = {}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            prediction = bool(namespace["solve"]())
            correct = prediction == expected
            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=1.0 if correct else 0.0,
                details={"predicted": prediction, "expected": expected},
            )
        except Exception as e:
            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=False,
                score=0.0,
                details={"error": str(e)[:500]},
            )
