"""Verifier for G2: Trotter error estimation."""
import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

RELATIVE_TOLERANCE = 0.10  # 10% relative error


class TrotterErrorVerifier(BaseVerifier):
    subtask = "G2_trotter_error"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        true_error = task.metadata["ground_truth_error"]

        try:
            namespace = {"np": np, "numpy": np}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            result = namespace["solve"]()

            if not isinstance(result, (int, float, np.floating)):
                raise ValueError(f"solve() must return a float, got {type(result)}")

            predicted = float(result)

            if true_error < 1e-12:
                # Commuting Hamiltonian case: error is ~0, accept if predicted < 1e-6
                rel_err = abs(predicted) if abs(predicted) >= 1e-6 else 0.0
                correct = abs(predicted) < 1e-6
            else:
                rel_err = abs(predicted - true_error) / true_error
                correct = rel_err < RELATIVE_TOLERANCE

            score = float(np.clip(1.0 - rel_err / (2 * RELATIVE_TOLERANCE), 0.0, 1.0))
            if not correct:
                score = min(score, 0.49)

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "predicted": predicted,
                    "true_error": true_error,
                    "relative_error": float(rel_err),
                    "tolerance": RELATIVE_TOLERANCE,
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
