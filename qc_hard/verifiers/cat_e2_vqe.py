"""Verifier for E2: VQE ground state energy."""
import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

TOLERANCE = 0.05  # eV-scale; tight enough to exclude random guessing


class VQEVerifier(BaseVerifier):
    subtask = "E2_vqe"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        true_energy = task.metadata["ground_energy"]

        try:
            namespace = {"np": np, "numpy": np}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            result = namespace["solve"]()

            if not isinstance(result, (int, float, np.floating)):
                raise ValueError(f"solve() must return a float, got {type(result)}")

            predicted = float(result)
            error = abs(predicted - true_energy)
            correct = error < TOLERANCE

            # Graded score: 1.0 at exact, 0.5 at tolerance boundary, 0 beyond 2x tolerance
            score = float(np.clip(1.0 - error / (2 * TOLERANCE), 0.0, 1.0))
            if not correct:
                score = min(score, 0.49)

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "predicted": predicted,
                    "true_energy": true_energy,
                    "error": error,
                    "tolerance": TOLERANCE,
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
