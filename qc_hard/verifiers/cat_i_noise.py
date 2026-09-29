import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult, DifficultyLevel

TOLERANCE_BY_LEVEL = {
    DifficultyLevel.TEXTBOOK: 0.05,
    DifficultyLevel.HOMEWORK: 0.05,
    DifficultyLevel.EXAM: 0.03,
    DifficultyLevel.RESEARCH: 0.02,
    DifficultyLevel.OPEN: 0.02,
}

BANNED_IMPORTS = ["qiskit", "qiskit_aer", "cirq", "pennylane", "NoiseModel",
                  "AerSimulator", "Statevector", "DensityMatrix"]


class NoiseFidelityVerifier(BaseVerifier):
    subtask = "T7_noise_fidelity"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        ground_truth = task.metadata["ground_truth_fidelity"]
        tolerance = TOLERANCE_BY_LEVEL[task.level]

        try:
            for banned in BANNED_IMPORTS:
                if banned in code:
                    return VerificationResult(
                        task_id=task.task_id,
                        model_name=response.model_name,
                        correct=False,
                        score=0.0,
                        details={"error": f"Used banned import/tool '{banned}'. "
                                 f"Must implement with numpy only."},
                    )

            prediction = self._execute_code(code)
            if not isinstance(prediction, (int, float)):
                raise ValueError(f"solve() must return a float, got {type(prediction)}")
            prediction = float(prediction)

            if not (0.0 <= prediction <= 1.0):
                return VerificationResult(
                    task_id=task.task_id,
                    model_name=response.model_name,
                    correct=False,
                    score=0.0,
                    details={"error": f"Prediction {prediction} not in [0, 1]",
                             "prediction": prediction, "ground_truth": ground_truth},
                )

            error = abs(prediction - ground_truth)
            correct = error < tolerance
            score = max(0.0, 1.0 - error / tolerance)

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=float(score),
                details={
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "error": error,
                    "tolerance": tolerance,
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
        namespace = {"np": np, "numpy": np}
        exec(code, namespace)
        if "solve" not in namespace:
            raise ValueError("Code must define a `solve()` function")
        return namespace["solve"]()
