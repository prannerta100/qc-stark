"""Verifier for T11: Circuit Reverse Engineering."""
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult


ALGO_ALIASES = {
    "qft": ["qft", "quantum fourier transform", "quantum_fourier_transform"],
    "inverse_qft": ["inverse_qft", "inverse qft", "iqft", "inverse quantum fourier transform"],
    "ghz": ["ghz", "ghz state", "ghz_state", "ghz state preparation"],
    "bell": ["bell", "bell state", "bell_state", "bell pair", "epr"],
    "grover_diffusion": ["grover_diffusion", "grover diffusion", "diffusion operator",
                         "grover", "amplitude amplification", "inversion about mean"],
    "swap_test": ["swap_test", "swap test", "fredkin"],
    "teleportation": ["teleportation", "quantum teleportation", "quantum_teleportation"],
    "phase_estimation_core": ["phase_estimation_core", "phase_estimation", "qpe",
                              "quantum phase estimation", "phase estimation"],
    "bernstein_vazirani": ["bernstein_vazirani", "bernstein-vazirani", "bv",
                           "bernstein vazirani"],
    "deutsch_jozsa": ["deutsch_jozsa", "deutsch-jozsa", "dj", "deutsch jozsa"],
}


def _normalize_algo_name(name: str) -> str | None:
    name_lower = name.strip().lower().replace("_", " ").replace("-", " ")
    for canonical, aliases in ALGO_ALIASES.items():
        for alias in aliases:
            if alias in name_lower or name_lower in alias:
                return canonical
    return None


class ReverseEngineeringVerifier(BaseVerifier):
    subtask = "T11_reverse"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        ground_truth = task.metadata["algorithm"]

        try:
            namespace = {}
            exec(code, namespace)
            if "solve" not in namespace:
                raise ValueError("Code must define a `solve()` function")
            result = namespace["solve"]()

            if not isinstance(result, dict):
                raise ValueError(f"solve() must return a dict, got {type(result)}")

            predicted_algo = str(result.get("algorithm", "")).strip()
            predicted_output = str(result.get("output", "")).strip()

            normalized = _normalize_algo_name(predicted_algo)
            algo_correct = (normalized == ground_truth)

            score = 0.0
            if algo_correct:
                score = 1.0
            else:
                if normalized is not None:
                    score = 0.0
                else:
                    score = 0.0

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=algo_correct,
                score=score,
                details={
                    "predicted_algorithm": predicted_algo,
                    "normalized_prediction": normalized,
                    "ground_truth": ground_truth,
                    "predicted_output": predicted_output[:200],
                    "expected_output": task.metadata["expected_output"],
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
