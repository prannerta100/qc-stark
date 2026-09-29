import logging
from qc_hard.generators.base import GeneratorRegistry
from qc_hard.verifiers.base import VerifierRegistry
from qc_hard.models.base import ModelRegistry
from qc_hard.types import DifficultyLevel, VerificationResult

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        generators: GeneratorRegistry,
        verifiers: VerifierRegistry,
        models: ModelRegistry,
    ):
        self._generators = generators
        self._verifiers = verifiers
        self._models = models

    def run(
        self,
        subtasks: list[str],
        models: list[str],
        seeds: list[int],
        levels: list[DifficultyLevel],
    ) -> list[VerificationResult]:
        results = []
        total = len(subtasks) * len(models) * len(seeds) * len(levels)
        done = 0

        for subtask in subtasks:
            generator = self._generators.get(subtask)
            verifier = self._verifiers.get(subtask)

            for level in levels:
                for seed in seeds:
                    task = generator.generate(seed=seed, level=level)

                    for model_name in models:
                        model = self._models.get(model_name)
                        done += 1
                        logger.info(f"[{done}/{total}] {subtask} L{level} seed={seed} model={model_name}")

                        try:
                            response = model.query(task)
                            result = verifier.verify(task, response)
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            result = VerificationResult(
                                task_id=task.task_id,
                                model_name=model_name,
                                correct=False,
                                score=0.0,
                                details={"error": str(e)[:500]},
                            )
                        results.append(result)

        return results
