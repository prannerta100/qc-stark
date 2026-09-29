from qc_hard.evaluation.runner import EvaluationRunner
from qc_hard.generators.base import GeneratorRegistry, BaseGenerator
from qc_hard.verifiers.base import VerifierRegistry, BaseVerifier
from qc_hard.models.base import ModelRegistry
from qc_hard.models.mock import MockModel
from qc_hard.types import (
    TaskInstance, ModelResponse, VerificationResult,
    Category, DifficultyLevel,
)


class StubGenerator(BaseGenerator):
    category = Category.A_SYNTHESIS
    subtask = "stub"

    def generate(self, seed, level):
        return TaskInstance(f"stub_{seed}_L{level}", self.category, self.subtask, level, seed, "prompt")


class StubVerifier(BaseVerifier):
    subtask = "stub"

    def verify(self, task, response):
        return VerificationResult(task.task_id, response.model_name, True, 1.0)


def test_runner_produces_results():
    gen_reg = GeneratorRegistry()
    gen_reg.register(StubGenerator())
    ver_reg = VerifierRegistry()
    ver_reg.register(StubVerifier())
    model_reg = ModelRegistry()
    model_reg.register(MockModel("mock", "def solve(): pass"))

    runner = EvaluationRunner(gen_reg, ver_reg, model_reg)
    results = runner.run(
        subtasks=["stub"],
        models=["mock"],
        seeds=[1, 2, 3],
        levels=[DifficultyLevel.TEXTBOOK],
    )
    assert len(results) == 3
    assert all(r.correct for r in results)


def test_runner_multiple_models():
    gen_reg = GeneratorRegistry()
    gen_reg.register(StubGenerator())
    ver_reg = VerifierRegistry()
    ver_reg.register(StubVerifier())
    model_reg = ModelRegistry()
    model_reg.register(MockModel("m1", "code"))
    model_reg.register(MockModel("m2", "code"))

    runner = EvaluationRunner(gen_reg, ver_reg, model_reg)
    results = runner.run(
        subtasks=["stub"],
        models=["m1", "m2"],
        seeds=[1],
        levels=[DifficultyLevel.TEXTBOOK],
    )
    assert len(results) == 2
