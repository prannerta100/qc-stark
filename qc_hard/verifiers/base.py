from abc import ABC, abstractmethod
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult


class BaseVerifier(ABC):
    subtask: str

    @abstractmethod
    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        ...


class VerifierRegistry:
    def __init__(self):
        self._verifiers: dict[str, BaseVerifier] = {}

    def register(self, verifier: BaseVerifier):
        self._verifiers[verifier.subtask] = verifier

    def get(self, subtask: str) -> BaseVerifier:
        return self._verifiers[subtask]

    def all(self) -> list[BaseVerifier]:
        return list(self._verifiers.values())
