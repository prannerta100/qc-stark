from abc import ABC, abstractmethod
from qc_hard.types import TaskInstance, Category, DifficultyLevel


class BaseGenerator(ABC):
    category: Category
    subtask: str

    @abstractmethod
    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        ...


class GeneratorRegistry:
    def __init__(self):
        self._generators: dict[str, BaseGenerator] = {}

    def register(self, generator: BaseGenerator):
        self._generators[generator.subtask] = generator

    def get(self, subtask: str) -> BaseGenerator:
        return self._generators[subtask]

    def list_by_category(self, category: Category) -> list[BaseGenerator]:
        return [g for g in self._generators.values() if g.category == category]

    def all(self) -> list[BaseGenerator]:
        return list(self._generators.values())
