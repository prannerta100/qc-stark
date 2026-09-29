from abc import ABC, abstractmethod
from qc_hard.types import TaskInstance, ModelResponse


class BaseModel(ABC):
    name: str

    @abstractmethod
    def query(self, task: TaskInstance) -> ModelResponse:
        ...


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, BaseModel] = {}

    def register(self, model: BaseModel):
        self._models[model.name] = model

    def get(self, name: str) -> BaseModel:
        return self._models[name]

    def list_names(self) -> list[str]:
        return list(self._models.keys())
