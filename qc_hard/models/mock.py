from qc_hard.models.base import BaseModel
from qc_hard.types import TaskInstance, ModelResponse


class MockModel(BaseModel):
    def __init__(self, name: str = "mock", fixed_response: str = ""):
        self.name = name
        self._fixed_response = fixed_response

    def query(self, task: TaskInstance) -> ModelResponse:
        return ModelResponse(
            task_id=task.task_id,
            model_name=self.name,
            raw_response=self._fixed_response,
            parsed_code=self._fixed_response,
            latency_ms=0.0,
            token_count=len(self._fixed_response.split()),
        )
