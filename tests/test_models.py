from qc_hard.models.mock import MockModel
from qc_hard.models.base import ModelRegistry
from qc_hard.types import TaskInstance, Category, DifficultyLevel


def test_mock_model_returns_response():
    model = MockModel(name="mock-v1", fixed_response="def solve():\n    pass")
    task = TaskInstance("t1", Category.A_SYNTHESIS, "A1_state_prep", DifficultyLevel.TEXTBOOK, 1, "Prepare |+>")
    resp = model.query(task)
    assert resp.model_name == "mock-v1"
    assert resp.raw_response == "def solve():\n    pass"
    assert resp.task_id == "t1"


def test_model_registry():
    registry = ModelRegistry()
    m = MockModel(name="mock-v1", fixed_response="x")
    registry.register(m)
    assert registry.get("mock-v1") is m
    assert "mock-v1" in registry.list_names()
