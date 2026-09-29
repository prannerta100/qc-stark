from qc_hard.types import TaskInstance, Category, DifficultyLevel, VerificationResult


def test_task_instance_creation():
    t = TaskInstance(
        task_id="A1_seed42_L1",
        category=Category.A_SYNTHESIS,
        subtask="A1_state_prep",
        level=DifficultyLevel.TEXTBOOK,
        seed=42,
        prompt="Prepare the state [0.707, 0.707]",
    )
    assert t.task_id == "A1_seed42_L1"
    assert t.category == Category.A_SYNTHESIS
    assert t.level == 1


def test_verification_result():
    v = VerificationResult(
        task_id="A1_seed42_L1",
        model_name="gpt-4o",
        correct=True,
        score=0.999,
        details={"fidelity": 0.9995},
    )
    assert v.correct
    assert v.score > 0.99
