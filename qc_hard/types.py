from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    A_SYNTHESIS = "A_synthesis"
    B_DEBUGGING = "B_debugging"
    C_COMPILATION = "C_compilation"
    D_QEC = "D_qec"
    E_OPTIMIZATION = "E_optimization"
    F_EQUIVALENCE = "F_equivalence"
    G_TROTTERIZATION = "G_trotterization"
    H_ORACLE = "H_oracle"
    I_NOISE = "I_noise"
    J_REVERSE = "J_reverse"
    K_MITIGATION = "K_mitigation"


class DifficultyLevel(int, Enum):
    TEXTBOOK = 1
    HOMEWORK = 2
    EXAM = 3
    RESEARCH = 4
    OPEN = 5


@dataclass
class TaskInstance:
    task_id: str
    category: Category
    subtask: str
    level: DifficultyLevel
    seed: int
    prompt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    task_id: str
    model_name: str
    raw_response: str
    parsed_code: str | None = None
    latency_ms: float = 0.0
    token_count: int = 0


@dataclass
class VerificationResult:
    task_id: str
    model_name: str
    correct: bool
    score: float
    details: dict = field(default_factory=dict)
