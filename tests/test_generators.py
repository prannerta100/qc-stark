import numpy as np
from qc_hard.generators.cat_a_synthesis import StatePrepGenerator
from qc_hard.generators.cat_b_debugging import SingleBugGenerator
from qc_hard.generators.cat_c_compilation import RoutingGenerator
from qc_hard.generators.cat_d_qec import SyndromeDecodingGenerator
from qc_hard.types import DifficultyLevel


class TestStatePrepGenerator:
    def test_level1_produces_2qubit(self):
        gen = StatePrepGenerator()
        task = gen.generate(seed=42, level=DifficultyLevel.TEXTBOOK)
        assert task.subtask == "A1_state_prep"
        state = np.array(task.metadata["target_state"])
        assert len(state) == 4  # 2 qubits

    def test_level3_produces_4qubit(self):
        gen = StatePrepGenerator()
        task = gen.generate(seed=42, level=DifficultyLevel.EXAM)
        state = np.array(task.metadata["target_state"])
        assert len(state) == 16

    def test_deterministic(self):
        gen = StatePrepGenerator()
        t1 = gen.generate(seed=99, level=DifficultyLevel.HOMEWORK)
        t2 = gen.generate(seed=99, level=DifficultyLevel.HOMEWORK)
        assert t1.metadata["target_state"] == t2.metadata["target_state"]

    def test_different_seeds_differ(self):
        gen = StatePrepGenerator()
        t1 = gen.generate(seed=1, level=DifficultyLevel.TEXTBOOK)
        t2 = gen.generate(seed=2, level=DifficultyLevel.TEXTBOOK)
        assert t1.metadata["target_state"] != t2.metadata["target_state"]

    def test_state_is_normalized(self):
        gen = StatePrepGenerator()
        task = gen.generate(seed=7, level=DifficultyLevel.HOMEWORK)
        state = np.array([complex(r, i) for r, i in task.metadata["target_state"]])
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10


class TestSingleBugGenerator:
    def test_produces_buggy_circuit(self):
        gen = SingleBugGenerator()
        task = gen.generate(seed=42, level=DifficultyLevel.TEXTBOOK)
        assert task.subtask == "B1_single_bug"
        assert "buggy_circuit_qasm" in task.metadata
        assert "correct_circuit_qasm" in task.metadata
        assert "mutation" in task.metadata
        assert task.metadata["buggy_circuit_qasm"] != task.metadata["correct_circuit_qasm"]

    def test_deterministic(self):
        gen = SingleBugGenerator()
        t1 = gen.generate(seed=7, level=DifficultyLevel.HOMEWORK)
        t2 = gen.generate(seed=7, level=DifficultyLevel.HOMEWORK)
        assert t1.metadata["buggy_circuit_qasm"] == t2.metadata["buggy_circuit_qasm"]

    def test_scaling(self):
        gen = SingleBugGenerator()
        t1 = gen.generate(seed=1, level=DifficultyLevel.TEXTBOOK)
        t3 = gen.generate(seed=1, level=DifficultyLevel.EXAM)
        assert len(t3.metadata["buggy_circuit_qasm"]) > len(t1.metadata["buggy_circuit_qasm"])


class TestRoutingGenerator:
    def test_produces_task(self):
        gen = RoutingGenerator()
        task = gen.generate(seed=42, level=DifficultyLevel.TEXTBOOK)
        assert task.subtask == "C1_routing"
        assert "circuit_qasm" in task.metadata
        assert "coupling_map" in task.metadata

    def test_deterministic(self):
        gen = RoutingGenerator()
        t1 = gen.generate(seed=5, level=DifficultyLevel.EXAM)
        t2 = gen.generate(seed=5, level=DifficultyLevel.EXAM)
        assert t1.metadata["circuit_qasm"] == t2.metadata["circuit_qasm"]


class TestSyndromeDecodingGenerator:
    def test_produces_task(self):
        gen = SyndromeDecodingGenerator()
        task = gen.generate(seed=42, level=DifficultyLevel.TEXTBOOK)
        assert task.subtask == "D1_syndrome_decoding"
        assert "syndrome" in task.metadata
        assert "observable_flip" in task.metadata
        assert task.metadata["code_distance"] == 3

    def test_level3_larger_distance(self):
        gen = SyndromeDecodingGenerator()
        task = gen.generate(seed=42, level=DifficultyLevel.EXAM)
        assert task.metadata["code_distance"] == 5

    def test_deterministic(self):
        gen = SyndromeDecodingGenerator()
        t1 = gen.generate(seed=10, level=DifficultyLevel.TEXTBOOK)
        t2 = gen.generate(seed=10, level=DifficultyLevel.TEXTBOOK)
        assert t1.metadata["syndrome"] == t2.metadata["syndrome"]
