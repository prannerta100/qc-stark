import numpy as np
from qc_hard.verifiers.cat_a_synthesis import StatePrepVerifier
from qc_hard.verifiers.cat_b_debugging import SingleBugVerifier
from qc_hard.verifiers.cat_c_compilation import RoutingVerifier
from qc_hard.verifiers.cat_d_qec import SyndromeDecodingVerifier
from qc_hard.types import TaskInstance, ModelResponse, Category, DifficultyLevel


class TestStatePrepVerifier:
    def test_correct_bell_state(self):
        verifier = StatePrepVerifier()
        bell = [[0.7071067811865476, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7071067811865476, 0.0]]
        task = TaskInstance(
            "t1", Category.A_SYNTHESIS, "A1_state_prep",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={"target_state": bell, "n_qubits": 2},
        )
        code = "from qiskit import QuantumCircuit\ndef solve():\n    qc = QuantumCircuit(2)\n    qc.h(0)\n    qc.cx(0, 1)\n    return qc\n"
        resp = ModelResponse("t1", "mock", code, parsed_code=code)
        result = verifier.verify(task, resp)
        assert result.correct
        assert result.score > 0.99

    def test_wrong_circuit(self):
        verifier = StatePrepVerifier()
        bell = [[0.7071067811865476, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7071067811865476, 0.0]]
        task = TaskInstance(
            "t2", Category.A_SYNTHESIS, "A1_state_prep",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={"target_state": bell, "n_qubits": 2},
        )
        code = "from qiskit import QuantumCircuit\ndef solve():\n    qc = QuantumCircuit(2)\n    qc.x(0)\n    return qc\n"
        resp = ModelResponse("t2", "mock", code, parsed_code=code)
        result = verifier.verify(task, resp)
        assert not result.correct

    def test_invalid_code(self):
        verifier = StatePrepVerifier()
        bell = [[0.7071067811865476, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7071067811865476, 0.0]]
        task = TaskInstance(
            "t3", Category.A_SYNTHESIS, "A1_state_prep",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={"target_state": bell, "n_qubits": 2},
        )
        resp = ModelResponse("t3", "mock", "not valid python }{", parsed_code="}{")
        result = verifier.verify(task, resp)
        assert not result.correct
        assert result.score == 0.0


class TestSingleBugVerifier:
    def test_correct_fix(self):
        verifier = SingleBugVerifier()
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        qc = QuantumCircuit(2)
        qc.h(0)
        u = Operator(qc).data
        task = TaskInstance(
            "t1", Category.B_DEBUGGING, "B1_single_bug",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={
                "correct_unitary_real": [[float(c.real) for c in row] for row in u],
                "correct_unitary_imag": [[float(c.imag) for c in row] for row in u],
                "n_qubits": 2,
            },
        )
        code = "from qiskit import QuantumCircuit\ndef solve():\n    qc = QuantumCircuit(2)\n    qc.h(0)\n    return qc\n"
        resp = ModelResponse("t1", "mock", code, parsed_code=code)
        result = verifier.verify(task, resp)
        assert result.correct

    def test_wrong_fix(self):
        verifier = SingleBugVerifier()
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        qc = QuantumCircuit(2)
        qc.h(0)
        u = Operator(qc).data
        task = TaskInstance(
            "t2", Category.B_DEBUGGING, "B1_single_bug",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={
                "correct_unitary_real": [[float(c.real) for c in row] for row in u],
                "correct_unitary_imag": [[float(c.imag) for c in row] for row in u],
                "n_qubits": 2,
            },
        )
        code = "from qiskit import QuantumCircuit\ndef solve():\n    qc = QuantumCircuit(2)\n    qc.x(0)\n    return qc\n"
        resp = ModelResponse("t2", "mock", code, parsed_code=code)
        result = verifier.verify(task, resp)
        assert not result.correct


class TestSyndromeDecodingVerifier:
    def test_correct_prediction(self):
        verifier = SyndromeDecodingVerifier()
        task = TaskInstance(
            "t1", Category.D_QEC, "D1_syndrome_decoding",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={"observable_flip": True, "syndrome": [1, 0, 1], "code_distance": 3},
        )
        code = "def solve():\n    return True"
        resp = ModelResponse("t1", "mock", code, parsed_code=code)
        result = verifier.verify(task, resp)
        assert result.correct

    def test_wrong_prediction(self):
        verifier = SyndromeDecodingVerifier()
        task = TaskInstance(
            "t2", Category.D_QEC, "D1_syndrome_decoding",
            DifficultyLevel.TEXTBOOK, 1, "",
            metadata={"observable_flip": True, "syndrome": [1, 0, 1], "code_distance": 3},
        )
        code = "def solve():\n    return False"
        resp = ModelResponse("t2", "mock", code, parsed_code=code)
        result = verifier.verify(task, resp)
        assert not result.correct
