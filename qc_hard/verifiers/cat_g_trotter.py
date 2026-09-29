import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

FIDELITY_THRESHOLD = 0.99
ALLOWED_GATES = {"rx", "ry", "rz", "cx", "h", "x", "y", "z", "s", "t", "sdg", "tdg",
                 "barrier", "measure", "id", "p", "u1", "u2", "u3", "sx", "sxdg",
                 "cz", "swap", "ccx", "crx", "cry", "crz"}


class TrotterVerifier(BaseVerifier):
    subtask = "T5_trotter"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        u_real = np.array(task.metadata["exact_unitary_real"])
        u_imag = np.array(task.metadata["exact_unitary_imag"])
        exact_unitary = u_real + 1j * u_imag

        try:
            circuit = self._execute_code(code)

            # Validate gate set — reject UnitaryGate, custom gates, etc.
            invalid_gates = self._check_gate_set(circuit)
            if invalid_gates:
                return VerificationResult(
                    task_id=task.task_id,
                    model_name=response.model_name,
                    correct=False,
                    score=0.0,
                    details={"error": f"Used non-standard gates: {invalid_gates}. "
                             f"Must use only standard gates (rx, ry, rz, cx, h, etc.)"},
                )

            from qiskit.quantum_info import Operator
            op = Operator(circuit)
            U = np.array(op.data)
            dim = exact_unitary.shape[0]

            fidelity = abs(np.trace(exact_unitary.conj().T @ U)) / dim
            correct = fidelity >= FIDELITY_THRESHOLD

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=float(fidelity),
                details={
                    "operator_fidelity": float(fidelity),
                    "n_qubits": task.metadata["n_qubits"],
                    "gate_count": circuit.size(),
                },
            )
        except Exception as e:
            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=False,
                score=0.0,
                details={"error": str(e)[:500]},
            )

    def _execute_code(self, code: str):
        namespace = {}
        exec(code, namespace)
        if "solve" not in namespace:
            raise ValueError("Code must define a `solve()` function")
        return namespace["solve"]()

    def _check_gate_set(self, circuit) -> list[str]:
        """Return list of invalid gate names found in the circuit."""
        invalid = []
        for inst in circuit.data:
            name = inst.operation.name.lower()
            if name not in ALLOWED_GATES:
                if name not in invalid:
                    invalid.append(name)
        return invalid
