import warnings
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import loads as qasm_loads
from qiskit.quantum_info import Operator
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult

FIDELITY_THRESHOLD = 0.99


class RoutingVerifier(BaseVerifier):
    subtask = "C1_routing"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        coupling_map = [tuple(e) for e in task.metadata["coupling_map"]]
        n_logical = task.metadata["n_qubits"]
        n_physical = task.metadata["n_physical"]

        try:
            routed_circuit = self._execute_code(code)

            if routed_circuit.num_qubits != n_physical:
                return VerificationResult(
                    task_id=task.task_id,
                    model_name=response.model_name,
                    correct=False,
                    score=0.0,
                    details={"error": f"Expected {n_physical} qubits, got {routed_circuit.num_qubits}"},
                )

            connectivity_valid = self._check_connectivity(routed_circuit, coupling_map)
            if not connectivity_valid:
                return VerificationResult(
                    task_id=task.task_id,
                    model_name=response.model_name,
                    correct=False,
                    score=0.0,
                    details={"connectivity_valid": False},
                )

            functionally_equiv = self._check_functional_equivalence(
                task.metadata["circuit_qasm"], routed_circuit, n_logical, n_physical
            )

            swap_count = sum(
                1 for inst in routed_circuit.data if inst.operation.name == "swap"
            )
            orig_circ = qasm_loads(task.metadata["circuit_qasm"])
            n_original_2q = sum(1 for inst in orig_circ.data if inst.operation.num_qubits == 2)

            correct = connectivity_valid and functionally_equiv
            score = 0.0
            if correct:
                if n_original_2q > 0:
                    swap_ratio = swap_count / n_original_2q
                    score = max(0.0, 1.0 - swap_ratio * 0.3)
                else:
                    score = 1.0

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=score,
                details={
                    "connectivity_valid": connectivity_valid,
                    "functionally_equivalent": functionally_equiv,
                    "swap_count": swap_count,
                    "total_gates": len(routed_circuit.data),
                    "original_2q_gates": n_original_2q,
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

    def _execute_code(self, code: str) -> QuantumCircuit:
        namespace = {}
        exec(code, namespace)
        if "solve" not in namespace:
            raise ValueError("Code must define a `solve()` function")
        return namespace["solve"]()

    def _check_connectivity(self, circuit: QuantumCircuit, coupling_map: list[tuple]) -> bool:
        adjacent = set(coupling_map) | {(b, a) for a, b in coupling_map}
        for inst in circuit.data:
            if inst.operation.num_qubits == 2:
                q0 = inst.qubits[0]._index
                q1 = inst.qubits[1]._index
                if (q0, q1) not in adjacent:
                    return False
        return True

    def _check_functional_equivalence(self, original_qasm: str,
                                       routed_circuit: QuantumCircuit,
                                       n_logical: int, n_physical: int) -> bool:
        """Check that the routed circuit implements the same operation on the
        logical subspace (qubits 0..n_logical-1) as the original circuit.
        Identity initial layout assumed: logical qubit i -> physical qubit i."""
        orig_circ = qasm_loads(original_qasm)

        padded_orig = QuantumCircuit(n_physical)
        for inst in orig_circ.data:
            qubit_indices = [q._index for q in inst.qubits]
            padded_orig.append(inst.operation, qubit_indices)
        padded_unitary = Operator(padded_orig).data

        routed_unitary = Operator(routed_circuit).data

        dim = 2 ** n_physical
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fidelity = abs(np.trace(padded_unitary.conj().T @ routed_unitary)) / dim
        if np.isnan(fidelity):
            return False
        return fidelity >= FIDELITY_THRESHOLD
