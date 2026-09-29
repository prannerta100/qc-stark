import numpy as np
from qc_hard.verifiers.base import BaseVerifier
from qc_hard.types import TaskInstance, ModelResponse, VerificationResult


class OracleVerifier(BaseVerifier):
    subtask = "T6_oracle"

    def verify(self, task: TaskInstance, response: ModelResponse) -> VerificationResult:
        code = response.parsed_code or response.raw_response
        n_input = task.metadata["n_input"]
        truth_table = task.metadata["truth_table"]
        max_ancillae = task.metadata["max_ancillae"]

        try:
            circuit = self._execute_code(code)
            n_total = circuit.num_qubits

            # Validate circuit structure
            if n_total < n_input + 1:
                return VerificationResult(
                    task_id=task.task_id,
                    model_name=response.model_name,
                    correct=False,
                    score=0.0,
                    details={"error": f"Circuit has {n_total} qubits, need at least {n_input + 1}"},
                )

            # Check ancilla constraint
            n_ancillae = n_total - n_input - 1
            if max_ancillae is not None and n_ancillae > max_ancillae:
                return VerificationResult(
                    task_id=task.task_id,
                    model_name=response.model_name,
                    correct=False,
                    score=0.0,
                    details={"error": f"Too many ancillae: {n_ancillae} > {max_ancillae}"},
                )

            # Check correctness for each input
            from qiskit.quantum_info import Statevector
            n_inputs = 2 ** n_input
            correct_count = 0

            for x in range(n_inputs):
                # Prepare input state: |x>|0...0>|0>
                # x on first n_input qubits, zeros on ancillae, 0 on output
                initial_state = [0] * (2 ** n_total)
                # In Qiskit, qubit ordering is little-endian
                # State index for |x>|0...0> with x on first n_input qubits
                state_idx = x  # x on least significant (first) qubits
                initial_state[state_idx] = 1
                sv = Statevector(initial_state)

                # Evolve through circuit
                sv_out = sv.evolve(circuit)
                probs = sv_out.probabilities()

                # Find the state with highest probability (should be deterministic)
                max_idx = int(np.argmax(probs))
                if probs[max_idx] < 0.999:
                    # Non-deterministic output - oracle is wrong
                    continue

                # Extract output qubit value (last qubit = most significant bit)
                output_bit = (max_idx >> (n_total - 1)) & 1

                # Extract input qubits (first n_input qubits = least significant)
                input_bits = max_idx & ((1 << n_input) - 1)

                # Extract ancilla qubits (middle)
                ancilla_bits = (max_idx >> n_input) & ((1 << n_ancillae) - 1)

                # Check: input unchanged, ancillae clean, output = f(x)
                input_ok = (input_bits == x)
                ancilla_ok = (ancilla_bits == 0) if n_ancillae > 0 else True
                output_ok = (output_bit == truth_table[x])

                if input_ok and ancilla_ok and output_ok:
                    correct_count += 1

            score = correct_count / n_inputs
            correct = (correct_count == n_inputs)

            return VerificationResult(
                task_id=task.task_id,
                model_name=response.model_name,
                correct=correct,
                score=float(score),
                details={
                    "correct_inputs": correct_count,
                    "total_inputs": n_inputs,
                    "n_ancillae": n_ancillae,
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
