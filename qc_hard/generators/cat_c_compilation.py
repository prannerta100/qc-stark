import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_PARAMS = {
    # Difficulty = (n_logical / n_physical) * n_2q_gates / edges — higher = harder
    # L1: 3 qubits, linear-5, 2 gates — very few constraints
    DifficultyLevel.TEXTBOOK: {"n_logical": 3, "n_physical": 5, "topology": "linear", "n_2q_gates": 2},
    # L2: 4 qubits, linear-6, 5 gates — tighter constraints on linear chain
    DifficultyLevel.HOMEWORK: {"n_logical": 4, "n_physical": 6, "topology": "linear", "n_2q_gates": 5},
    # L3: 5 qubits, linear-7, 8 gates — dense on linear (hardest topology)
    DifficultyLevel.EXAM: {"n_logical": 5, "n_physical": 7, "topology": "linear", "n_2q_gates": 8},
    # L4: 7 qubits, linear-9, 12 gates — very constrained
    DifficultyLevel.RESEARCH: {"n_logical": 7, "n_physical": 9, "topology": "linear", "n_2q_gates": 12},
    # L5: 10 qubits, linear-12, 18 gates — extreme
    DifficultyLevel.OPEN: {"n_logical": 10, "n_physical": 12, "topology": "linear", "n_2q_gates": 18},
}


def build_topology(name: str, n: int) -> list[tuple[int, int]]:
    if name == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    elif name == "grid":
        side = int(np.ceil(np.sqrt(n)))
        edges = []
        for i in range(n):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < n:
                edges.append((i, i + 1))
            if i + side < n:
                edges.append((i, i + side))
        return edges
    elif name == "heavy_hex":
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))
            if i % 3 == 0 and i + 2 < n:
                edges.append((i, i + 2))
        return edges
    return [(i, i + 1) for i in range(n - 1)]


class RoutingGenerator(BaseGenerator):
    category = Category.C_COMPILATION
    subtask = "C1_routing"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        rng = np.random.default_rng(seed + level * 10000)
        params = LEVEL_TO_PARAMS[level]
        n_logical = params["n_logical"]
        n_physical = params["n_physical"]
        n_2q_gates = params["n_2q_gates"]
        topology = params["topology"]

        coupling_map = build_topology(topology, n_physical)
        circuit = self._gen_circuit(rng, n_logical, n_2q_gates)
        qasm = qasm_dumps(circuit)

        adjacent_set = set(coupling_map) | {(b, a) for a, b in coupling_map}
        requires_routing = any(
            (inst.qubits[0]._index, inst.qubits[1]._index) not in adjacent_set
            for inst in circuit.data
            if inst.operation.num_qubits == 2
        )

        prompt = self._build_prompt(qasm, coupling_map, n_logical, n_physical)

        return TaskInstance(
            task_id=f"C1_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "circuit_qasm": qasm,
                "coupling_map": coupling_map,
                "n_qubits": n_logical,
                "n_physical": n_physical,
                "topology": topology,
                "requires_routing": requires_routing,
            },
        )

    def _gen_circuit(self, rng, n_qubits, n_2q_gates):
        qc = QuantumCircuit(n_qubits)
        for _ in range(n_2q_gates):
            q0, q1 = rng.choice(n_qubits, size=2, replace=False).tolist()
            qc.cx(int(q0), int(q1))
            if rng.random() < 0.4:
                qubit = int(rng.integers(0, n_qubits))
                angle = float(rng.uniform(0, 2 * np.pi))
                qc.rz(angle, qubit)
        return qc

    def _build_prompt(self, qasm, coupling_map, n_logical, n_physical):
        edges_str = ", ".join(f"({a},{b})" for a, b in coupling_map)
        return (
            f"The following quantum circuit operates on {n_logical} logical qubits:\n\n"
            f"```\n{qasm}\n```\n\n"
            f"Map this circuit to a device with {n_physical} physical qubits and connectivity:\n"
            f"[{edges_str}]\n\n"
            f"Requirements:\n"
            f"- Insert SWAP gates so ALL two-qubit gates act on connected qubit pairs.\n"
            f"- The mapped circuit must be functionally equivalent to the original.\n"
            f"- Minimize the number of inserted SWAPs.\n"
            f"- Do NOT use qiskit.transpile() or any PassManager. Build the routed circuit "
            f"manually using QuantumCircuit methods (cx, swap, etc.).\n"
            f"- The output circuit must use {n_physical} qubits. Map logical qubit i to "
            f"physical qubit i (identity initial layout).\n\n"
            f"Write a Qiskit function `solve()` that returns the routed QuantumCircuit "
            f"on {n_physical} qubits. Use only cx and swap gates for 2-qubit operations.\n"
        )
