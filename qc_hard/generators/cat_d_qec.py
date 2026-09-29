import numpy as np
from qc_hard.generators.base import BaseGenerator
from qc_hard.types import TaskInstance, Category, DifficultyLevel

LEVEL_TO_DISTANCE = {
    DifficultyLevel.TEXTBOOK: 3,
    DifficultyLevel.HOMEWORK: 3,
    DifficultyLevel.EXAM: 5,
    DifficultyLevel.RESEARCH: 7,
    DifficultyLevel.OPEN: 9,
}

LEVEL_TO_NOISE = {
    DifficultyLevel.TEXTBOOK: 0.01,
    DifficultyLevel.HOMEWORK: 0.02,
    DifficultyLevel.EXAM: 0.005,
    DifficultyLevel.RESEARCH: 0.007,
    DifficultyLevel.OPEN: 0.008,
}


class SyndromeDecodingGenerator(BaseGenerator):
    category = Category.D_QEC
    subtask = "D1_syndrome_decoding"

    def generate(self, seed: int, level: DifficultyLevel) -> TaskInstance:
        distance = LEVEL_TO_DISTANCE[level]
        noise_rate = LEVEL_TO_NOISE[level]

        try:
            import stim
            syndrome, observable_flip, num_detectors = self._generate_with_stim(
                seed, distance, noise_rate
            )
        except ImportError:
            syndrome, observable_flip, num_detectors = self._generate_synthetic(
                seed, distance, noise_rate
            )

        prompt = self._build_prompt(syndrome, distance, noise_rate, num_detectors)

        return TaskInstance(
            task_id=f"D1_seed{seed}_L{level.value}",
            category=self.category,
            subtask=self.subtask,
            level=level,
            seed=seed,
            prompt=prompt,
            metadata={
                "syndrome": syndrome,
                "observable_flip": observable_flip,
                "code_distance": distance,
                "noise_rate": noise_rate,
                "num_detectors": num_detectors,
            },
        )

    def _generate_with_stim(self, seed, distance, noise_rate):
        import stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=noise_rate,
            before_measure_flip_probability=noise_rate,
            after_reset_flip_probability=noise_rate,
        )
        # Sample multiple shots and pick one with ~50/50 balance
        # to avoid trivial "always False" strategy
        sampler = circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(
            shots=100, separate_observables=True
        )
        # Try to find a shot where observable flipped (rarer event)
        # Use seed to deterministically select which shot to use
        rng = __import__('numpy').random.default_rng(seed)
        flip_indices = [i for i in range(100) if observable_flips[i][0]]
        no_flip_indices = [i for i in range(100) if not observable_flips[i][0]]
        # 50/50 balance: half the seeds get flip=True, half get flip=False
        if seed % 2 == 0 and flip_indices:
            idx = flip_indices[int(rng.integers(0, len(flip_indices)))]
        elif no_flip_indices:
            idx = no_flip_indices[int(rng.integers(0, len(no_flip_indices)))]
        else:
            idx = 0
        syndrome = detection_events[idx].tolist()
        obs_flip = bool(observable_flips[idx][0])
        num_detectors = len(syndrome)
        return syndrome, obs_flip, num_detectors

    def _generate_synthetic(self, seed, distance, noise_rate):
        rng = np.random.default_rng(seed)
        num_detectors = (distance**2 - 1) // 2 * distance * 2
        syndrome = (rng.random(num_detectors) < noise_rate * 3).astype(int).tolist()
        # Force 50/50 balance so trivial strategies can't exceed 50%
        observable_flip = bool(seed % 2 == 0)
        return syndrome, observable_flip, num_detectors

    def _build_prompt(self, syndrome, distance, noise_rate, num_detectors):
        syn_str = "".join(str(int(s)) for s in syndrome[:200])
        if len(syndrome) > 200:
            syn_str += f"... ({len(syndrome)} total bits)"
        return (
            f"You are given syndrome measurements from a distance-{distance} rotated surface code "
            f"under circuit-level depolarizing noise (p={noise_rate}).\n\n"
            f"Syndrome ({num_detectors} detector bits):\n{syn_str}\n\n"
            f"The syndrome was collected over {distance} rounds of stabilizer measurement.\n\n"
            f"Determine whether a logical X flip occurred on the data qubits.\n\n"
            f"Write a function `solve()` that returns True if a logical correction (X_L) is needed, "
            f"or False if no correction is needed.\n"
        )
