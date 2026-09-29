from qc_hard.generators.base import GeneratorRegistry
from qc_hard.verifiers.base import VerifierRegistry
from qc_hard.models.base import ModelRegistry
from qc_hard.models.mock import MockModel
from qc_hard.generators.cat_a_synthesis import StatePrepGenerator
from qc_hard.generators.cat_b_debugging import SingleBugGenerator
from qc_hard.generators.cat_c_compilation import RoutingGenerator
from qc_hard.generators.cat_d_qec import SyndromeDecodingGenerator
from qc_hard.generators.cat_e_optimization import OptimizationGenerator
from qc_hard.generators.cat_f_equivalence import EquivalenceGenerator
from qc_hard.generators.cat_g_trotter import TrotterGenerator
from qc_hard.generators.cat_h_oracle import OracleGenerator
from qc_hard.generators.cat_i_noise import NoiseFidelityGenerator
from qc_hard.verifiers.cat_a_synthesis import StatePrepVerifier
from qc_hard.verifiers.cat_b_debugging import SingleBugVerifier
from qc_hard.verifiers.cat_c_compilation import RoutingVerifier
from qc_hard.verifiers.cat_d_qec import SyndromeDecodingVerifier
from qc_hard.verifiers.cat_e_optimization import OptimizationVerifier
from qc_hard.verifiers.cat_f_equivalence import EquivalenceVerifier
from qc_hard.verifiers.cat_g_trotter import TrotterVerifier
from qc_hard.verifiers.cat_h_oracle import OracleVerifier
from qc_hard.verifiers.cat_i_noise import NoiseFidelityVerifier


def build_generator_registry() -> GeneratorRegistry:
    reg = GeneratorRegistry()
    reg.register(StatePrepGenerator())
    reg.register(SingleBugGenerator())
    reg.register(RoutingGenerator())
    reg.register(SyndromeDecodingGenerator())
    reg.register(OptimizationGenerator())
    reg.register(EquivalenceGenerator())
    reg.register(TrotterGenerator())
    reg.register(OracleGenerator())
    reg.register(NoiseFidelityGenerator())
    return reg


def build_verifier_registry() -> VerifierRegistry:
    reg = VerifierRegistry()
    reg.register(StatePrepVerifier())
    reg.register(SingleBugVerifier())
    reg.register(RoutingVerifier())
    reg.register(SyndromeDecodingVerifier())
    reg.register(OptimizationVerifier())
    reg.register(EquivalenceVerifier())
    reg.register(TrotterVerifier())
    reg.register(OracleVerifier())
    reg.register(NoiseFidelityVerifier())
    return reg


def build_model_registry(model_names: list[str] | None = None) -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(MockModel("mock", "def solve():\n    pass"))

    if model_names:
        for name in model_names:
            if name == "mock":
                continue
            if name.startswith("gpt") or name.startswith("o"):
                from qc_hard.models.openai_proxy import OpenAIModel
                reg.register(OpenAIModel(name=name, model_id=name))
            elif name.startswith("claude"):
                from qc_hard.models.anthropic_proxy import AnthropicModel
                reg.register(AnthropicModel(name=name, model_id=name))

    return reg
