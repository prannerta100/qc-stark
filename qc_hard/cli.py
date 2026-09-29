import argparse
import logging
from qc_hard.registry import build_generator_registry, build_verifier_registry, build_model_registry
from qc_hard.evaluation.runner import EvaluationRunner
from qc_hard.reporting.csv_report import generate_report, generate_summary
from qc_hard.types import DifficultyLevel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALL_SUBTASKS = ["A1_state_prep", "B1_single_bug", "C1_routing", "D1_syndrome_decoding"]


def run_benchmark(
    subtasks: list[str] | None = None,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    levels: list[int] | None = None,
    output_path: str = "results/results.csv",
    summary_path: str = "results/summary.csv",
):
    subtasks = subtasks or ALL_SUBTASKS
    models = models or ["mock"]
    seeds = seeds or list(range(1, 11))
    levels_enum = [DifficultyLevel(lvl) for lvl in (levels or [1, 2, 3])]

    gen_reg = build_generator_registry()
    ver_reg = build_verifier_registry()
    model_reg = build_model_registry(models)

    runner = EvaluationRunner(gen_reg, ver_reg, model_reg)
    results = runner.run(
        subtasks=subtasks,
        models=models,
        seeds=seeds,
        levels=levels_enum,
    )

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    generate_report(results, output_path)
    generate_summary(results, summary_path)
    logging.info(f"Results: {output_path} | Summary: {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="QC-Hard Benchmark Runner")
    parser.add_argument("--subtasks", nargs="+", default=None, help="Subtasks to run")
    parser.add_argument("--models", nargs="+", default=["mock"], help="Model names")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", default="results/results.csv")
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()

    run_benchmark(
        subtasks=args.subtasks,
        models=args.models,
        seeds=args.seeds,
        levels=args.levels,
        output_path=args.output,
        summary_path=args.summary,
    )


if __name__ == "__main__":
    main()
