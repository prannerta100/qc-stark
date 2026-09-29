import os
import tempfile
import pandas as pd
from qc_hard.cli import run_benchmark


def test_full_pipeline_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "results.csv")
        summary_path = os.path.join(tmpdir, "summary.csv")
        results = run_benchmark(
            subtasks=["A1_state_prep", "D1_syndrome_decoding"],
            models=["mock"],
            seeds=[1, 2],
            levels=[1],
            output_path=results_path,
            summary_path=summary_path,
        )
        assert len(results) == 4  # 2 subtasks * 2 seeds * 1 model
        assert os.path.exists(results_path)
        assert os.path.exists(summary_path)
        df = pd.read_csv(results_path)
        assert len(df) == 4


def test_deterministic_across_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        r1 = run_benchmark(
            subtasks=["D1_syndrome_decoding"], models=["mock"], seeds=[42], levels=[1],
            output_path=os.path.join(tmpdir, "r1.csv"),
            summary_path=os.path.join(tmpdir, "s1.csv"),
        )
        r2 = run_benchmark(
            subtasks=["D1_syndrome_decoding"], models=["mock"], seeds=[42], levels=[1],
            output_path=os.path.join(tmpdir, "r2.csv"),
            summary_path=os.path.join(tmpdir, "s2.csv"),
        )
        assert r1[0].score == r2[0].score
        assert r1[0].correct == r2[0].correct
