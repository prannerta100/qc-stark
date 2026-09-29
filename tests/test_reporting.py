import os
import tempfile
import pandas as pd
from qc_hard.reporting.csv_report import generate_report, generate_summary
from qc_hard.types import VerificationResult


def _make_results():
    return [
        VerificationResult("A1_seed1_L1", "gpt-4o", True, 0.999, {"fidelity": 0.999}),
        VerificationResult("A1_seed2_L1", "gpt-4o", False, 0.5, {"fidelity": 0.5}),
        VerificationResult("A1_seed1_L1", "claude-4", True, 0.998, {"fidelity": 0.998}),
        VerificationResult("A1_seed2_L1", "claude-4", True, 0.995, {"fidelity": 0.995}),
        VerificationResult("B1_seed1_L2", "gpt-4o", False, 0.0, {"error": "timeout"}),
        VerificationResult("B1_seed1_L2", "claude-4", True, 1.0, {}),
    ]


def test_generate_report_creates_csv():
    results = _make_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.csv")
        generate_report(results, path)
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) == 6
        assert "task_id" in df.columns
        assert "model_name" in df.columns
        assert "correct" in df.columns
        assert "score" in df.columns


def test_generate_summary():
    results = _make_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "summary.csv")
        generate_summary(results, path)
        df = pd.read_csv(path)
        assert "model_name" in df.columns
        assert "accuracy" in df.columns
        assert "mean_score" in df.columns
