import pandas as pd
from qc_hard.types import VerificationResult


def generate_report(results: list[VerificationResult], output_path: str) -> pd.DataFrame:
    rows = []
    for r in results:
        parts = r.task_id.split("_")
        category = parts[0] if parts else "unknown"
        level = ""
        for p in parts:
            if p.startswith("L"):
                level = p
                break
        rows.append({
            "task_id": r.task_id,
            "model_name": r.model_name,
            "category": category,
            "level": level,
            "correct": r.correct,
            "score": r.score,
            **{f"detail_{k}": v for k, v in r.details.items()},
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def generate_summary(results: list[VerificationResult], output_path: str) -> pd.DataFrame:
    rows = []
    for r in results:
        parts = r.task_id.split("_")
        category = parts[0] if parts else "unknown"
        level = ""
        for p in parts:
            if p.startswith("L"):
                level = p
                break
        rows.append({
            "task_id": r.task_id,
            "model_name": r.model_name,
            "category": category,
            "level": level,
            "correct": r.correct,
            "score": r.score,
        })

    df = pd.DataFrame(rows)
    summary = df.groupby(["model_name", "category"]).agg(
        accuracy=("correct", "mean"),
        mean_score=("score", "mean"),
        n_tasks=("task_id", "count"),
    ).reset_index()

    summary.to_csv(output_path, index=False)
    return summary
