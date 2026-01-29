import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from core.config import eval_runs_path  # noqa: E402

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a summary.csv for eval runs (score_eval outputs) under run_root."
    )
    parser.add_argument("--run_root", default=str(eval_runs_path))
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def guess_timestamp(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        if TIMESTAMP_RE.match(part):
            return part
    return None


def collect_run_summaries(run_root: Path) -> list[dict]:
    summaries = []
    for metrics_path in sorted(run_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        if not run_dir.is_dir():
            continue

        rel = run_dir.relative_to(run_root)
        parts = rel.parts
        run_name = parts[0] if parts else ""
        timestamp = guess_timestamp(parts)

        metrics = load_json(metrics_path)
        config_path = run_dir / "run_config.json"
        config = load_json(config_path) if config_path.exists() else {}

        row = {
            "run_id": str(rel),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "timestamp": timestamp,
        }
        row.update(config)
        row.update(
            {
                "dataset": metrics.get("dataset"),
                "predictions": metrics.get("predictions"),
                "missing_predictions": metrics.get("missing_predictions"),
                "counts": metrics.get("counts"),
                "metrics": metrics.get("metrics"),
            }
        )
        summaries.append(row)
    return summaries


def build_header(summaries: list[dict]) -> list[str]:
    if not summaries:
        return []

    run_info = ["run_id", "run_dir", "run_name", "timestamp"]
    summary_fields = ["dataset", "predictions", "missing_predictions", "counts", "metrics"]

    keys = set()
    for row in summaries:
        keys.update(row.keys())

    config_keys = sorted(k for k in keys if k not in set(run_info + summary_fields))
    header = []
    for key in run_info + config_keys + summary_fields:
        if key in keys:
            header.append(key)

    for key in sorted(keys):
        if key not in header:
            header.append(key)
    return header


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if row.get(h) is None else str(row.get(h)) for h in header) + "\n")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    summaries = collect_run_summaries(run_root)

    output_csv = Path(args.output_csv) if args.output_csv else run_root / "summary.csv"
    output_json = Path(args.output_json) if args.output_json else run_root / "summary.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    header = build_header(summaries)
    if header:
        write_csv(output_csv, header, summaries)
    else:
        output_csv.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
