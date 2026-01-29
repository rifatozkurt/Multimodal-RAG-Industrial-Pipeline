import argparse
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Job:
    name: str
    image_retrieval_mode: str
    merge_type: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run run_eval + score_eval experiments across GPUs.")
    parser.add_argument("--dataset", default="eval/datasets/dataset_mcq_fib.json")
    parser.add_argument("--run_root", default="eval/runs")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--filtered_image_retrieval", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {base_dir}")
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in: {base_dir}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def run_job(job: Job, gpu: str, args: argparse.Namespace) -> None:
    run_dir = Path(args.run_root) / job.name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [
        sys.executable,
        "eval/metrics/run_eval.py",
        "--dataset",
        args.dataset,
        "--run_dir",
        str(run_dir),
        "--image_retrieval_mode",
        job.image_retrieval_mode,
    ]
    if job.merge_type is not None:
        cmd.extend(["--merge_type", job.merge_type])
    if not args.filtered_image_retrieval:
        cmd.append("--no-filtered_image_retrieval")

    subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    latest = latest_run_dir(run_dir)
    predictions_path = latest / "predictions.jsonl"

    score_cmd = [
        sys.executable,
        "eval/metrics/score_eval.py",
        "--dataset",
        args.dataset,
        "--predictions",
        str(predictions_path),
    ]
    subprocess.run(score_cmd, check=True, cwd=ROOT, env=env)


def worker(gpu: str, queue: Queue, args: argparse.Namespace, progress: tqdm) -> None:
    while True:
        job = queue.get()
        if job is None:
            queue.task_done()
            return
        try:
            print(f"[GPU {gpu}] Starting {job.name}")
            run_job(job, gpu, args)
            print(f"[GPU {gpu}] Finished {job.name}")
            progress.update(1)
        finally:
            queue.task_done()


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus to specify GPU ids.")

    jobs: list[Job] = [Job(name="vlm_caption", image_retrieval_mode="vlm_caption", merge_type=None)]
    for merge_type in ["normalized_mean", "similarity_weighted", "query_interpolation", "pca"]:
        jobs.append(
            Job(
                name=f"clip_repr_{merge_type}",
                image_retrieval_mode="clip_repr",
                merge_type=merge_type,
            )
        )

    queue: Queue = Queue()
    for job in jobs:
        queue.put(job)

    progress = tqdm(total=len(jobs), desc="Experiments", unit="job")
    threads = []
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu, queue, args, progress), daemon=True)
        t.start()
        threads.append(t)

    queue.join()
    progress.close()
    for _ in threads:
        queue.put(None)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
