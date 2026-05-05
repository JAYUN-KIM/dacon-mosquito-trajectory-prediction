from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def latest_experiment(log_path: Path) -> dict:
    logs = json.loads(log_path.read_text(encoding="utf-8"))
    if not logs:
        raise ValueError(f"no experiment entries found in {log_path}")
    return logs[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline experiment, validate output, and print the readout.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--skip-param-search", action="store_true")
    parser.add_argument("--publish-github", action="store_true", help="Commit and push code/results after a successful run.")
    parser.add_argument("--remote-url", default="https://github.com/JAYUN-KIM/dacon-mosquito-trajectory-prediction.git")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = ROOT / "experiments" / "log.json"
    report_path = ROOT / "reports" / "latest_physics_baseline.md"

    baseline_cmd = [
        sys.executable,
        "scripts/run_physics_baselines.py",
        "--data-dir",
        str(args.data_dir),
        "--val-frac",
        str(args.val_frac),
        "--seed",
        str(args.seed),
        "--experiment-log",
        str(log_path),
        "--report-path",
        str(report_path),
    ]
    if args.skip_submission:
        baseline_cmd.append("--skip-submission")

    run_command(baseline_cmd)
    latest = latest_experiment(log_path)

    submission_path = latest.get("submission_path")
    if submission_path:
        run_command([sys.executable, "scripts/validate_submission.py", str(submission_path)])

    param_submission_path = ROOT / "submissions" / "physics_param_search_best.csv"
    param_report_path = ROOT / "reports" / "latest_physics_param_search.md"
    if not args.skip_param_search:
        run_command(
            [
                sys.executable,
                "scripts/search_physics_params.py",
                "--data-dir",
                str(args.data_dir),
                "--val-frac",
                str(args.val_frac),
                "--seed",
                str(args.seed),
                "--report-path",
                str(param_report_path),
            ]
        )
        run_command([sys.executable, "scripts/validate_submission.py", str(param_submission_path)])

    print("\n=== Latest experiment readout ===")
    print(f"best_method: {latest.get('best_method')}")
    print(f"submission_path: {submission_path}")
    print(f"report_path: {report_path}")
    if not args.skip_param_search:
        print(f"param_search_submission_path: {param_submission_path}")
        print(f"param_search_report_path: {param_report_path}")

    if args.publish_github:
        run_command(
            [
                sys.executable,
                "scripts/publish_to_github.py",
                "--remote-url",
                args.remote_url,
                "--message",
                "Add automated mosquito baseline pipeline",
            ]
        )


if __name__ == "__main__":
    main()
