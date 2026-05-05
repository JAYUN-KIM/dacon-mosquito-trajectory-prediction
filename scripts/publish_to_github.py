from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH = "main"
DEFAULT_REMOTE = "https://github.com/JAYUN-KIM/dacon-mosquito-trajectory-prediction.git"
TRACKED_SCOPE = [
    "README.md",
    "docs",
    "experiments",
    "reports",
    "scripts",
    "src",
]


def git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=check)


def ensure_remote(remote_url: str) -> None:
    remotes = git(["remote"], check=True).stdout.split()
    if "origin" not in remotes:
        print(f"Adding origin: {remote_url}")
        git(["remote", "add", "origin", remote_url])
        return

    current = git(["remote", "get-url", "origin"]).stdout.strip()
    if current != remote_url:
        print(f"Updating origin from {current} to {remote_url}")
        git(["remote", "set-url", "origin", remote_url])


def has_staged_changes() -> bool:
    diff = git(["diff", "--cached", "--quiet"], check=False)
    return diff.returncode != 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit and push project code/results to GitHub.")
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--message", default="Update mosquito trajectory pipeline")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        ensure_remote(args.remote_url)

    current_branch = git(["branch", "--show-current"]).stdout.strip()
    if current_branch != args.branch:
        if args.dry_run:
            print(f"Dry run: would switch/create branch {args.branch}")
            return
        existing = git(["branch", "--list", args.branch]).stdout.strip()
        if existing:
            git(["switch", args.branch])
        else:
            git(["switch", "-c", args.branch])

    if args.dry_run:
        print(git(["status", "--short"]).stdout)
        print("Dry run: no files were staged, committed, or pushed.")
        return

    git(["add", *TRACKED_SCOPE])
    if not has_staged_changes():
        print("No tracked code/report changes to commit.")
    else:
        git(["commit", "-m", args.message])
        print(f"Committed: {args.message}")

    push = subprocess.run(
        ["git", "push", "-u", "origin", args.branch],
        cwd=ROOT,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if push.returncode != 0:
        raise SystemExit(
            "git push failed. If this machine is not authenticated, install GitHub CLI and run `gh auth login`, "
            "or sign in through Git Credential Manager, then retry."
        )


if __name__ == "__main__":
    main()
