"""Subprocess wrapper for ba_cli.exe."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from calibri.tools.utils import find_ba_cli


@dataclass
class SolverResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    result_path: Path


def run_solver(
    problem_path: Path,
    result_path: Path,
    cli_path: Optional[Path] = None,
    timeout: int = 120,
) -> SolverResult:
    """Run ba_cli.exe on a problem.json and produce result.json.

    Args:
        problem_path: Path to input problem.json.
        result_path: Path where result.json will be written.
        cli_path: Explicit path to ba_cli.exe (auto-detected if None).
        timeout: Subprocess timeout in seconds.

    Returns:
        SolverResult with exit code, captured output, and timing.
    """
    cli = find_ba_cli(cli_path)
    problem_path = Path(problem_path)
    result_path = Path(result_path)

    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_path}")

    cmd = [str(cli), str(problem_path), str(result_path)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        return SolverResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=elapsed,
            result_path=result_path,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return SolverResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Solver timed out after {timeout}s",
            elapsed_seconds=elapsed,
            result_path=result_path,
        )
