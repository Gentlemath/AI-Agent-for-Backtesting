from __future__ import annotations

import subprocess
import sys
from pathlib import Path

class CodeVerifierAgent:
    """Ensures generated strategy modules are syntactically runnable."""

    def __init__(self, python_bin: str | None = None):
        self.python_bin = python_bin or sys.executable

    def verify(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy module missing: {file_path}")
        cmd = [
            self.python_bin,
            "-m",
            "py_compile",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"py_compile failed for {path.name}: {stderr or stdout or 'unknown error'}"
            )
