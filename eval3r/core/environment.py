"""Environment metadata capture.

Records only the minimal, non-identifying facts needed to interpret a run: the
software versions and the platform/OS/architecture it ran on. It deliberately does
**not** record identifying or unnecessary information — no environment variables,
tokens, cookies, credentials, working directory, git state, timestamp, or timezone
(``.agent/reproducibility.md`` data rules).
"""

from __future__ import annotations

import platform
from typing import Any

from eval3r import __version__ as EVAL3R_VERSION


def capture_environment(command: str | None = None) -> dict[str, Any]:
    """Return a minimal, non-identifying environment metadata dict.

    ``command`` is the resolved CLI command (without any surrounding shell/user
    context); pass ``None`` to omit it.
    """
    return {
        "eval3r_version": EVAL3R_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu_architecture": platform.machine(),
        "command": command,
    }
