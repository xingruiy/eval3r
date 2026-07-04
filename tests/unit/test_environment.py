"""Task 006 environment-capture tests: minimal shape and no identifying info."""

from __future__ import annotations

from eval3r.core.environment import capture_environment

EXPECTED_KEYS = {
    "eval3r_version",
    "python_version",
    "python_implementation",
    "platform",
    "os",
    "cpu_architecture",
    "command",
}


def test_capture_environment_shape() -> None:
    env = capture_environment(command="e3r protocol show single_geometry")
    assert set(env) == EXPECTED_KEYS
    assert env["command"] == "e3r protocol show single_geometry"


def test_command_omitted_when_none() -> None:
    env = capture_environment()
    assert env["command"] is None


def test_environment_records_no_identifying_or_secret_info() -> None:
    env = capture_environment(command="e3r protocol show single_geometry")
    # No secret-ish keys, and no working-dir / git / timestamp / timezone fields.
    for banned in (
        "api_key",
        "token",
        "password",
        "secret",
        "cookie",
        "environ",
        "working_directory",
        "cwd",
        "git",
        "timestamp",
        "timezone",
        "hostname",
        "username",
        "executable",
    ):
        assert banned not in env, f"environment leaked '{banned}'"
        assert banned not in str(env).lower()
