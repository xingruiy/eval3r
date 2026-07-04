"""Task 001 smoke tests: package imports and the CLI command tree is present."""

from __future__ import annotations

from typer.testing import CliRunner

runner = CliRunner()


def test_import_eval3r() -> None:
    import eval3r  # noqa: F401


def test_cli_help_lists_command_groups() -> None:
    from eval3r.cli.main import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for group in ("metric", "benchmark", "dataset", "protocol", "diff"):
        assert group in result.output, f"missing command group '{group}' in --help output"


def test_stub_command_fails_loudly_with_reason() -> None:
    from eval3r.cli.main import app

    result = runner.invoke(app, ["metric", "geometry"])
    # A stub must not succeed silently; it raises an explicit NotImplementedError.
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert "task 007" in str(result.exception)
