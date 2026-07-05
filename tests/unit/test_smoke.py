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


def test_diff_on_missing_run_fails_loudly_with_reason(tmp_path) -> None:
    # No CLI stubs remain (task 016 implemented `diff`, the last one); failures
    # must still be loud and name the concrete missing input, never a bare exit.
    from eval3r.cli.main import app

    result = runner.invoke(
        app, ["diff", str(tmp_path / "run_a"), str(tmp_path / "run_b")]
    )
    assert result.exit_code == 1
    assert "results.json" in result.output
    assert "run_a" in result.output
