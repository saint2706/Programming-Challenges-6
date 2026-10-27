import pytest
from dist_pipeline.cli import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.parametrize(
    "command", ["--help", "download", "prepare", "verify", "run", "report"]
)
def test_commands_show_help(command):
    args = [command, "--help"] if command != "--help" else [command]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output or "usage" in result.output.lower()


@pytest.mark.parametrize("command", ["verify", "run"])
@pytest.mark.parametrize("scale", ["x/../../y", "1m'; --", "2m", ""])
def test_unknown_scale_is_a_usage_error_before_any_work(command, scale):
    result = runner.invoke(app, [command, "--scale", scale])
    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)
