import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, mock_open
from src.modeluniversity.cli import (
    cli,
    create_curriculum,
    create_questions,
    transform_data,
)


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_help(cli_runner):
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ModelUniversity CLI tool" in result.output


@patch("modeluniversity.cli.generate_curriculum")
def test_create_curriculum(mock_gen_curr, cli_runner):
    mock_gen_curr.return_value = {"topics": []}
    result = cli_runner.invoke(cli, ["create-curriculum"])
    assert result.exit_code == 0
    assert "Curriculum generated and saved to curriculum.json" in result.output
    mock_gen_curr.assert_called_once_with(Path("curriculum.json"))


@patch("modeluniversity.cli.generate_curriculum")
@patch("modeluniversity.cli.generate_questions")
def test_create_questions(mock_gen_q, mock_gen_curr, cli_runner):
    mock_gen_curr.return_value = {"topics": []}
    mock_gen_q.return_value = None

    with patch("builtins.open", mock_open()):
        result = cli_runner.invoke(cli, ["create-questions"])

    assert result.exit_code == 0
    assert "Questions generated" in result.output
    mock_gen_curr.assert_called_once()
    mock_gen_q.assert_called_once()


@patch("modeluniversity.cli.transform_to_trainable_json")
def test_transform_data(mock_transform, cli_runner):
    result = cli_runner.invoke(cli, ["transform-data"])
    assert result.exit_code == 0
    assert "Data transformed and saved to trainable_data.json" in result.output
    mock_transform.assert_called_once_with(
        training_questions_file=Path("training_questions.json"),
        trainable_questions_file=Path("trainable_data.json"),
    )


def test_invalid_command(cli_runner):
    result = cli_runner.invoke(cli, ["invalid-command"])
    assert result.exit_code != 0
