import click
from pathlib import Path

from .datagen import (
    generate_curriculum,
    generate_questions,
    transform_to_trainable_json,
)


@click.group()
def cli():
    """ModelUniversity CLI tool for data generation and management."""
    pass


@cli.command()
@click.option(
    "--curriculum-file",
    type=click.Path(),
    default="curriculum.json",
    help="Path to the curriculum JSON file",
)
def create_curriculum(curriculum_file):
    """Generate a new curriculum."""
    curriculum = generate_curriculum(Path(curriculum_file))
    click.echo(f"Curriculum generated and saved to {curriculum_file}")
    return curriculum


@cli.command()
@click.option(
    "--curriculum-file",
    type=click.Path(exists=True),
    default="curriculum.json",
    help="Path to the curriculum JSON file",
)
@click.option(
    "--training-file",
    type=click.Path(),
    default="training_questions.json",
    help="Path to save training questions",
)
@click.option(
    "--testing-file",
    type=click.Path(),
    default="test_questions.json",
    help="Path to save test questions",
)
def create_questions(curriculum_file, training_file, testing_file):
    """Generate training and test questions from curriculum."""
    with open(curriculum_file, "r") as f:
        curriculum = generate_curriculum(Path(curriculum_file))

    generate_questions(
        curriculum=curriculum,
        training_questions_file=training_file,
        testing_questions_file=testing_file,
    )
    click.echo(
        f"Questions generated:\nTraining: {training_file}\nTesting: {testing_file}"
    )


@cli.command()
@click.option(
    "--training-file",
    type=click.Path(exists=True),
    default="training_questions.json",
    help="Path to training questions JSON file",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default="trainable_data.json",
    help="Path to save transformed trainable data",
)
def transform_data(training_file, output_file):
    """Transform training questions into trainable format."""
    transform_to_trainable_json(
        training_questions_file=Path(training_file),
        trainable_questions_file=Path(output_file),
    )
    click.echo(f"Data transformed and saved to {output_file}")


if __name__ == "__main__":
    cli()
