import json
from pathlib import Path
import pickle
from src.modeluniversity.datagen import (
    transform_to_trainable_json,
    create_conversation,
    generate_curriculum,
    generate_questions,
)
from src.modeluniversity.datagen_models import (
    CurriculumSchema,
    TrainQuestionsSchema,
    TestQuestionsSchema,
)
import pytest


@pytest.fixture(scope="session")
def a_curriculum_file_that_exists(test_data_dir) -> Path:
    return test_data_dir / Path("test_curriculum.json")


@pytest.fixture(scope="session")
def a_curriculum_file_that_doesnt_exist(test_outputs_dir) -> Path:
    return test_outputs_dir / Path("a_random_file_YkfkqJWNkRRomEOlNFQlJJAiCG.json")


@pytest.fixture(scope="session")
def the_curriculum_read_from_test_data_folder(
    a_curriculum_file_that_exists,
) -> Path:
    pre_defined_curriculum: dict = generate_curriculum(a_curriculum_file_that_exists)
    return pre_defined_curriculum


@pytest.fixture(scope="session")
def some_training_questions():
    training_questions = [
        {
            "topic": "Estate Planning Fundamentals",
            "subtopic": "introduction to estate planning",
            "question": "What is the purpose of a living will?",
            "answer": "To outline an individual's wishes for end-of-life medical care",
            "explanation": "A living will is a document that outlines an individual's wishes for end-of-life medical care, often used in conjunction with a power of attorney.",
        },
        {
            "topic": "Estate Planning Fundamentals",
            "subtopic": "introduction to estate planning",
            "question": "How does the Uniform Probate Code (UPC) affect estate planning?",
            "answer": "The UPC provides a standardized framework for probate and estate administration, making it easier to navigate the process",
            "explanation": "The Uniform Probate Code (UPC) provides a standardized framework for probate and estate administration, making it easier to navigate the process and reducing the complexity of estate planning.",
        },
    ]
    return training_questions


def test_transform_to_trainable_json(
    test_outputs_dir, mock_data_dir, some_training_questions
):
    # Create mock training questions in test outputs

    training_file = mock_data_dir / "mock_training_questions_produced_by_datagen.json"
    trainable_file = test_outputs_dir / "trainable_data.json"
    with training_file.open("w") as f:
        json.dump(some_training_questions, f)

    # Call the function
    transform_to_trainable_json(training_file, trainable_file)

    # Validate the output
    with trainable_file.open("r") as f:
        data = json.load(f)

    assert len(data) > 1

    expected_key_first_hierarchy = ["messages"]
    expected_keys_of_second_hierarchy = ["role", "content"]
    assert len(data) > 1

    # How to check that data contain the above keys in the right hierarchy?
    assert all(key in data[0] for key in expected_key_first_hierarchy)
    assert all(
        key in data[0]["messages"][0] for key in expected_keys_of_second_hierarchy
    )

    # Check that for each element of the training_questions list, the first element of the data list contains the question
    # That's included in the `data` list in the respective ["messages"] in the dict that contains "role": "assistant" and it's the value behind the "content" key
    for q, question in enumerate(some_training_questions):
        target_trainable = data[q]
        dict_with_answer = [
            message
            for message in target_trainable["messages"]
            if message["role"] == "assistant"
        ][0]
        the_answer = dict_with_answer["content"]

        the_answer_in_previous_format = question["answer"]
        assert the_answer_in_previous_format in the_answer


def test_create_conversation():
    sample = {
        "question": "What is Python?",
        "answer": "A programming language",
        "explanation": "Python is a versatile programming language.",
    }

    conversation = create_conversation(sample)

    assert conversation["messages"][0]["role"] == "system"
    assert "Answer the following question" in conversation["messages"][1]["content"]


def test_discover_existing_curriculum(the_curriculum_read_from_test_data_folder):
    assert "topics" in the_curriculum_read_from_test_data_folder
    assert "topic" in the_curriculum_read_from_test_data_folder["topics"][0]
    assert (
        "Estate Planning Fundamentals"
        in the_curriculum_read_from_test_data_folder["topics"][0]["topic"]
    )


def test_generate_curriculum(
    a_curriculum_file_that_doesnt_exist, mock_data_dir, monkeypatch
):
    assert not a_curriculum_file_that_doesnt_exist.exists()

    mock_curriculum_is_at = mock_data_dir / Path(
        "mock_curriculum_generated_by_datagen.pickle"
    )

    def mock_curriculum(*args, **kwargs):
        with open(mock_curriculum_is_at, "rb") as f:
            loaded_data = pickle.load(f)
        return loaded_data

    # Mock the completion function in the litellm module
    monkeypatch.setattr("src.modeluniversity.datagen.completion", mock_curriculum)

    curriculum_made_on_the_spot = generate_curriculum(
        a_curriculum_file_that_doesnt_exist
    )

    # Test that a file is written at a_curriculum_file_that_doesnt_exist
    assert a_curriculum_file_that_doesnt_exist.exists()

    # Validate the content of the generated curriculum
    with a_curriculum_file_that_doesnt_exist.open("r") as f:
        curriculum_data_from_written_file = json.load(f)

    assert "topics" in curriculum_data_from_written_file
    assert isinstance(curriculum_data_from_written_file["topics"], list)
    assert len(curriculum_data_from_written_file["topics"]) > 0
    assert "topic" in curriculum_data_from_written_file["topics"][0]
    assert "subtopics" in curriculum_data_from_written_file["topics"][0]

    # Also test the read content curriculum_data is the same as the one function returned pre_defined_curriculum
    assert curriculum_data_from_written_file == curriculum_made_on_the_spot


@pytest.fixture
def mock_training_questions(mock_data_dir):
    """Fixture to load mock training questions data."""
    with open(mock_data_dir / "mock_training_questions.json") as f:
        return json.load(f)


@pytest.fixture
def mock_test_questions(mock_data_dir):
    """Fixture to load mock test questions data."""
    with open(mock_data_dir / "mock_test_questions.json") as f:
        return json.load(f)


def test_generate_questions(
    the_curriculum_read_from_test_data_folder,
    test_outputs_dir,
    mock_training_questions,
    mock_test_questions,
    monkeypatch,
):

    # the mock function
    def mock_question_prompt_call(prompt, schema):
        if schema == TrainQuestionsSchema:
            return json.dumps(mock_training_questions)
        elif schema == TestQuestionsSchema:
            return json.dumps(mock_test_questions)
        else:
            return None

    # Set file paths for outputs
    training_file_for_tests = test_outputs_dir / "training_questions.json"
    testing_file_for_tests = test_outputs_dir / "test_questions.json"

    def keep_first_element_only(data):
        if isinstance(data, list):
            return [keep_first_element_only(data[0])]
        elif isinstance(data, dict):
            return {k: keep_first_element_only(v) for k, v in data.items()}
        else:
            return data

    test_curriculum_light = keep_first_element_only(
        the_curriculum_read_from_test_data_folder
    )

    # Use monkeypatch to replace the real `question_prompt_call` with the mock
    monkeypatch.setattr(
        "src.modeluniversity.datagen.question_prompt_call", mock_question_prompt_call
    )

    # Generate questions
    generate_questions(
        curriculum=test_curriculum_light,
        training_questions_file=training_file_for_tests,
        testing_questions_file=testing_file_for_tests,
    )

    # Validate training questions
    with training_file_for_tests.open("r") as f:
        training_data = json.load(f)
    assert len(training_data) > 0

    # Validate testing questions
    with testing_file_for_tests.open("r") as f:
        testing_data = json.load(f)
    assert len(testing_data) > 0
