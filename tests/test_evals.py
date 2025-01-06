from pathlib import Path
from src.modeluniversity.evals import run_the_evaluation
import pytest
from opik import Opik
from src.modeluniversity.evals_models import SameFirstLetterMetric
from src.modeluniversity.config import settings


@pytest.fixture(scope="session")
def opik_client_for_test_session() -> Opik:
    return Opik()


@pytest.fixture(scope="session")
def metrics_for_test_session() -> Opik:
    return [SameFirstLetterMetric("Multiple-choice match")]


@pytest.fixture(scope="session")
def list_of_llms_to_judge() -> Opik:
    return settings.llm_evals_list


@pytest.fixture(scope="session")
def location_of_questions_for_test_session(test_data_dir) -> Path:
    return test_data_dir / Path("test_questions.json")


@pytest.mark.parametrize("use_textbook", [True, False])
def test_run_the_eval_with_textbook(
    opik_client_for_test_session,
    metrics_for_test_session,
    list_of_llms_to_judge,
    location_of_questions_for_test_session,
    use_textbook,
):
    eval_results = run_the_evaluation(
        an_opik_client=opik_client_for_test_session,
        metrics=metrics_for_test_session,
        llm_evals_list=list_of_llms_to_judge,
        test_questions_location=location_of_questions_for_test_session,
        use_textbook=use_textbook,
    )
    assert eval_results is not None
    # TODO: What will we be checking?!
    assert ...
