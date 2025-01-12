import json
import logging
from pathlib import Path
from time import sleep
import opik
from opik import Opik
from opik.evaluation import evaluate
from src.modeluniversity.evals_models import SameFirstLetterMetric  # noqa: F401
from functools import partial

from litellm import RateLimitError
from litellm import completion
from termcolor import colored
from .opentextbook import OpenTextBook, create_textbook_instance
from .config import settings
import random


def setup(
    client,
    dataset_name: str,
    test_questions_location: Path = Path("test_questions.json"),
):
    if not dataset_name:
        logging.warning(
            "No dataset name provided. Will use the one from the settings: {settings.opik_dataset}"
        )
        dataset_name = settings.opik_dataset
    # Create a dataset
    dataset = client.get_or_create_dataset(name=dataset_name)
    # Load the json file test_questions.json
    # Upload each [topic,subtopic,question,answer] to the dataset
    with open(test_questions_location, "r") as file:
        data = json.load(file)
        for item in data:
            item_json = json.loads(item) if not isinstance(item, dict) else item
            dataset.insert(
                [
                    {
                        "topic": item_json["topic"],
                        "subtopic": item_json["subtopic"],
                        "question": item_json["question"],
                        "question_difficulty": item_json["question_difficulty"],
                        "answer": item_json["answer"],
                        "wrong_answer1": item_json["wrong_answer1"],
                        "wrong_answer2": item_json["wrong_answer2"],
                        "wrong_answer3": item_json["wrong_answer3"],
                        "explanation": item_json["explanation"],
                    }
                ]
            )
    return dataset


def question_prompt_call(prompt: str, a_model: str):
    retries = 0
    while retries < 5:
        try:
            response = completion(
                model=a_model,
                messages=[
                    {"role": "system", "content": settings.student_role},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=256,
            )

            return response["choices"][0]["message"]["content"]
        except RateLimitError as e:
            print(
                colored("Rate limit error. Waiting 30 seconds before retrying", "red")
            )
            sleep(30)
            retries += 1


def create_answer_choices(correct_answer, wrong_answer1, wrong_answer2, wrong_answer3):
    correct_choice = random.choice(["A", "B", "C", "D"])
    wrong_choices = [
        choice for choice in ["A", "B", "C", "D"] if choice != correct_choice
    ]
    correct_choice_answer = f"{correct_choice}. {correct_answer}"
    wrong_choice1_answer = f"{wrong_choices.pop()}. {wrong_answer1}"
    wrong_choice2_answer = f"{wrong_choices.pop()}. {wrong_answer2}"
    wrong_choice3_answer = f"{wrong_choices.pop()}. {wrong_answer3}"
    answer_choices = [
        correct_choice_answer,
        wrong_choice1_answer,
        wrong_choice2_answer,
        wrong_choice3_answer,
    ]
    answer_choices.sort(key=lambda x: x[0])
    multiline = "\n".join(answer_choices)
    return (correct_choice, multiline)


def evaluation_task_open(
    dataset_item,
    a_model: str,
    textbook: OpenTextBook,
):
    # your LLM application is called here
    input = dataset_item["question"]
    print(colored("Evaluating question: " + input, "green"))
    (correct_choice, answer_choices) = create_answer_choices(
        dataset_item["answer"],
        dataset_item["wrong_answer1"],
        dataset_item["wrong_answer2"],
        dataset_item["wrong_answer3"],
    )
    precontext = settings.student_role
    prompt = (
        "What is the letter (A/B/C/D) describing the correct answer for the following multi-choice question:{"
        + input
        + "\n"
        + answer_choices
        + "\n } Provide the letter(A/B/C/D) followed by an explanation .\n"
    )
    textbook_content = textbook.query([answer_choices], 5)
    answer = question_prompt_call(
        "The following is retrieved material to help you answer the question: \n\n"
        + str(textbook_content)
        + "\n\n Your TASK:\n "
        + prompt,
        a_model=a_model,
    )
    result = {
        "input": prompt,
        "output": answer,
        "context": [precontext, str(textbook_content)],
        "reference": str(correct_choice),
    }
    return result


def evaluation_task_closed(
    dataset_item,
    a_model: str,
):
    # your LLM application is called here
    input = dataset_item["question"]
    print(colored("Evaluating question: " + input, "green"))
    (correct_choice, answer_choices) = create_answer_choices(
        dataset_item["answer"],
        dataset_item["wrong_answer1"],
        dataset_item["wrong_answer2"],
        dataset_item["wrong_answer3"],
    )
    precontext = settings.student_role
    prompt = (
        "What is the letter (A/B/C/D) describing the correct answer for the following multi-choice question:{"
        + input
        + "\n"
        + answer_choices
        + "\n } Respond first with ONE LETTER picking the answer, followed by an explanation why you chose it over other answers. For example: E. because this answer included more complete information about the topic.\n"
    )
    answer = question_prompt_call(prompt, a_model=a_model)
    result = {
        "input": prompt,
        "output": answer,
        "context": [precontext],
        "reference": str(correct_choice),
    }
    return result


def run_the_evaluation(
    an_opik_client: Opik,
    metrics: list[opik.evaluation.metrics.base_metric],
    llm_evals_list: list[str],
    use_textbook: bool,
    test_questions_location: Path = Path("test_questions.json"),
    number_of_task_threads=4,
    evaluation_dataset_name: str = None,
):
    dataset = setup(
        client=an_opik_client,
        dataset_name=evaluation_dataset_name,
        test_questions_location=test_questions_location,
    )
    accompanying_comment = (
        "my_evaluation-open: " if use_textbook else "my_evaluation-closed: "
    )

    if use_textbook:
        # Create the textbook instance
        textbook = create_textbook_instance()
        # Wrap `evaluation_task_open` so it only requires `dataset_item`
        base_task_function = evaluation_task_open
    else:
        base_task_function = evaluation_task_closed

    results_to_report = {}

    for llm in llm_evals_list:
        if use_textbook:
            task_function = partial(base_task_function, a_model=llm, textbook=textbook)
        else:
            task_function = partial(base_task_function, a_model=llm)

        eval_results = evaluate(
            experiment_name=f"{accompanying_comment}{llm}",
            dataset=dataset,
            task=task_function,
            scoring_metrics=metrics,
            task_threads=number_of_task_threads,
        )
        results_to_report[llm] = eval_results
    return results_to_report


def main():
    main_opik_client = Opik()
    runs_metrics = [SameFirstLetterMetric("Multiple-choice match")]

    use_textbook_in_main = settings.open_textbook_eval

    run_the_evaluation(
        an_opik_client=main_opik_client,
        metrics=runs_metrics,
        llm_evals_list=settings.llm_evals_list,
        use_textbook=use_textbook_in_main,
    )
    print(colored("Evaluation completed", "green"))


if __name__ == "__main__":
    main()
