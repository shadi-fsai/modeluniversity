import json
from time import sleep
import yaml
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Equals
from typing import Any
from opik.evaluation.metrics import base_metric, score_result


from litellm import BaseModel, RateLimitError
from litellm import completion
from termcolor import colored
from .opentextbook import OpenTextBook

from .config import settings
import random

opik_client = Opik()
my_model = ""
textbook = None


class LLMJudgeSchema(BaseModel):
    score: float
    reason: str


class SameFirstLetterMetric(base_metric.BaseMetric):
    def __init__(self, name: str):
        self.name = name

    def score(self, output: str, reference: str, **ignored_kwargs: Any):
        prompt = (
            "You are an impartial judge evaluating the answer to a multiple-choice question. The correct answer is provided in REFERENCE, and the student's answer is provided in OUTPUT. If the student picked the right multi-choice letter give a score of 1.0, if they picked the wrong one give a score of 0.0; provide your reason for the choice.\n\n"
            + "REFERENCE: "
            + reference
            + "\n\n"
            + "OUTPUT: "
            + output
            + "\n\n"
            + "SCORE: "
        )

        response = completion(
            model=settings.opik_eval_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge of a multiple-choice question answering. ",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
            response_format=LLMJudgeSchema,
        )

        response_json = json.loads(response.choices[0].message.content)

        return score_result.ScoreResult(
            value=response_json["score"], name=self.name, reason=response_json["reason"]
        )


def setup(client):
    # Create a dataset
    dataset = client.get_or_create_dataset(name=settings.opik_dataset)
    # Load the json file test_questions.json
    # Upload each [topic,subtopic,question,answer] to the dataset
    with open("test_questions.json", "r") as file:
        data = json.load(file)
        for item in data:
            item_json = json.loads(item) if not isinstance(item, dict) else item
            dataset.insert(
                [
                    {
                        "topic": item_json["topic"],
                        "subtopic": item_json["subtopic"],
                        "question": item_json["question"],
                        "answer": item_json["answer"],
                        "wrong_answer1": item_json["wrong_answer1"],
                        "wrong_answer2": item_json["wrong_answer2"],
                        "wrong_answer3": item_json["wrong_answer3"],
                        "explanation": item_json["explanation"],
                    }
                ]
            )
    return dataset


def question_prompt_call(prompt):
    retries = 0
    while retries < 5:
        try:
            response = completion(
                model=my_model,
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


def evaluation_task_open(dataset_item):
    # your LLM application is called here
    input = dataset_item["question"]
    print(colored("Evaluating question: " + input, "green"))
    (correct_choice, answer_choices) = create_answer_choices(
        dataset_item["answer"],
        dataset_item["wrong_answer1"],
        dataset_item["wrong_answer2"],
        dataset_item["wrong_answer3"],
    )
    precontext = config["student_role"]
    prompt = (
        "What is the letter (A/B/C/D) describing the correct answer for the following multi-choice question:{"
        + input
        + "\n"
        + answer_choices
        + "\n } Provide the letter(A/B/C/D) followed by an explanation .\n"
    )
    # prompt_textbook = question_prompt_call("If you had access to a search engine, What search query would you use to answer the following multi-choice question:" + answer_choices)
    textbook_content = textbook.query([answer_choices], 5)
    answer = question_prompt_call(
        "The following is retrieved material to help you answer the question: \n\n"
        + str(textbook_content)
        + "\n\n Your TASK:\n "
        + prompt
    )
    result = {
        "input": prompt,
        "output": answer,
        "context": [precontext, str(textbook_content)],
        "reference": str(correct_choice),
    }
    return result


def evaluation_task_closed(dataset_item):
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
    answer = question_prompt_call(prompt)
    result = {
        "input": prompt,
        "output": answer,
        "context": [precontext],
        "reference": str(correct_choice),
    }
    return result


def main():
    dataset = setup(opik_client)

    metrics = [SameFirstLetterMetric("Multiple-choice match")]
    global my_model

    if settings.closed_textbook_eval:
        for llm in settings.llm_evals_list:
            my_model = llm
            eval_results = evaluate(
                experiment_name="my_evaluation-closed:" + llm,
                dataset=dataset,
                task=evaluation_task_closed,
                scoring_metrics=metrics,
                task_threads=4,
            )

    if settings.open_textbook_eval:
        global textbook
        textbook = OpenTextBook()
        for llm in settings.llm_evals_list:
            my_model = llm
            eval_results = evaluate(
                experiment_name="my_evaluation-open:" + llm,
                dataset=dataset,
                task=evaluation_task_open,
                scoring_metrics=metrics,
                task_threads=4,
            )

    print(colored("Evaluation completed", "green"))


if __name__ == "__main__":
    main()
