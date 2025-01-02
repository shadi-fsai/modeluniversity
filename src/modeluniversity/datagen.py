from time import sleep
from litellm import completion
from litellm import RateLimitError
from termcolor import colored
import json
from pydantic import BaseModel
from typing import List, Union
from .config import settings
from pathlib import Path
import argparse


class TopicSchema(BaseModel):
    topic: str
    subtopics: List[str]


class CurriculumSchema(BaseModel):
    topics: List[TopicSchema]


def generate_curriculum(
    curriculum_file_at: Path = Path("curriculum.json"),
):
    try:
        with open(curriculum_file_at, "r") as file:
            curriculum = json.load(file)
            print(colored(f"Loaded curriculum from {curriculum_file_at}", "green"))
    except FileNotFoundError:
        print(
            colored(
                f"""File: 
                {curriculum_file_at} 
                not found for curriculum. 
                Will attempt to create it.""",
                "green",
            )
        )

        prompt = settings.curriculum_prompt
        response = completion(
            model=settings.datagen_model,
            messages=[
                {"role": "system", "content": settings.teacher_role},
                {"role": "user", "content": prompt},
            ],
            response_format=CurriculumSchema,
            temperature=0,
            max_tokens=4096,
        )

        curriculum_str = response["choices"][0]["message"]["content"]
        curriculum = json.loads(curriculum_str)
        with open(curriculum_file_at, "w") as file:
            json.dump(curriculum, file, indent=4)
            print(colored("Saved curriculum to curriculum.json", "green"))

    return curriculum


class SingleQuestionSchema(BaseModel):
    question: str
    question_difficulty: str # easy, medium, hard
    correct_answer: str
    explanation: str


class TrainQuestionsSchema(BaseModel):
    questions: List[SingleQuestionSchema]


class MultiAnswerQuestionSchema(BaseModel):
    question: str
    question_difficulty: str # easy, medium, hard
    correct_answer: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str
    explanation: str


class TestQuestionsSchema(BaseModel):
    questions: List[MultiAnswerQuestionSchema]


def question_prompt_call(prompt, schema):
    retries = 0
    while retries < 5:
        try:
            response = completion(
                model=settings.datagen_model,
                messages=[
                    {"role": "system", "content": settings.teacher_role},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4096,
                response_format=schema,
            )
            return response["choices"][0]["message"]["content"]
        except RateLimitError as e:
            print(colored("Rate limit error. Waiting 5 seconds before retrying", "red"))
            sleep(5)
            retries += 1


def generate_questions(
    curriculum,
    training_questions_file: str = "training_questions.json",
    testing_questions_file: str = "test_questions.json",
):

    def create_base_prompt(questions_list_provided: Union[None, list] = None) -> str:
        purpose_is_practice: bool = questions_list_provided is None
        settings_subset = settings.practice if purpose_is_practice else settings.test
        purpose_part = (
            "practice questions and answers"
            if purpose_is_practice
            else "multi-answer test questions"
        )

        main_part = f"""Create {settings.practice.num_total} {purpose_part} for the topic: {topic['topic']}. Subtopic: {subtopic}."""
        amounts_per_difficulty_part = f"{settings_subset.num_easy} easy, {settings_subset.num_medium} medium, {settings_subset.num_hard} hard questions."

        additional_questions_prompt = (
            "Make sure you cover the most critical concepts, add more questions if you need to."
            if settings_subset.allow_expansion
            else ""
        )
        final_prompt = (
            f"{main_part} {amounts_per_difficulty_part} {additional_questions_prompt}"
        )
        if not purpose_is_practice:
            final_prompt += (
                "avoid repeating these exact questions, its ok to test the same concepts with different questions:"
                + str(questions_list_provided)
            )
        return final_prompt

    # Ensure files are initialized (clear old data if any)
    with open(training_questions_file, "w") as file:
        file.write("[]")
    with open(testing_questions_file, "w") as file:
        file.write("[]")

    for topic in curriculum["topics"]:
        for subtopic in topic["subtopics"]:
            training_prompt = create_base_prompt()

            training_questions = json.loads(
                question_prompt_call(training_prompt, TrainQuestionsSchema)
            )
            print(colored(f"Training questions for subtopic: {subtopic}", "green"))

            questions_list = []
            # Write training questions directly to file
            with open(training_questions_file, "r+") as file:
                data = json.load(file)
                for question in training_questions["questions"]:
                    entry = {
                        "topic": topic["topic"],
                        "subtopic": subtopic,
                        "question": question["question"],
                        "question_difficulty": question["question_difficulty"],
                        "answer": question["correct_answer"],
                        "explanation": question["explanation"],
                    }
                    data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4)

            # Prepare list of questions for testing
            questions_list = [q["question"] for q in training_questions["questions"]]

            testing_prompt = create_base_prompt(questions_list)

            test_questions = json.loads(
                question_prompt_call(testing_prompt, TestQuestionsSchema)
            )
            print(colored(f"Test questions for subtopic: {subtopic}", "green"))

            # Write test questions directly to file
            with open(testing_questions_file, "r+") as file:
                data = json.load(file)
                for question in test_questions["questions"]:
                    entry = {
                        "topic": topic["topic"],
                        "subtopic": subtopic,
                        "question": question["question"],
                        "question_difficulty": question["question_difficulty"],
                        "answer": question["correct_answer"],
                        "wrong_answer1": question["wrong_answer1"],
                        "wrong_answer2": question["wrong_answer2"],
                        "wrong_answer3": question["wrong_answer3"],
                        "explanation": question["explanation"],
                    }
                    data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4)


def main():
    curriculum = generate_curriculum()
    generate_questions(curriculum)


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": settings.student_role},
            {
                "role": "user",
                "content": "Answer the following question and add an explanation in the format 'ANSWER. My explanation: EXPLANATION': "
                + str(sample["question"]),
            },
            {
                "role": "assistant",
                "content": str(sample["answer"])
                + ". My explanation: "
                + str(sample["explanation"]),
            },
        ]
    }


def transform_to_trainable_json(
    training_questions_file: Path = Path("training_questions.json"),
    trainable_questions_file: Path = Path("trainable_data.json"),
):
    with training_questions_file.open("r") as file:
        data = json.load(file)
        conversations = []
        for item in data:
            item_json = json.loads(item) if not isinstance(item, dict) else item
            sample = {
                "question": item_json["question"],
                "answer": item_json["answer"],
                "explanation": item_json["explanation"],
            }
            conversation = create_conversation(sample)
            conversations.append(conversation)

        with trainable_questions_file.open("w") as json_file:
            json.dump(conversations, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script entry point for curriculum generation and transformation."
    )
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Run the transform_to_trainable_json function instead of main().",
    )
    parser.add_argument(
        "--training_questions_file",
        type=Path,
        default=Path("training_questions.json"),
        help="Path to the training questions JSON file (default: training_questions.json).",
    )
    parser.add_argument(
        "--trainable_questions_file",
        type=Path,
        default=Path("trainable_data.json"),
        help="Path to the output trainable JSON file (default: trainable_data.json).",
    )

    args = parser.parse_args()

    if args.transform:
        transform_to_trainable_json(
            training_questions_file=args.training_questions_file,
            trainable_questions_file=args.trainable_questions_file,
        )
    else:
        main()
