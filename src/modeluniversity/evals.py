import json
from time import sleep
import yaml
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Equals

from litellm import RateLimitError
from litellm import completion
from termcolor import colored

import dotenv
import random

config = None

dotenv.load_dotenv()

opik_client = Opik()
my_model = ""

def setup(client):
    # Create a dataset
    dataset = client.get_or_create_dataset(name=config['opik_dataset'])
    # Load the json file test_questions.json
    # Upload each [topic,subtopic,question,answer] to the dataset
    with open("test_questions.json", "r") as file:
        data = json.load(file)
        for item in data:
            item_json = json.loads(item) 
            dataset.insert([{"topic": item_json['topic'], "subtopic": item_json['subtopic'],
                              "question": item_json['question'], "answer": item_json['answer'],
                              "wrong_answer1": item_json['wrong_answer1'], "wrong_answer2": item_json['wrong_answer2'],
                              "wrong_answer3": item_json['wrong_answer3'],
                              "explanation": item_json['explanation']}])
    return dataset


def question_prompt_call(prompt):
    retries = 0
    while retries < 5:
        try:
            response = completion(
                model=my_model,
                messages=[
                    {"role": "system", "content": config['student_role']},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1
            )

            return response['choices'][0]['message']['content'] 
        except RateLimitError as e:
            print(colored("Rate limit error. Waiting 30 seconds before retrying", "red"))
            sleep(30)
            retries += 1    

def create_answer_choices(correct_answer, wrong_answer1, wrong_answer2, wrong_answer3):
    correct_choice = random.choice(['A', 'B', 'C', 'D'])
    wrong_choices = [choice for choice in ['A', 'B', 'C', 'D'] if choice != correct_choice]
    correct_choice_answer = f"{correct_choice}. {correct_answer}"
    wrong_choice1_answer = f"{wrong_choices.pop()}. {wrong_answer1}"
    wrong_choice2_answer = f"{wrong_choices.pop()}. {wrong_answer2}"
    wrong_choice3_answer = f"{wrong_choices.pop()}. {wrong_answer3}"
    answer_choices = [correct_choice_answer, wrong_choice1_answer, wrong_choice2_answer, wrong_choice3_answer]
    answer_choices.sort(key=lambda x: x[0])
    multiline = "\n".join(answer_choices)
    return (correct_choice, multiline)


def evaluation_task(dataset_item):
    # your LLM application is called here
    input = dataset_item["question"]
    print(colored("Evaluating question: " + input, "green"))
    (correct_choice, answer_choices) = create_answer_choices(dataset_item["answer"], dataset_item["wrong_answer1"], dataset_item["wrong_answer2"], dataset_item["wrong_answer3"])
    precontext = config['student_role']
    prompt = "What is the correct answer for the following multi-choice(A/B/C/D) question:" + input + "\n" + answer_choices + "\n Return one letter only as the answer. For example: A\n" 
    answer = question_prompt_call(prompt)
    result = {
        "input": prompt,
        "output": answer,
        "context": [precontext],
        "reference": str(correct_choice)
    }
    return result

def load_config():
    global config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file.read())

def main():
    load_config()
    dataset = setup(opik_client)

    metrics = [Equals()]
    global my_model
    for llm in config['llm_evals_list']:
        my_model = llm
        eval_results = evaluate(
            experiment_name="my_evaluation:" + llm,
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=metrics,
            task_threads=4,
        )
    print (colored("Evaluation completed", "green"))
