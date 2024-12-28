import json
from time import sleep
import pydantic
from opik import Opik
from opik.evaluation import evaluate
import csv
from opik.evaluation.metrics import (LevenshteinRatio, IsJson)
from opik.evaluation.metrics import base_metric, score_result

from typing import Any

from pydantic import BaseModel
from litellm import RateLimitError
from termcolor import colored
from typing import List
from litellm import completion

import dotenv

my_eval_model = "gpt-4o-mini" # "groq/Llama-3.3-70b-Versatile"

class LLMJudgeScoreFormat(pydantic.BaseModel):
    score: int
    reason: str

class LLMJudgeMetric(base_metric.BaseMetric):
    def __init__(self, name: str = "Accurate Response"):
        self.name = name
        self.prompt_template = """
        TASK:
        You are an expert judge tasked with evaluating the accuracy of a given AI response.
        In the provided text, the OUTPUT must be the same answer as REFERENCE.
        If the answers are saying the same thing but written differently, the score should be 1.
        If the answers are very different concepts, the score should be 0.
        If the answer in OUTPUT is conceptually a subset of REFERENCE, the score should be 0.5.
        If the answer in REFERENCE is conceptually a subset of OUTPUT, the score should be 0.5.
        
        Your answer should include a score and a reason for the score.

        OUTPUT: {output}
        REFERENCE: {reference}
        """

    def score(self, output: str, reference:str, **ignored_kwargs: Any):
        """
        Score the output a of an LLM.

        Args:
            output: The output of an LLM to score.
            reference: Text that the output should be compared against.
            **ignored_kwargs: Any additional keyword arguments. This is important so that the metric can be used in the `evaluate` function.
        """
        # Construct the prompt based on the output of the LLM
        prompt = self.prompt_template.format(output=output, reference=reference)

        # Generate and parse the response from the LLM
        response = completion(
            model=my_eval_model, 
            messages=[{"role": "user", "content": prompt}],
            response_format=LLMJudgeScoreFormat,
        )

        final_score = float(json.loads(response.choices[0].message.content)["score"])
        reason = json.loads(response.choices[0].message.content)["reason"]
            # Return the score and the reason
        return score_result.ScoreResult(
            name=self.name, value=final_score, reason=reason
        )
 
dotenv.load_dotenv()

opik_client = Opik()
my_model = ""


def setup(client):
    # Create a dataset
    dataset = client.get_or_create_dataset(name="Tax Advisor Dataset")
    # Load the json file test_questions.json
    # Upload each [topic,subtopic,question,answer] to the dataset
    with open("test_questions.json", "r") as file:
        data = json.load(file)
        for item in data:
            item_json = json.loads(item)
            dataset.insert([{"topic": item_json['topic'], "subtopic": item_json['subtopic'],
                              "question": item_json['question'], "answer": item_json['answer'],
                              "explanation": item_json['explanation']}])
    return dataset



def question_prompt_call(prompt):
    retries = 0
    while retries < 5:
        try:
            response = completion(
                model=my_model,
                messages=[
                    {"role": "system", "content": "You are a tax preparer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4096
            )

            return response['choices'][0]['message']['content'] 
        except RateLimitError as e:
            print(colored("Rate limit error. Waiting 30 seconds before retrying", "red"))
            sleep(30)
            retries += 1    

def evaluation_task(dataset_item):
    # your LLM application is called here
    input = dataset_item["question"]
    print(colored("Evaluating question: " + input, "green"))
    precontext = "You are a tax preparer."    
    answer = question_prompt_call("Answer the following question:" + input)
    result = {
        "input": input,
        "output": answer,
        "context": [precontext],
        "reference": str(dataset_item["answer"])
    }
    return result

def main():
    dataset = setup(opik_client)

    metrics = [LLMJudgeMetric()]
    global my_model
    for llm in [
                "ollama/hf.co/shadicopty/Llama3.2-1b-taxadvisor",
                "groq/llama-3.2-1b-preview",
                "groq/Llama-3.3-70b-Versatile",
                "gpt-4o-mini",
                "gemini/gemini-1.5-flash",
                ]:
        my_model = llm
        eval_results = evaluate(
            experiment_name="my_evaluation:" + llm,
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=metrics,
            task_threads=4,
        )
        print(eval_results)