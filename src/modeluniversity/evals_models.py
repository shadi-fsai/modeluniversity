import json
from typing import Any
from opik.evaluation.metrics import base_metric, score_result
from litellm import BaseModel, completion
from .config import settings


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
