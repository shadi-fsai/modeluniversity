from pydantic import BaseModel
from typing import List, Union


class TopicSchema(BaseModel):
    topic: str
    subtopics: List[str]


class CurriculumSchema(BaseModel):
    topics: List[TopicSchema]


class SingleQuestionSchema(BaseModel):
    question: str
    question_difficulty: str  # easy, medium, hard
    correct_answer: str
    explanation: str


class TrainQuestionsSchema(BaseModel):
    questions: List[SingleQuestionSchema]


class MultiAnswerQuestionSchema(BaseModel):
    question: str
    question_difficulty: str  # easy, medium, hard
    correct_answer: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str
    explanation: str


class TestQuestionsSchema(BaseModel):
    questions: List[MultiAnswerQuestionSchema]
