from time import sleep
from litellm import completion
from litellm import RateLimitError
import yaml
from termcolor import colored
import json
from pydantic import BaseModel
from typing import List

import dotenv
dotenv.load_dotenv()

config = None

class TopicSchema(BaseModel):
    topic: str
    subtopics: List[str]

class CurriculumSchema(BaseModel):
    topics: List[TopicSchema]

def generate_curriculum():
    try:
        with open('curriculum.json', 'r') as file:
            curriculum = json.load(file)
            print(colored("Loaded curriculum from curriculum.json", "green"))
    except FileNotFoundError:

        prompt = config['curriculum_prompt']
        response = completion(
                    model=config['datagen_model'],
                    messages=[
                        {"role": "system", "content": config['teacher_role']},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=CurriculumSchema,
                    temperature=0,
                    max_tokens=4096
                )
        
        curriculum_str = response['choices'][0]['message']['content']
        curriculum = json.loads(curriculum_str)
        with open('curriculum.json', 'w') as file:
            json.dump(curriculum, file, indent=4)    
            print(colored("Saved curriculum to curriculum.json", "green"))            

    return curriculum

class SingleQuestionSchema(BaseModel):
    question: str
    correct_answer: str
    explanation: str

class TrainQuestionsSchema(BaseModel):
    questions: List[SingleQuestionSchema]

class MultiAnswerQuestionSchema(BaseModel):
    question: str
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
                model=config['datagen_model'],
                messages=[
                    {"role": "system", "content": config['teacher_role'] },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4096,
                response_format=schema
            )
            return response['choices'][0]['message']['content']
        except RateLimitError as e:
            print(colored("Rate limit error. Waiting 5 seconds before retrying", "red"))
            sleep(5)
            retries += 1
    

training_questions_bank = []
testing_questions_bank = []

def generate_questions(curriculum):
    for topic in curriculum['topics']:
        for subtopic in topic['subtopics']:
            prompt = f"Create {config['num_practice_questions']} practice questions and answers for the topic {topic['topic']} subtopic: {subtopic}. {config['num_easy_practice_questions']} easy, {config['num_medium_practice_questions']} medium, {config['num_hard_practice_questions']} hard."
            training_questions = json.loads(question_prompt_call(prompt, TrainQuestionsSchema))
            print (colored(f"Training questions for subtopic: {subtopic}", "green"))

            with open('training_questions.json', 'a') as file:
                for question in training_questions['questions']:
                    training_questions_bank.append(json.dumps({
                        "topic": topic['topic'],
                        "subtopic": subtopic,
                        "question": question['question'],
                        "answer": question['correct_answer'],
                        "explanation": question['explanation']
                    }))
            
            questions_list = [q['question'] for q in training_questions['questions']]

            prompt = f"Create {config['num_test_questions']} multi-answer test questions for the topic {topic['topic']} subtopic: {subtopic}. {config['num_easy_test_questions']} easy, {config['num_medium_test_questions']} medium, {config['num_hard_test_questions']} hard. \n"+\
                "avoid repeating these exact questions, its ok to test the same concepts with different questions:" + str(questions_list)
            
            test_questions = json.loads(question_prompt_call(prompt, TestQuestionsSchema))
            print (colored(f"Test questions for subtopic: {subtopic}", "green"))

            with open('test_questions.json', 'a') as file:
                for question in test_questions['questions']:
                    testing_questions_bank.append(json.dumps({
                        "topic": topic['topic'],
                        "subtopic": subtopic,
                        "question": question['question'],
                        "answer": question['correct_answer'],
                        "wrong_answer1": question['wrong_answer1'],
                        "wrong_answer2": question['wrong_answer2'],
                        "wrong_answer3": question['wrong_answer3'],
                        "explanation": question['explanation']
                    }))

    with open('training_questions.json', 'w') as file:
        json.dump(training_questions_bank, file, indent=4)
        print(colored("Saved training questions to training_questions.json", "green"))

    with open('test_questions.json', 'w') as file:
        json.dump(testing_questions_bank, file, indent=4)
        print(colored("Saved test questions to test_questions.json", "green"))


def main():
    global config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file.read())
    curriculum = generate_curriculum()
    generate_questions(curriculum)


def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": config['student_role']},
      {"role": "user", "content": "Answer the following question and add an explanation in the format 'ANSWER. My explanation: EXPLANATION': " + str(sample["question"])},
      {"role": "assistant", "content": str(sample["answer"]) + ". My explanation: " + str(sample["explanation"])}
    ]
  }

def transform_to_trainable_json():
    with open("test_questions.json", "r") as file:
        data = json.load(file)
        conversations = []
        for item in data:
            item_json = json.loads(item)
            sample = {
                "question" : item_json['question'],
                "answer" : item_json['answer'],
                "explanation" : item_json['explanation']
                }
            conversation = create_conversation(sample)
            conversations.append(conversation)
        
        with open('trainable_data.json', 'w') as json_file:
            json.dump(conversations, json_file, indent=4)