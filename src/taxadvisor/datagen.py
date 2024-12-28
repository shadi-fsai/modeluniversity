from time import sleep
from litellm import completion
from litellm import RateLimitError
from termcolor import colored
import json
from pydantic import BaseModel
from typing import List

import dotenv
dotenv.load_dotenv()

my_model = "groq/Llama-3.3-70b-Versatile"
my_api_base = None 

class TopicSchema(BaseModel):
    topic: str
    subtopics: List[str]

class CirriculumSchema(BaseModel):
    topics: List[TopicSchema]

def generate_cirriculum():
    try:
        with open('cirriculum.json', 'r') as file:
            cirriculum = json.load(file)
            print(colored("Loaded cirriculum from cirriculum.json", "green"))
    except FileNotFoundError:

        prompt = "Create a verythorough list of tax topics a tax preparer must understand to be a good tax preparer. For each topic create a very thorough list of subtopics that a tax preparer must understand. Do not omit anything."
        response = completion(
                    model=my_model,
                    messages=[
                        {"role": "system", "content": "You are a teacher of tax preparers."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=CirriculumSchema,
                    temperature=0,
                    max_tokens=4096
                )
        
        cirriculum_str = response['choices'][0]['message']['content']
        cirriculum = json.loads(cirriculum_str)
        with open('cirriculum.json', 'w') as file:
            json.dump(cirriculum, file, indent=4)    
            print(colored("Saved cirriculum to cirriculum.json", "green"))            

    return cirriculum

class SingleQuestionSchema(BaseModel):
    question: str
    correct_answer: str
    explanation: str

class QuestionsSchema(BaseModel):
    questions: List[SingleQuestionSchema]

def question_prompt_call(prompt):
    retries = 0
    while retries < 5:
        try:
            response = completion(
                model=my_model,
                messages=[
                    {"role": "system", "content": "You are a teacher of tax preparers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4096,
                response_format=QuestionsSchema
            )
            return response['choices'][0]['message']['content']
        except RateLimitError as e:
            print(colored("Rate limit error. Waiting 5 seconds before retrying", "red"))
            sleep(5)
            retries += 1
    

training_questions_bank = []
testing_questions_bank = []

def generate_questions(cirriculum):
    for topic in cirriculum['topics']:
        for subtopic in topic['subtopics']:
            prompt = f"Create 10 practice questions and answers for the topic {topic['topic']} subtopic: {subtopic}. 4 easy, 4 medium, 2 hard."
            training_questions = json.loads(question_prompt_call(prompt))
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

            prompt = f"Create 10 test questions for  the topic {topic['topic']} subtopic: {subtopic}. 2 easy, 4 medium, 4 hard. \n"+\
                "avoid repeating any of these questions:" + str(questions_list)
            
            test_questions = json.loads(question_prompt_call(prompt))
            print (colored(f"Test questions for subtopic: {subtopic}", "green"))

            with open('test_questions.json', 'a') as file:
                for question in test_questions['questions']:
                    testing_questions_bank.append(json.dumps({
                        "topic": topic['topic'],
                        "subtopic": subtopic,
                        "question": question['question'],
                        "answer": question['correct_answer'],
                        "explanation": question['explanation']
                    }))

    with open('training_questions.json', 'w') as file:
        json.dump(training_questions_bank, file, indent=4)
        print(colored("Saved training questions to training_questions.json", "green"))

    with open('test_questions.json', 'w') as file:
        json.dump(testing_questions_bank, file, indent=4)
        print(colored("Saved test questions to test_questions.json", "green"))


def main():
    cirriculum = generate_cirriculum()
    generate_questions(cirriculum)

def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": "You are a tax preparer."},
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