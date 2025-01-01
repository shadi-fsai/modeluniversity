# ModelUni

This project creates the training and testing/eval artifacts to train a small Llama model with.

It synthetically generates training and testing data using llama3.3 70B by creating a cirriculum and varying difficulty questions.

The test questions are multiple-choice, evals are done by asking the small model to pick the right answer and comparing for equality.

## Installation

You will need to add your groq keys to .env.
You will also need your own api keys for Opik which serves as the evals system.

To run start with 'poetry lock' / 'poetry install'

Edit config.yaml to choose the topic you want to train for, its currently set on tax preparation capabilities

## Data generation and Evals

1. 'poetry run datagen' generates the training/test data

2. 'poetry run transform_to_trainable_json' prepares the data for unsloth training

3. Use this to train your 3.2 1b model and push it to huggingface: https://colab.research.google.com/drive/12RH6ojAY_TFvQ02ZLvQdjFe944o0IIDQ

4. Add your model to the list of evaluated llms in config.yaml; if you're running on ollama you'll want to download it to your desktop first (Ollama run hf.co/username/model)

5. 'poetry run evals' runs the evaluations
