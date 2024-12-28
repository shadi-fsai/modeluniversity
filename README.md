# Tax Advisor

This project creates the training and testing/eval artifacts for a tax-advisor.

It synthetically generates training and testing data using llama3.3 70B by creating a cirriculum and varying difficulty questions.

Evaluation is done using LLM-judge which checks for conceptual matching of answers by the large model and the small trained model

You will need to add your groq / openai keys to .env; you will also need your own api keys for Opik which serves as the evals system.

To run start with 'poetry lock' / 'poetry install'

'poetry run datagen' generates the training/test data
'poetry run transform_to_trainable_json' prepares the data for unsloth training
'poetry run evals' runs the evaluations

Use this to train your 3.2 1b model and push it to huggingface: https://colab.research.google.com/drive/12RH6ojAY_TFvQ02ZLvQdjFe944o0IIDQ#scrollTo=FqfebeAdT073

Update src/taxadvisor/evals.py to pick the models you want to evaluate against (and update your.env accordingly)