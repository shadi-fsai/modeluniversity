# ModelUni

ModelUni is a tool for generating synthetic training data and evaluating language models in educational contexts.

This project creates training, testing, and evaluation artifacts to train a small Llama model. It synthetically generates training and testing data using llama3.3 70B by creating a curriculum and questions of varying difficulty.

Test questions are multiple-choice; evaluations compare the model’s chosen answer for equality.

## Installation

1. Clone the repository

2. Create your `.env` file by copying the `.env-example`:
```bash
cp .env-example .env
```
Then edit the `.env` file and replace the placeholder values with your actual API keys, for example:
```
GROQ_API_KEY=your_groq_key
OPIK_API_KEY=your_opik_key
```

3. Install dependencies:
```bash
poetry lock
poetry install
```

4. Verify installation:
```bash
poetry run modeluni --help
```

## CLI Usage and Workflow

The CLI tool has three main commands that form the typical workflow:

1. Create a curriculum:
```bash
poetry run modeluni create-curriculum
```
This command generates a structured curriculum that defines topics and subtopics. By default, it tries to load from `curriculum.json` in the current directory. If the file doesn't exist, it creates a new curriculum based on the `curriculum_prompt` defined in your `config.yaml`. You can specify a custom path using the `--curriculum-file` option:
```bash
poetry run modeluni create-curriculum --curriculum-file custom/curriculum.json
```

2. Generate questions based on the curriculum:

After defining topics for the model, generate questions and answers for training and testing. Configure the `practice` and `test` parameters of your `config.yaml` to specify the number of questions, and set the `datagen_model` parameter to choose the model that creates them.

The files with the questions will be written by default to the main repo folder, but you can change that:

```bash
# Default: Uses curriculum.json, outputs to training.json and testing.json
poetry run modeluni create-questions

# Example with custom paths:
poetry run modeluni create-questions --curriculum-file custom/input.json --training-file custom/train.json --testing-file custom/test.json
```

3. Transform the data for training:

Since you created a set of training questions, transform them into a format suitable for [this colab notebook](https://colab.research.google.com/drive/12RH6ojAY_TFvQ02ZLvQdjFe944o0IIDQ):

```bash
# Default: Uses training_questions.json, outputs to trainable_data.json
poetry run modeluni transform-data

# Example with custom paths:
poetry run modeluni transform-data --training-file custom/train.json --output-file custom/final.json
```

4. Train

Using the referenced [colab notebook](https://colab.research.google.com/drive/12RH6ojAY_TFvQ02ZLvQdjFe944o0IIDQ), load `trainable_data.json`:

```bash
dataset = load_dataset("json", data_files="trainable_data.json", split="train")
```

Then it will be used with:

```bash
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
```

to create your fine-tuned model based on the synthetic data.

5. Evaluate your model:

After fine-tuning, use `test_questions.json` to benchmark your model and others (the `llm_evals_list` in `config.yaml`) with a judge model (defined as `opik_eval_model` in `config.yaml`). Upload questions and answers to Opik, then monitor evaluation results.

For example:
```bash
poetry run modeluni run-evals \
     --evaluation-dataset-name my_test_questions \
     --test-questions-file tests/data/test_questions.json
```
- This creates a new dataset (e.g., “from_cli”) in Opik, embeds your textbook for reference, then runs multiple-choice evaluations.
- Check the Opik dashboard for logs and metrics (e.g., “Multiple-choice match” score).
- Upon completion, results appear in the console.

Optional: All commands accept `--help` for more details:
```bash
poetry run modeluni <command> --help
```
