# ModelUni

This project creates the training and testing/eval artifacts to train a small Llama model with.

It synthetically generates training and testing data using llama3.3 70B by creating a curriculum and varying difficulty questions.

The test questions are multiple-choice, evals are done by asking the small model to pick the right answer and comparing for equality.

## Installation

1. Clone the repository

2. Create your `.env` file by copying the `.env-example`:
```bash
cp .env-example .env
```
Then edit the `.env` file and replace the placeholder values with your actual API keys, like for example:
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

After we have clearly defined the topics on which we want to train a model, we need to create questions and answers both to train it and to test it.

To define the number of training and testing questions we want to generate we need to configure the `practice` and `test` parameters of our `config.yaml`.
The model selected to generate all these is defined by the `datagen_model` parameter of our `config.yaml`.

The files with the questions will be written by default in the main repo folder. We can define otherwise though:

```bash
# Default: Uses curriculum.json, outputs to training.json and testing.json
poetry run modeluni create-questions

# Example with custom paths:
poetry run modeluni create-questions --curriculum-file custom/input.json --training-file custom/train.json --testing-file custom/test.json
```

3. Transform the data for training:

Since we created a set of training questions, we only need to transform them in a format, specifically for [this colab notebook](https://colab.research.google.com/drive/12RH6ojAY_TFvQ02ZLvQdjFe944o0IIDQ) that's utilizing colab's computing power and unsloth's utilities for fine tuning:

```bash
# Default: Uses training_questions.json, outputs to trainable_data.json
poetry run modeluni transform-data

# Example with custom paths:
poetry run modeluni transform-data --training-file custom/train.json --output-file custom/final.json
```

4. Train 

Using the abovementioned [colab notebook](https://colab.research.google.com/drive/12RH6ojAY_TFvQ02ZLvQdjFe944o0IIDQ),  you pretty much throw your `trainable_data.json` and it will find your own data when running this line:

```bash
dataset = load_dataset("json", data_files="trainable_data.json", split = "train")
```

Where it will be used later at this part:

```bash
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
```

To create your own fine-tune, based on the synthetic data you just created.

5. Evaluate your model:

After you have your fine-tuned model ready, you want to make sure you did a good job on it.

You already have the `test_questions.json` (generated on the data generation stage, where you specified the  `--testing-file` to be, otherwise in your main repo's folder) that will help you benchmark your model against various other models (the `llm_evals_list` of your `config.yaml`) with a judge model that you will also define in the `config.yaml` (the `opik_eval_model`).

To be able to achieve this, you need to upload the questions (with their answers) on opik and monitor the results of the evaluations.

You can achieve this with a command like this:

    ```bash
    poetry run modeluni run-evals \
         --evaluation-dataset-name my_test_questions \
         --test-questions-file tests/data/test_questions.json
    ```
    - This creates a new dataset (e.g., "from_cli") in Opik, embeds your textbook for reference, and runs multiple-choice evaluations.
    - Check the Opik dashboard for detailed logs and metrics such as the "Multiple-choice match" score.
    - Upon completion, the console shows final results (e.g., "Evaluation completed. Results: [...]").

Optional: All commands accept the `--help` flag for detailed usage information:
```bash
poetry run modeluni <command> --help
```
