# ============================================
# Deep Learning Project: Training and Fine-Tuning a Language Model with Unsloth
# ============================================

# ============================================
# 1. Installing and Upgrading Required Libraries
# ============================================

# It's recommended to use a virtual environment to manage dependencies.
# You can create one using:
# python -m venv unsloth_env
# source unsloth_env/bin/activate  # On Windows: unsloth_env\Scripts\activate

# Install the Unsloth library and other dependencies
import sys
import subprocess

def install_packages():
    """Install necessary packages using pip."""
    packages = [
        "unsloth",
        "torch",
        "datasets",
        "scikit-learn",
        "optuna",
        "wandb",
        "transformers",
        "trl",
        "nltk",
        "spacy",
        "deepdiff",
        "tensorflow"  # Required by some monitoring tools
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment the following line to install packages
# install_packages()

# After installation, it's good practice to upgrade pip
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# ============================================
# 2. Importing Necessary Modules
# ============================================

import os
import json
import re
import random
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

import optuna
import wandb

# Optional: Import for advanced data augmentation
import nltk
from nltk.corpus import wordnet

# Initialize NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# --------------------------------------------
# Explanation:
# - **FastLanguageModel:** Handles language models efficiently from the Unsloth library.
# - **torch:** PyTorch library for deep learning operations.
# - **datasets:** Library to load and manage datasets.
# - **sklearn.model_selection:** For splitting datasets into training and validation sets.
# - **optuna:** For hyperparameter optimization.
# - **wandb:** Weights & Biases for performance monitoring and logging.
# - **nltk & wordnet:** For advanced data augmentation techniques.
#
# Tips:
# - **Modular Code:** Import all necessary modules at the beginning to enhance readability and simplify debugging.
# - **Lazy Imports:** For large libraries that are not always used, consider importing them only when needed to save memory.
# --------------------------------------------

# ============================================
# 3. Setting Configuration Parameters
# ============================================

# Configuration settings for training
CONFIG = {
    "max_seq_length": 2048,             # Maximum sequence length the model can process
    "load_in_4bit": True,               # Enables 4-bit loading for efficient memory usage
    "dtype": None,                      # Data type configuration (currently set to None)
    "checkpoint_dir": "/content/drive/MyDrive/Defense/outputs_Meta-Llama-3.1-8B-bnb-4bit",  # Directory path to store model checkpoints
    "augmentation_factor": 3,           # Number of augmented samples to create per original sample
    "batch_size": 4,                    # Batch size per device
    "gradient_accumulation_steps": 8,   # Number of steps to accumulate gradients
    "learning_rate": 3e-4,              # Learning rate for the optimizer
    "max_steps": 1000,                   # Total number of training steps
    "save_steps": 500,                   # Frequency of saving checkpoints
    "save_total_limit": 2,               # Maximum number of checkpoints to save
    "seed": 42,                          # Random seed for reproducibility
    "project_name": "Unsloth_Fine_Tuning"  # Weights & Biases project name
}

# Ensure checkpoint directory exists
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# Set random seeds for reproducibility
random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# ============================================
# 4. Setting Up Logging
# ============================================

# Configure logging
logging.basicConfig(
    filename=os.path.join(CONFIG["checkpoint_dir"], "training.log"),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger()

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

logger.info("Logging is set up.")

# --------------------------------------------
# Explanation:
# - **Logging:** Captures detailed information about the training process, which is essential for debugging and monitoring.
#
# Tips:
# - **Dual Logging:** Logging to both a file and the console ensures that you can review logs later and monitor progress in real-time.
# - **Log Levels:** Use appropriate log levels (INFO, DEBUG, WARNING, ERROR) to categorize log messages.
# --------------------------------------------

# ============================================
# 5. Checkpoint Management Functions
# ============================================

def is_model_processed(model_name):
    """
    Checks if a checkpoint exists for a specific model.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if the checkpoint exists, False otherwise.
    """
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], model_name.split("/")[-1], "checkpoint-500", "trainer_state.json")
    logger.info(f"Checking checkpoint path: {checkpoint_path}")
    return os.path.exists(checkpoint_path)

def mark_model_as_processed(model_name):
    """
    Marks a model as processed by creating a '.done' file.

    Args:
        model_name (str): The name of the model.
    """
    checkpoint_file = os.path.join(CONFIG["checkpoint_dir"], f"{model_name.replace('/', '_')}.done")
    logger.info(f"Marking model as processed: {checkpoint_file}")
    with open(checkpoint_file, 'w') as f:
        f.write("")

# --------------------------------------------
# Explanation:
# - **is_model_processed:** Checks if a specific model has already been processed by looking for an existing checkpoint.
# - **mark_model_as_processed:** Creates a marker file indicating that the model has been successfully processed and saved.
#
# Tips:
# - **State Tracking:** Using checkpoint files helps in resuming training seamlessly in case of interruptions.
# - **File Naming:** Replace special characters in model names to prevent issues with file paths.
# - **Logging:** Ensure all checkpoint operations are logged for better traceability.
# --------------------------------------------

# ============================================
# 6. Advanced Data Preprocessing and Augmentation
# ============================================

def advanced_augment_text(text):
    """
    Create variations of text for augmentation using WordNet synonyms.

    Args:
        text (str): The original text.

    Returns:
        str: The augmented text.
    """
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        if lemmas:
            new_word = random.choice(list(lemmas)).replace('_', ' ')
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def preprocess_dataset(input_path, output_path, train_path, val_path, augmentation_factor=CONFIG["augmentation_factor"]):
    """
    Preprocesses, validates, and augments the dataset.

    Args:
        input_path (str): Path to the original dataset.
        output_path (str): Path to save the cleaned and augmented dataset.
        train_path (str): Path to save the training dataset.
        val_path (str): Path to save the validation dataset.
        augmentation_factor (int): Number of augmented samples to create per original sample.
    """
    logger.info("Starting dataset preprocessing, validation, and augmentation.")
    valid_entries = 0

    def clean_text(text):
        """Normalize and clean text."""
        # Remove unwanted characters
        text = re.sub(r"[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ.,!?\\-]", " ", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        # Convert to lowercase
        return text.lower()

    def augment_text(text):
        """Create variations of text for augmentation."""
        synonyms = {
            "highlight": ["emphasize", "focus on", "spotlight"],
            "identify": ["detect", "recognize", "pinpoint"],
            "discuss": ["elaborate on", "examine", "analyze"],
            "important": ["crucial", "key", "essential"]
        }
        for word, replacements in synonyms.items():
            if word in text:
                text = text.replace(word, random.choice(replacements))
        return text

    augmented_data = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                if 'instruction' in data and 'input' in data and 'output' in data:
                    # Clean the data
                    cleaned_data = {
                        "instruction": clean_text(data.get("instruction", "")),
                        "input": clean_text(data.get("input", "")),
                        "output": clean_text(data.get("output", ""))
                    }
                    augmented_data.append(cleaned_data)
                    valid_entries += 1

                    # Augment the data
                    for _ in range(augmentation_factor):
                        augmented_entry = {
                            "instruction": augment_text(cleaned_data['instruction']),
                            "input": augment_text(cleaned_data['input']),
                            "output": augment_text(cleaned_data['output'])
                        }
                        # Optionally apply advanced augmentation
                        if random.random() < 0.5:  # 50% chance to apply advanced augmentation
                            augmented_entry = {
                                "instruction": advanced_augment_text(augmented_entry['instruction']),
                                "input": advanced_augment_text(augmented_entry['input']),
                                "output": advanced_augment_text(augmented_entry['output'])
                            }
                        augmented_data.append(augmented_entry)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line.")

    logger.info(f"Dataset preprocessing complete. Valid entries: {valid_entries}")

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(augmented_data, test_size=0.2, random_state=CONFIG["seed"])

    # Save the cleaned and augmented dataset
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in augmented_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logger.info(f"Enhanced dataset saved to {output_path}.")

    # Save the training set
    with open(train_path, 'w', encoding='utf-8') as trainfile:
        for entry in train_data:
            trainfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logger.info(f"Training set saved to {train_path}.")

    # Save the validation set
    with open(val_path, 'w', encoding='utf-8') as valfile:
        for entry in val_data:
            valfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logger.info(f"Validation set saved to {val_path}.")

    # --------------------------------------------
    # Explanation:
    # - **clean_text:** Cleans and normalizes text by removing unwanted characters, extra spaces, and converting text to lowercase.
    # - **augment_text:** Enhances the dataset by replacing specific words with their synonyms to create variations.
    # - **advanced_augment_text:** Uses WordNet to perform more sophisticated synonym replacements, increasing data diversity.
    # - **preprocess_dataset:** Reads the input dataset, cleans and augments the data, splits it into training and validation sets, and saves the processed data.
    #
    # Tips:
    # - **Data Cleaning:** Adjust regular expressions to accommodate different languages or specific dataset requirements.
    # - **Data Augmentation:** Incorporate more sophisticated augmentation techniques, such as paraphrasing or back-translation, to increase data diversity.
    # - **Error Handling:** Enhance error logging to capture more details about problematic data entries.
    # - **Scalability:** For very large datasets, consider using data processing libraries like Pandas or Dask to handle data more efficiently.
    # - **Advanced Augmentation:** Use libraries like `nltk` or `spacy` for more advanced text manipulation and augmentation.
    # --------------------------------------------

# ============================================
# 7. Listing Models for Fine-Tuning
# ============================================

# List of models to try for fine-tuning
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
]

# --------------------------------------------
# Explanation:
# This list contains various **4-bit** models from Unsloth that will be fine-tuned.
# The models vary in size and configuration, optimizing them for different tasks.
#
# Tips:
# - **Model Selection:** Choose models based on the specific requirements of your task, balancing between performance and computational resources.
# - **Diversity:** Experiment with different architectures to identify which performs best for your use case.
# - **Documentation:** Refer to the Unsloth library documentation for detailed information about each model's capabilities and intended use cases.
# --------------------------------------------

# ============================================
# 8. Initializing Weights & Biases for Experiment Tracking
# ============================================

# Initialize Weights & Biases (W&B)
def init_wandb(project_name):
    """
    Initializes Weights & Biases for experiment tracking.

    Args:
        project_name (str): The name of the W&B project.
    """
    wandb.login()  # Ensure you are logged in to W&B
    wandb.init(
        project=project_name,
        config=CONFIG,
        name=f"Fine-Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger.info("Weights & Biases initialized.")

# Initialize W&B
init_wandb(CONFIG["project_name"])

# --------------------------------------------
# Explanation:
# - **Weights & Biases (W&B):** A tool for experiment tracking, model monitoring, and collaboration.
# - **init_wandb:** Function to initialize W&B with the project name and configuration.
#
# Tips:
# - **Logging Hyperparameters:** W&B automatically logs hyperparameters, making it easier to track experiments.
# - **Visualizations:** Utilize W&B's visualization tools to monitor training metrics in real-time.
# - **Version Control:** Use W&B to keep track of different runs and compare their performances.
# --------------------------------------------

# ============================================
# 9. Loading and Testing Models Sequentially with Fine-Tuning
# ============================================

# Function to format prompts for the model
def format_prompts(instruction, input_text, response):
    """
    Formats the prompts using the defined template.

    Args:
        instruction (str): The instruction text.
        input_text (str): The input text providing context.
        response (str): The expected response.

    Returns:
        str: The formatted prompt.
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    return alpaca_prompt.format(instruction, input_text, response) + tokenizer.eos_token

# Function to perform hyperparameter optimization using Optuna
def objective(trial, model_name, train_dataset, val_dataset):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        model_name (str): The name of the model.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.

    Returns:
        float: The validation loss.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 16)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

    # Configure TrainingArguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=CONFIG["max_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=CONFIG["seed"],
        output_dir=os.path.join(CONFIG["checkpoint_dir"], model_name.split("/")[-1]),
        report_to="wandb",
        disable_tqdm=True  # Disable tqdm to reduce noise in Optuna
    )

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Start training
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", None)

    # Report the validation loss to Optuna
    trial.report(val_loss, step=CONFIG["max_steps"])

    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss

# Iterate through each model in the list, load it, and fine-tune
for model_name in fourbit_models:
    logger.info(f"Processing model: {model_name}")
    
    # Define the model directory based on the checkpoint directory and model name
    model_dir = os.path.join(CONFIG["checkpoint_dir"], model_name.split("/")[-1])
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Model directory: {model_dir}")

    # Define the path to the specific checkpoint file
    checkpoint_path = os.path.join(model_dir, "checkpoint-500", "trainer_state.json")
    logger.info(f"Checkpoint path: {checkpoint_path}")

    if is_model_processed(model_name):
        # If the model is already processed, skip to the next
        logger.info(f"Model {model_name} has already been processed. Skipping.")
        continue

    if os.path.exists(checkpoint_path):
        # If a checkpoint exists, resume training from the checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            os.path.dirname(checkpoint_path),
            max_seq_length=CONFIG["max_seq_length"],
            dtype=CONFIG["dtype"],
            load_in_4bit=CONFIG["load_in_4bit"],
        )
    else:
        # If no checkpoint exists, start training from scratch
        logger.info(f"Starting training for model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=CONFIG["max_seq_length"],
            dtype=CONFIG["dtype"],
            load_in_4bit=CONFIG["load_in_4bit"],
        )

    logger.info(f"Loaded model: {model_name}")

    # Apply Low-Rank Adaptation (LoRA) to the model for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank value; lower values save memory
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # Layers to adapt
        lora_alpha=16,  # Scaling factor for the learning rate
        lora_dropout=0.1,  # Dropout rate for regularization
        bias="none",  # Bias configuration
        use_gradient_checkpointing="unsloth",  # Enables gradient checkpointing for memory efficiency
        random_state=CONFIG["seed"],  # Ensures reproducibility by setting a random seed
    )
    logger.info(f"LoRA configuration done for model: {model_name}")

    # Load and format the training and validation datasets
    train_dataset = load_dataset("json", data_files=CONFIG["train_dataset_path"], split="train")
    val_dataset = load_dataset("json", data_files=CONFIG["val_dataset_path"], split="train")

    # Define the prompt formatting function
    def formatting_prompts_func(examples):
        """Formats dataset examples using the defined prompt template."""
        instructions = examples.get("instruction", "")
        inputs = examples.get("input", "")
        outputs = examples.get("output", "")
        texts = [format_prompts(instr, inp, outp) for instr, inp, outp in zip(instructions, inputs, outputs)]
        return {"text": texts}

    # Apply the formatting function to the datasets
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
    logger.info("Datasets loaded and formatted.")

    # Initialize Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize", study_name=f"FineTune_{model_name.split('/')[-1]}", storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, model_name, train_dataset, val_dataset), n_trials=10)

    # Best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters for {model_name}: {best_params}")

    # Configure TrainingArguments with the best hyperparameters
    training_args = TrainingArguments(
        per_device_train_batch_size=best_params["batch_size"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        warmup_steps=50,
        max_steps=CONFIG["max_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        learning_rate=best_params["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=best_params["weight_decay"],
        lr_scheduler_type="linear",
        seed=CONFIG["seed"],
        output_dir=model_dir,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        run_name=f"FineTune_{model_name.split('/')[-1]}"
    )

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=4,          # Increased number of processes for faster data loading
        packing=False,
        args=training_args,
    )

    # Implement Early Stopping
    from transformers import EarlyStoppingCallback

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

    # Start training with early stopping
    logger.info("Starting training...")
    trainer.add_callback(early_stopping)
    trainer.train(resume_from_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None)
    logger.info("Training completed.")

    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results for {model_name}: {eval_results}")

    # Save the fine-tuned model and tokenizer
    logger.info(f"Saving model to: {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Model {model_name} fine-tuned and saved to {model_dir}")

    # Mark the model as processed
    mark_model_as_processed(model_name)
    logger.info("----------------------------------------")

    # ============================================
    # 10. Finalizing Weights & Biases Run
    # ============================================

    wandb.finish()
    logger.info("Weights & Biases run finished.")

    # --------------------------------------------
    # Explanation:
    # - **Optuna:** Used for hyperparameter optimization to find the best combination of parameters.
    # - **Weights & Biases (W&B):** Integrated for experiment tracking and performance monitoring.
    # - **Early Stopping:** Prevents overfitting by stopping training when validation performance stops improving.
    # - **Advanced Data Augmentation:** Utilizes WordNet synonyms for more diverse data augmentation.
    #
    # Tips:
    # - **Hyperparameter Optimization:** Optuna's `n_trials` can be adjusted based on computational resources.
    # - **Experiment Tracking:** W&B provides detailed insights into model training, including metrics, hyperparameters, and model checkpoints.
    # - **Scalability:** Adjust `dataset_num_proc` based on the number of CPU cores to optimize data loading speed.
    # - **Reproducibility:** Setting seeds and logging configurations ensures that experiments can be reproduced reliably.
    # --------------------------------------------

# ============================================
# End of Script
# ============================================

# ============================================
# Additional Recommendations:
# ============================================

# - **Documentation:** Maintain thorough documentation of your training processes, configurations, and results to ensure reproducibility.
# - **Backup Checkpoints:** Regularly back up your checkpoints and final models to prevent data loss.
# - **Security:** Ensure that sensitive data is handled securely, especially when using cloud storage solutions.
# - **Scalability:** Design your training pipeline to be scalable, allowing you to handle larger datasets and more complex models as needed.
# - **Community Engagement:** Engage with the deep learning community to stay updated on best practices, new tools, and emerging trends.
# - **Continuous Integration:** Implement CI/CD pipelines to automate testing and deployment of models.
# - **Monitoring:** Continuously monitor model performance in production to detect and address issues promptly.
# - **Version Control:** Use Git or other version control systems to track changes in your codebase and configurations.

# ============================================
# Final Notes:
# ============================================

# This enhanced script serves as a robust foundation for training and fine-tuning language models using the Unsloth library. By integrating advanced features like logging, hyperparameter optimization, experiment tracking, and early stopping, it ensures an efficient and scalable deep learning workflow. Feel free to customize and expand upon this script to suit your specific project needs and objectives.
