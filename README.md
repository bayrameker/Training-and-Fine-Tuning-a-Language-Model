# Deep Learning Project: Training and Fine-Tuning a Language Model with Unsloth

## Table of Contents
1. [Overview](#overview)
2. [Project Setup](#project-setup)
   - [Prerequisites](#prerequisites)
   - [Installing Dependencies](#installing-dependencies)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Fine-Tuning Process](#fine-tuning-process)
   - [Model Selection](#model-selection)
   - [Hyperparameter Optimization](#hyperparameter-optimization)
   - [Training with Early Stopping](#training-with-early-stopping)
6. [Experiment Tracking](#experiment-tracking)
7. [Checkpoint Management](#checkpoint-management)
8. [Advanced Features](#advanced-features)
   - [Logging](#logging)
   - [Data Augmentation](#data-augmentation)
9. [Final Notes](#final-notes)

---

## Overview
This project demonstrates training and fine-tuning a language model using the **Unsloth** library. It incorporates advanced features such as logging, hyperparameter optimization with **Optuna**, experiment tracking using **Weights & Biases (W&B)**, and early stopping to improve performance and efficiency.

---

## Project Setup

### Prerequisites
- Python 3.8 or higher
- GPU-enabled environment (recommended for faster training)
- Access to **Weights & Biases** account (optional for experiment tracking)

### Installing Dependencies
1. Clone the repository:
   ```bash
   git clone https://github.com/bayrameker/Training-and-Fine-Tuning-a-Language-Model.git
   cd unsloth-fine-tuning
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv unsloth_env
   source unsloth_env/bin/activate  # On Windows: unsloth_env\Scripts\activate
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Upgrade `pip` (optional but recommended):
   ```bash
   pip install --upgrade pip
   ```

---

## Configuration
Edit the `CONFIG` dictionary in the script to adjust settings such as:
- **Model Parameters**: `max_seq_length`, `batch_size`, `learning_rate`, etc.
- **Paths**: `checkpoint_dir`, `train_dataset_path`, `val_dataset_path`.
- **Logging Settings**: Log files are saved in the checkpoint directory.

---

## Data Preparation
The script provides functions for preprocessing, cleaning, and augmenting datasets.
1. Place your dataset in JSON format.
2. Use the `preprocess_dataset` function to clean and augment the data.
   ```python
   preprocess_dataset(input_path="data/input.json", 
                      output_path="data/processed.json", 
                      train_path="data/train.json", 
                      val_path="data/val.json")
   ```

---

## Fine-Tuning Process

### Model Selection
A list of pre-trained models from **Unsloth** is included in the script. These models can be fine-tuned sequentially.

### Hyperparameter Optimization
The script uses **Optuna** for hyperparameter tuning. You can customize the number of trials:
```python
study.optimize(lambda trial: objective(trial, model_name, train_dataset, val_dataset), n_trials=10)
```

### Training with Early Stopping
Training includes early stopping to prevent overfitting. The patience level and threshold can be adjusted as follows:
```python
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
```

---

## Experiment Tracking
Enable **Weights & Biases (W&B)** to track your experiments:
1. Log in to your W&B account:
   ```bash
   wandb login
   ```
2. Initialize tracking in the script:
   ```python
   init_wandb(CONFIG["project_name"])
   ```
3. Monitor your experiments on the W&B dashboard.

---

## Checkpoint Management
Functions are included to manage checkpoints:
- **`is_model_processed`**: Checks if a model has already been fine-tuned.
- **`mark_model_as_processed`**: Marks a model as processed to avoid duplicate runs.

---

## Advanced Features

### Logging
- Logs are saved to both console and file for real-time monitoring and post-training review.
- Customize logging levels as needed (e.g., `INFO`, `DEBUG`).

### Data Augmentation
- Basic and advanced data augmentation techniques are implemented, including synonym replacement using **WordNet**.
- Customize augmentation rules in the `augment_text` and `advanced_augment_text` functions.

---

## Final Notes
This project serves as a foundation for efficient and scalable deep learning workflows. Key highlights:
- **Reproducibility**: Configurations and random seeds ensure consistent results.
- **Scalability**: Supports large datasets and complex models with optimized data loading and augmentation.
- **Flexibility**: Easily extendable to support additional models, datasets, and training configurations.

For further assistance or contributions, feel free to open an issue or submit a pull request!
