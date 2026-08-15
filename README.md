# LLM Fine-tuning Workbench

A full-stack platform for dataset processing, model fine-tuning, evaluation, and interactive analysis using Large Language Models (LLMs).

---

## Overview

This project provides an end-to-end workflow for:

* Uploading and validating datasets
* Automatically splitting datasets into train/validation/test
* Fine-tuning LLMs (LoRA / QLoRA / full)
* Monitoring training progress
* Evaluating model performance
* Interacting with an AI assistant for analysis and suggestions

It is designed to be **reproducible, modular, and easy to run**.

---

## Project Structure

```
.
├── backend/              # FastAPI backend
│   ├── app.py
│   └── core/
│       └── trainer.py
├── frontend/             # Streamlit UI
│   └── streamlit_app.py
├── script/               # Utility scripts
│   └── download_assistant_model.py
├── data/                 # Dataset storage (ignored in Git)
├── sample_datasets/      # Example datasets for testing
├── models_cache/         # Local model cache (ignored in Git)
├── outputs/              # Training outputs (ignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/limour58077202-art/LLM-SFT-Platform.git
cd LLM-SFT-Platform
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model And API Setup

Training/fine-tuning still uses a local Hugging Face model such as:

```text
Qwen/Qwen2.5-0.5B
```

AI Assistant and Inference use the DeepSeek API instead of downloading a local
assistant model.

### DeepSeek API

You can either enter your DeepSeek API key in the frontend sidebar, or set it
before starting the app:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

Default API settings:

```text
DEEPSEEK_API_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### Local Training Model

Base models are downloaded by Transformers when training or evaluation needs
them. You can also point to an existing local model path in the frontend model
selector or backend request.


---

## Running the Project

### Option 1: One-click launcher on macOS

Double-click:

```text
LLM Fine-tuning Workbench.command
```

This starts both the FastAPI backend and Streamlit frontend, then opens:

```text
http://127.0.0.1:8501
```

To stop the local services, double-click:

```text
Stop LLM Fine-tuning Workbench.command
```

### Option 2: Manual startup

### 1. Start backend (FastAPI)

```bash
uvicorn backend.app:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### 2. Start frontend (Streamlit)

```bash
streamlit run frontend/streamlit_app.py
```

Frontend runs at:

```
http://localhost:8501
```

---

## How To Use The Software

### 1. Open The App

On macOS, double-click:

```text
LLM Fine-tuning Workbench.command
```

Wait until the terminal shows:

```text
Backend is ready.
Frontend is ready.
```

Then open the app in your browser:

```text
http://127.0.0.1:8501
```

### 2. Configure DeepSeek

In the left sidebar, enter your DeepSeek API key in:

```text
DeepSeek API Key
```

DeepSeek is used by:

* AI Assistant
* Evaluation explanation and suggestions
* Inference testing

### 3. Upload A Dataset

You can upload your own CSV, JSON, or JSONL file. The file can contain only:

```text
input
output
```

For a quick test, use the built-in sample dataset:

```text
sample_datasets/sentiment_demo.jsonl
```

Before uploading the sample dataset, enter this in `Dataset Instruction`:

```text
判断下面这句话的情感倾向，只回答正面或负面。
```

Then click:

```text
Upload to Backend
```

### 4. Check The Dataset

Click:

```text
Check Dataset
```

The backend will validate the dataset and split it into:

```text
train 70% / validation 15% / test 15%
```

For the built-in 500-sample dataset, the split is:

```text
train 350 / validation 75 / test 75
```

### 5. Start Training

For a fast local test, use:

```text
Base Model: Qwen2.5-0.5B
Fine-tuning Method: LoRA
Epochs: 1
Batch Size: 1 or 2
Max Length: 128
```

Then click:

```text
Start Fine-tuning
```

If the selected model is not already in `models_cache/`, it will be downloaded
automatically by Transformers before training starts.

### 6. Monitor Training

Open the `Monitor` tab.

During training, the page shows:

```text
训练中
```

After training finishes, the page shows:

```text
训练完成
```

### 7. Generate A Report

After training finishes, click:

```text
生成报告
```

The app generates an HTML report and saves it under:

```text
outputs/reports/
```

### 8. Run Inference

Open the `Inference` tab and enter a test sentence, for example:

```text
这个酒店房间很干净，服务也很好。
```

The DeepSeek inference result should return:

```text
正面
```

### 9. Ask The AI Assistant

Use the `AI Assistant` button to ask questions about:

* Dataset quality
* Training status
* Evaluation results
* Error patterns
* Suggested next steps

### 10. Stop The App

When finished, double-click:

```text
Stop LLM Fine-tuning Workbench.command
```

---

## Workflow

1. Enter a shared instruction and upload dataset (CSV / JSON / JSONL)
2. System validates and splits data automatically
3. Configure training parameters
4. Start fine-tuning
5. Monitor training progress
6. Generate an HTML training report after training completes
7. Evaluate model performance
8. Use AI assistant for insights

---

## Dataset Format

Uploaded files can contain only:

* `input`
* `output`

Enter the shared `instruction` in the frontend sidebar before uploading. If the
uploaded file also has an `instruction` field, the sidebar instruction overrides
it when provided. After upload, the backend normalizes every sample to:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

Datasets are randomly split into train / validation / test with a default ratio
of `70% / 15% / 15%`.

### Sample Dataset

This project includes a ready-to-use sentiment classification dataset:

```text
sample_datasets/sentiment_demo.jsonl
```

It contains 500 examples with balanced labels:

* 250 positive samples
* 250 negative samples

Use this instruction in the frontend sidebar when uploading it:

```text
判断下面这句话的情感倾向，只回答正面或负面。
```

---

## Features

* Dataset validation and normalization
* Automatic train/validation/test split
* LoRA / QLoRA / full fine-tuning
* Training monitoring (loss curves)
* Evaluation metrics (accuracy, token-level)
* Error analysis with examples
* DeepSeek-powered AI assistant for evaluation insights
* DeepSeek-powered inference test

## Credits

This project is modified based on [RayanYe/LLM-SFT-Platform](https://github.com/RayanYe/LLM-SFT-Platform).

Original copyright belongs to the original author.

The original project is licensed under the Apache License 2.0.

Modifications in this repository include:

* DeepSeek API integration
* Frontend instruction input
* Built-in 500-sample sentiment dataset
* HTML training report export
* One-click macOS launcher
* Simplified training monitor workflow
