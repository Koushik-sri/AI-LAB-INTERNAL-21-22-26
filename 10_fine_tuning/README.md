# Project 10 — Fine-Tuning Studio for Domain Adaptation

> **Instruction Data Validation & Parameter-Efficient LoRA Tuning Studio**

---

## 🎯 Overview

The **Fine-Tuning Studio for Domain Adaptation** manages parameter-efficient fine-tuning (PEFT) workflows using Low-Rank Adaptation (LoRA). It validates domain instruction JSONL data, configures rank (`r=8`) and scaling factor (`alpha=16`), tracks epoch training loss, and evaluates base vs. fine-tuned model performance gains.

---

## 🏗️ Fine-Tuning Pipeline

```
[Domain Instruction Dataset (`domain_data.jsonl`)]
          │
          ▼
[Dataset Validator & Tokenizer Inspector]
          │
          ▼
[LoRA Configurer (`r=8`, `alpha=16`, `target_modules=[q_proj, v_proj]`)]
          │
          ▼
[LoRA Adapter Trainer (Multi-Epoch Loss Tracking)]
          │
          ▼
[Base vs. Adapted Model Evaluator (+44.9% Domain Accuracy Gain)]
```

---

## ⚡ Key Features

- **LoRA Configuration Engine**: Configures PEFT parameters (`r=8, alpha=16, dropout=0.05`).
- **Training Loss Monitoring**: Real-time epoch loss curve visualization (`2.45 -> 0.54`).
- **Base vs. Adapted Evaluator**: Compares baseline Gemini model against domain-adapted LoRA checkpoint.
- **Instruction Data Validator**: Verifies prompt-completion pair structures before training runs.

---

## 🔌 API Endpoints Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Fine-Tuning Studio UI. |
| `POST` | `/api/run-fine-tune` | Executes LoRA adapter training and returns training history and evaluation metrics. |

---

## 🚀 Running Standalone Server

To run Project 10 standalone on port `8010`:

```bash
cd 10_fine_tuning/src
python -m uvicorn server:app --host 127.0.0.1 --port 8010
```

Access the UI at: **http://127.0.0.1:8010**

---

## 🧪 Automated Testing

Run unit tests for Project 10:

```bash
python -m pytest 10_fine_tuning/tests/
```
