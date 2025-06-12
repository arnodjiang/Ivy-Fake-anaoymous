# IVY-FAKE: Unified Explainable Benchmark and Detector for AIGC Content

[![Paper](https://img.shields.io/badge/paper-OpenReview-B31B1B.svg)](https://openreview.net/attachment?id=RIBj1KPAWM&name=pdf)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake)
[![GitHub Code](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Pi3AI/IvyFake) [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

![Intro-image](static/images/figure1-poster-v2_00.png)

This repository provides the official implementation of **IVY-FAKE** and **IVY-xDETECTOR**, a unified explainable framework and benchmark for detecting AI-generated content (AIGC) across **both images and videos**.

---

## 🔍 Overview

**IVY-FAKE** is the **first large-scale dataset** designed for **multimodal explainable AIGC detection**. It contains:
- **150K+** training samples (images + videos)
- **18.7K** evaluation samples
- **Fine-grained annotations** including:
  - Spatial and temporal artifact analysis
  - Natural language reasoning (<think>...</think>)
  - Binary labels with explanations (<conclusion>real/fake</conclusion>)

**IVY-xDETECTOR** is a vision-language detection model trained to:
- Identify synthetic artifacts in images and videos
- Generate **step-by-step reasoning**
- Achieve **SOTA performance** across multiple benchmarks

---

## 📦 Evaluation

```bash
conda create -n ivy-detect python=3.10
conda activate ivy-detect

# Install dependencies
pip install -r requirements.txt
```

---

🚀 Evaluation Script

We provide an evaluation script to test large language model (LLM) performance on reasoning-based AIGC detection.

🔑 Environment Variables

Before running, export the following environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or OpenAI's default base URL
```

▶️ Run Evaluation

```bash
python eva_scripts.py \
  --eva_model_name gpt-4o-mini \
  --res_json_path ./error_item.json
```

This script compares model predictions (<conclusion>real/fake</conclusion>) to the ground truth and logs mismatches to error_item.json.

---

🧪 Input Format

The evaluation script `res_json_path` accepts a JSON array (Dict in List) where each item has:
```json
{
  "rel_path": "relative/path/to/file.mp4",
  "label": "real or fake",
  "raw_ground_truth": "<think>...</think><conclusion>fake</conclusion>",
  "infer_result": "<think>...</think><conclusion>real</conclusion>"
}
```

- label: ground truth
- raw_ground_truth: reasoning by gemini2.5 pro
- infer_result: model reasoning and prediction

Example file: `./evaluate_scripts/error_item.json`

---
