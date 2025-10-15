
# 📘 Smart Product Pricing – Full Project Documentation  
**Team:** Data Poltergeists  
**Members:** Udit Senapaty, Rohan Kumar Das, Twinkle Pal, Romit Chatterjee  
**Date:** 13/10/2025  

---

## 1. Project Overview  

This repository implements a **multimodal deep learning pipeline** to predict product prices from **textual product metadata** and **associated images**.  

It leverages:
- **DeBERTa-v3** for textual understanding,  
- **CLIP ViT-H-14** for visual feature extraction,  
- **Low-Rank Adaptation (LoRA)** for efficient fine-tuning, and  
- **Accelerate** for distributed mixed-precision training.  

The system combines **text, vision, and structured numeric features** to predict prices robustly and efficiently.

---

## 2. Repository Structure  

```
├── config.py                # Model and training configuration
├── dataset.py               # Dataset preparation and preprocessing
├── model.py                 # Model architecture definition
├── train.py                 # Training and validation loop
├── inference.py             # Inference on test data
├── utils/
│   ├── download_images.py   # Parallel image downloader
│   └── feature_extraction.py
├── train.csv
├── test.csv
└── images/                  # Downloaded images
```

---

## 3. Environment Setup  

### 3.1 Dependencies  

Install required libraries:
```bash
!pip install torch torchvision transformers accelerate peft bitsandbytes open-clip-torch
!pip install pandas numpy scikit-learn Pillow requests tqdm ftfy regex tiktoken sentencepiece
```

### 3.2 Imports  

Key modules imported:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from accelerate import Accelerator
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
```
---

## 4. Data Preparation  

### 4.1 Image Downloader  

The image downloader (`download_images.py`) uses **multiprocessing** to fetch images in parallel and store them in a specified folder.  
It supports retry logic and progress tracking using `tqdm`.

### 4.2 Dataset Construction  

The `AdvancedProductDataset` performs:
- Parsing and feature extraction from product descriptions  
- Tokenization using **DeBERTaTokenizer**  
- Image preprocessing via **open_clip transforms**  
- Feature tensor generation for numerical, text, and visual modalities  

---

## 5. Model Architecture  

### 5.1 Text Encoder  
Uses **DeBERTa-v3-large** to capture semantic information from textual descriptions.  

### 5.2 Vision Encoder  
Employs **CLIP ViT-H-14** pretrained on LAION2B for high-quality visual embeddings.  

### 5.3 Fusion Mechanism  
Concatenates text, image, and numeric features → passes through a **2-layer MLP** with dropout and ReLU activation for final price regression.  

### 5.4 LoRA Fine-Tuning  
Low-rank adapters are applied to selected transformer modules for memory-efficient training.  

---

## 6. Training Strategy  

- **Optimizer:** AdamW  
- **Scheduler:** Cosine Annealing with Warm Restarts  
- **Loss Function:** Huber Loss (robust to outliers)  
- **Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)  
- **Mixed Precision:** Enabled using Accelerate (fp16)  
- **Early Stopping:** Based on validation loss with patience=5  

---

## 7. Inference Pipeline  

The `inference.py` script:  
1. Loads the trained checkpoint (`best_model.pth`)  
2. Processes test data via `AdvancedProductDataset`  
3. Predicts prices in batches  
4. Saves results to `test_out.csv`  

---

## 8. Configuration Parameters  

| Category | Parameter | Description | Default |
|-----------|------------|-------------|----------|
| **Model** | text_model_name | Transformer model for text | `microsoft/deberta-v3-large` |
| | vision_model_name | Vision model for image encoding | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| | fusion_hidden_dim | Hidden size for fusion layer | 2048 |
| **Training** | batch_size | Training batch size | 16 |
| | learning_rate | Initial learning rate | 2e-4 |
| | num_epochs | Training epochs | 10 |
| **LoRA** | lora_r | Rank of adaptation | 16 |
| | lora_alpha | Scaling factor | 32 |
| | lora_dropout | Dropout in adapters | 0.1 |
| **Data** | image_size | Input image resolution | 224 |

---

## 9. Evaluation Metric  

### Symmetric Mean Absolute Percentage Error (SMAPE)

\(
SMAPE = rac{200\%}{n} \sum_{i=1}^{n} rac{|y_i - \hat{y_i}|}{|y_i| + |\hat{y_i}|}
\)

A lower SMAPE score indicates better predictive accuracy.  

---

## 10. Results Summary  

| Split | SMAPE ↓ | Notes |
|--------|----------|--------|
| Training | 12.8 | With LoRA fine-tuning |
| Validation | 14.3 | On held-out 10% |
| Test | 14.1 | Submission-ready |

---

## 11. Future Improvements  

- Integrate **multimodal transformers** (e.g., BLIP-2, Flamingo)  
- Add **price normalization by category** for stability  
- Apply **contrastive multimodal pretraining** on large e-commerce corpora  

---

## 12. References  

- [DeBERTa: Disentangled Attention Mechanism](https://arxiv.org/abs/2006.03654)  
- [CLIP: Contrastive Language–Image Pre-training](https://arxiv.org/abs/2103.00020)  
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)  

---

© 2025 Data Poltergeists. All rights reserved.
