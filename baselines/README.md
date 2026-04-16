# EHR Baselines

This repository contains various baseline implementations for modeling Electronic Health Record (EHR) data using Large Language Models (LLMs) and Generative Pre-trained Transformers (GPT).

## Repository Structure

- **`EGE/`**: "Enhanced Generative EHR" - A GPT-based model that explicitly models time using duration tokens (coin factorization system) between clinical events.
- **`Multiclass/`**: A standard Next-Token Prediction (NTP) baseline for EHR sequences.
- **`SeqLoss/`**: A variant that supports "Multiset Loss," allowing for parallel/unordered supervision of clinical codes within a visit.
- **`LLM_inference/`**: Scripts for zero-shot clinical evaluation using external LLMs like MedGemma-27B via vLLM.

## Environment Setup

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (A100 recommended for LLM inference)

### Installation

1. **Create and activate a new environment:**
   ```bash
   conda create -n baselines-env python=3.10
   conda activate baselines-env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Each directory contains specialized scripts for training and evaluation. Refer to the README files in each subdirectory for detailed instructions on how to run them directly using Python.

- [EGE README](./EGE/README.md)
- [Multiclass README](./Multiclass/README.md)
- [SeqLoss README](./SeqLoss/README.md)
- [LLM_inference README](./LLM_inference/README.md)
