# LLM Inference for Clinical Evaluation

This directory provides scripts for zero-shot clinical prediction tasks using external Large Language Models (LLMs) such as MedGemma-27B.

## Clinical Tasks

Evaluates patient outcomes over specific time horizons for various conditions:
- **Acute Myocardial Infarction (MI)**
- **Breast Cancer**
- **Congestive Heart Failure (CHF)**
- **COPD**
- **Dementia**
- **Knee Osteoarthritis**
- **Pancreatic Cancer**
- **Prostate Cancer**
- **Stroke**

## Setup

Requires a GPU with high memory capacity (e.g., A100-80GB) to run larger models like `medgemma-27b-it`.

1. **Environment**: Ensure `vllm` and `transformers` are installed (see root `requirements.txt`).
2. **Hugging Face Token**: Ensure your `HF_TOKEN` is set as an environment variable to access gated models:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

## Usage

Run the evaluation script directly for a specific clinical condition:
```bash
python evaluate.py \
    --model "google/medgemma-27b-it" \
    --input_chunk "/path/to/data_chunk_0.pkl" \
    --output_file "results/acute_mi_output.pkl" \
    --time_horizon 1825 \
    --condition "acute_mi"
```

### Detailed Evaluation Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | str | `google/medgemma-27b-it` | HuggingFace model repository identifier. |
| `--input_chunk` | str | (required) | Path to the specific preprocessed data chunk (`.pkl` file). |
| `--output_file` | str | (required) | Path to save the evaluation results (`.pkl` file). |
| `--time_horizon` | int | 1825 | Prediction horizon in days (e.g., 1825 for 5 years). |
| `--condition` | str | (required) | The clinical condition to evaluate (choices: `acute_mi`, `breast_cancer`, `chf`, `copd`, `dementia`, `kneeoa`, `pancreatic_cancer`, `prostate_cancer`, `stroke`). |

## Key Parameters Explained
- **Time Horizon**: The script calculates whether a specific event occurs within the specified number of days from the last recorded visit in the clinical sequence.
- **Condition Metadata**: The script includes hardcoded ICD-10 codes and medication names to identify positive outcomes for each condition.
- **vLLM Integration**: Uses `vllm` for high-throughput batch inference, which is significantly faster than standard HuggingFace generation.
