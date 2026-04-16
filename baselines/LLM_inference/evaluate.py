import json, re, pickle, math, os, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from vllm import LLM, SamplingParams

# ===== 1) Args & vLLM Initialization =====
parser = argparse.ArgumentParser(description="Clinical Evaluation: Unified LLM Inference")
parser.add_argument("--model", type=str, default="google/medgemma-27b-it", help="HuggingFace model repository")
parser.add_argument("--input_chunk", type=str, required=True, help="Path to the specific data chunk .pkl file")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the results .pkl file")
parser.add_argument("--time_horizon", type=int, default=1825, help="Prediction horizon in days")
parser.add_argument("--condition", type=str, required=True, choices=[
    "acute_mi", "breast_cancer", "chf", "copd", "dementia", 
    "kneeoa", "pancreatic_cancer", "prostate_cancer", "stroke"
], help="Clinical condition to evaluate")

args, unknown = parser.parse_known_args()

os.environ['HF_TOKEN'] = "********"

# Hardcoded optimal parameters
MAX_TOKENS = 256
K_CONF = 50
HORIZON_TEXT = f"NEXT {args.time_horizon} DAYS"
KEEP_FRAC = 0.4

# ===== 2) Condition Metadata =====
CONDITION_METADATA = {
    "acute_mi": {
        "full_name": "Heart Attack",
        "json_key": "Heart_Attack",
        "idx": [
            "I21.01","I21.02","I21.09","I21.11","I21.19","I21.21","I21.29","I21.3","I21.4","I21.9","I21.A1","I21.A9","I21.B",
            "I22.0","I22.1","I22.2","I22.8","I22.9","I23.0","I23.1","I23.2","I23.3","I23.4","I23.5","I23.6","I23.7","I23.8"
        ]
    },
    "breast_cancer": {
        "full_name": "Breast Cancer",
        "json_key": "Breast_Cancer",
        "idx": [
            'C50.01', 'C50.012', 'C50.019', 'C50.021', 'C50.022', 'C50.029', 
            'C50.111', 'C50.112', 'C50.119', 'C50.121', 'C50.122', 'C50.129', 
            'C50.211', 'C50.212', 'C50.219', 'C50.221', 'C50.222', 'C50.229', 
            'C50.311', 'C50.312', 'C50.319', 'C50.321', 'C50.322', 'C50.329', 
            'C50.411', 'C50.412', 'C50.419', 'C50.421', 'C50.422', 'C50.429', 
            'C50.511', 'C50.512', 'C50.519', 'C50.521', 'C50.522', 'C50.611', 
            'C50.612', 'C50.619', 'C50.621', 'C50.622', 'C50.811', 'C50.812', 
            'C50.819', 'C50.821', 'C50.822', 'C50.829', 'C50.911', 'C50.912', 
            'C50.919', 'C50.921', 'C50.922', 'C50.929', 'D05.10', 'D05.11', 
            'D05.12', 'D05.80', 'D05.81', 'D05.82', 'D05.90', 'D05.91', 
            'D05.92', 'Z85.3', 'Z86.000'
        ]
    },
    "chf": {
        "full_name": "Congestive Heart Failure",
        "json_key": "CHF",
        "idx": [
            "I09.81","I11.0","I13.0","I13.2","I50.1","I50.20","I50.21","I50.22","I50.23",
            "I50.30","I50.31","I50.32","I50.33","I50.40","I50.41","I50.42","I50.43",
            "I50.810","I50.811","I50.812","I50.813","I50.814","I50.82","I50.83","I50.84",
            "I50.89","I50.9"
        ]
    },
    "copd": {
        "full_name": "COPD",
        "json_key": "COPD",
        "idx": [
            'J40', 'J41.0', 'J41.1', 'J41.8', 'J42', 'J43.0', 'J43.1', 'J43.2', 
            'J43.9', 'J44.0', 'J44.1', 'J44.81', 'J44.89', 'J44.9', 'J47.0', 
            'J47.1', 'J47.9', 'J98.2', 'J98.3'
        ]
    },
    "dementia": {
        "full_name": "Dementia",
        "json_key": "Dementia",
        "idx": [
            'F01', 'G30', 'G31.01', 'G31.09', 'G31.83', 'G31.85', 'G31.9', 'F02', 'F03', 
            'G31.84', 'G31.1', 'DONEPEZIL', 'RIVASTIGMINE', 'GALANTAMINE', 'MEMANTINE', 
            'TACRINE', 'G23.1'
        ]
    },
    "kneeoa": {
        "full_name": "Knee Osteoarthritis",
        "json_key": "KneeOA",
        "idx": ['M17.0', 'M17.1', 'M17.2', 'M17.3', 'M17.4', 'M17.5', 'M17.9']
    },
    "pancreatic_cancer": {
        "full_name": "Pancreatic Cancer",
        "json_key": "Pancreatic_Cancer",
        "idx": ['C25.0', 'C25.1', 'C25.2', 'C25.3', 'C25.7', 'C25.8', 'C25.9']
    },
    "prostate_cancer": {
        "full_name": "Prostate Cancer",
        "json_key": "Prostate_Cancer",
        "idx": ['C61', 'D07.5', 'Z85.46']
    },
    "stroke": {
        "full_name": "Stroke",
        "json_key": "Stroke",
        "idx": ['I60', 'I61', 'I63']
    }
}

meta = CONDITION_METADATA[args.condition]
AMI_SET = set(meta["idx"])
STOP = ["[END]","<|eot_id|>","</s>"]

print(f"Initializing vLLM for {meta['full_name']} ({MAX_TOKENS} tokens) on chunk: {args.input_chunk}")

llm = LLM(
    model=args.model,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=16384,
    enforce_eager=True,
    tensor_parallel_size=2 
)

tok = llm.get_tokenizer()

# ===== 3) Optimized vLLM Generator =====
def vllm_generate(prompts, n=1, temperature=0.7, top_p=0.95, max_tokens=256, stop="[END]"):
    stop_list = [stop] if isinstance(stop, str) else stop
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop_list)
    return llm.generate(prompts, sp)

# ===== 4) Globals & Helpers =====
def _pid(s): return s["key"].split("_")[0]

def safe_div(num, den):
    out = np.full_like(den, np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den>0))
    return out

def eval_scores(y_true, p, name):
    y_true=np.asarray(y_true); p=np.asarray(p,dtype=float)
    ok=~np.isnan(p); y_true=y_true[ok]; p=p[ok]
    unique_classes = len(np.unique(y_true))
    return {
        "name":name,"n":int(ok.sum()),
        "auroc":float(roc_auc_score(y_true,p)) if unique_classes==2 else np.nan,
        "auprc":float(average_precision_score(y_true,p)) if unique_classes==2 else np.nan,
        "brier":float(brier_score_loss(y_true,p)) if len(y_true) > 0 else np.nan,
        "mean":float(p.mean()) if len(p) > 0 else np.nan
    }

def trim_history_by_fraction(clinical_json, prefix, suffix, keep_frac, max_tokens_out):
    max_len=llm.llm_engine.model_config.max_model_len
    budget=max_len - max_tokens_out
    pref=len(tok.encode(prefix)); suf=len(tok.encode(suffix))
    overhead=pref+suf
    if overhead>=budget: raise ValueError(f"Scaffold too big: overhead={overhead} budget={budget}")
    hist_budget=budget-overhead
    ids=tok.encode(clinical_json); L=len(ids)
    target=max(256, int(math.floor(L*keep_frac)))
    target=min(target, hist_budget, L)
    return tok.decode(ids[-target:])

# ===== 5) Prompt Builder =====
def build_prompt_mod_cot(sample, horizon_text, keep_frac, max_tokens_out):
    patient_id = _pid(sample)
    clinical_json = sample["prompt"]
    cond_name = meta["full_name"]
    json_key = meta["json_key"]
    
    prefix = f"""<s>[SYSTEM_PROMPT]
    You are an information extraction and forecasting engine for clinical risk prediction. Your task is to predict the risk of {cond_name} based on patient history.

    You MUST follow these rules exactly:
    1) Output MUST be EXACTLY one JSON object, and NOTHING else.
    2) The JSON MUST have exactly this structure:
        {{
          "{patient_id}": {{
            "reason": "<brief_1_to_2_sentence_summary>",
            "confidence": <float_0_to_1>,
            "{json_key}": "<Y_or_N>"
          }}
        }}
    3) **CRITICAL**: The "confidence" value MUST represent the numerical probability $P(Y)$ of the condition occurring (e.g., 0.85 for high risk, 0.12 for low risk).
    4) Evaluate the patient's records and determine if they will develop {cond_name} within the {horizon_text}.
    5) Provide a concise reason for your prediction based on the evidence in the records.
    [/SYSTEM_PROMPT]
    [PATIENT_HISTORY]
    """
    suffix = f"""
    [/PATIENT_HISTORY]
    [QUERY]
    Predict the risk of {cond_name} for this patient in the {horizon_text} based on the records provided above.
    [/QUERY]
    [OUTPUT]"""
    
    trimmed_hist = trim_history_by_fraction(clinical_json, prefix, suffix, keep_frac, max_tokens_out)
    return prefix + trimmed_hist + suffix

# ===== 6) Main Execution =====
with open(args.input_chunk, "rb") as f:
    chunk_data = pickle.load(f)

print(f"Generating prompts for {len(chunk_data)} samples...")
prompts = [build_prompt_mod_cot(s, HORIZON_TEXT, KEEP_FRAC, MAX_TOKENS) for s in chunk_data]

print("Running batch inference...")
outputs = vllm_generate(prompts, max_tokens=MAX_TOKENS, stop=STOP)

results = []
for i, out in enumerate(outputs):
    raw_text = out.outputs[0].text.strip()
    # Clean up markdown JSON if present
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()
    
    pid = _pid(chunk_data[i])
    prediction = None
    confidence = np.nan
    reason = ""
    
    try:
        parsed = json.loads(clean_text)
        if pid in parsed:
            entry = parsed[pid]
            prediction = entry.get(meta["json_key"])
            confidence = float(entry.get("confidence", np.nan))
            reason = entry.get("reason", "")
    except Exception:
        # Fallback regex if JSON parsing fails
        pred_match = re.search(f'"{meta["json_key"]}":\s*"([YN])"', clean_text)
        conf_match = re.search(r'"confidence":\s*([0-9.]+)', clean_text)
        if pred_match: prediction = pred_match.group(1)
        if conf_match: confidence = float(conf_match.group(1))

    # Determine ground truth from chunk_data if available
    # (Assuming chunk_data structure matches your original scripts)
    # The original scripts seem to use a 'label' field if available
    label = chunk_data[i].get("label", np.nan)
    
    results.append({
        "key": chunk_data[i]["key"],
        "patient_id": pid,
        "prediction": prediction,
        "confidence": confidence,
        "label": label,
        "reason": reason,
        "raw_output": raw_text
    })

# Save results
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, "wb") as f:
    pickle.dump(results, f)

# Print mini summary
y_true = [r["label"] for r in results if not np.isnan(r["label"])]
y_score = [r["confidence"] for r in results if not np.isnan(r["label"])]
if len(y_true) > 0:
    metrics = eval_scores(y_true, y_score, meta["full_name"])
    print(f"Inference complete. AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
else:
    print("Inference complete. No labels found for metric calculation.")
