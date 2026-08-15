from __future__ import annotations

import json
import os
import random
import re
import time
import ssl
import inspect
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "models_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 你的本地 Qwen2.5-0.5B-Instruct snapshot 目录
# 例如：
# /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be...
ASSISTANT_MODEL_ID = os.environ.get(
    "ASSISTANT_MODEL_ID",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

ASSISTANT_MODEL_PATH = os.environ.get(
    "ASSISTANT_MODEL_PATH",
    str(CACHE_DIR / "models--Qwen--Qwen2.5-0.5B-Instruct")
)

DEEPSEEK_API_BASE_URL = os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
# ------------------------------------------------------------
# Optional ML stack
# ------------------------------------------------------------
torch = None  # type: ignore
Dataset = None  # type: ignore
AutoModelForCausalLM = None  # type: ignore
AutoTokenizer = None  # type: ignore
DataCollatorForLanguageModeling = None  # type: ignore
Trainer = None  # type: ignore
TrainingArguments = None  # type: ignore
TrainerCallback = object  # type: ignore
set_seed = None  # type: ignore
BitsAndBytesConfig = None  # type: ignore
LoraConfig = None  # type: ignore
TaskType = None  # type: ignore
get_peft_model = None  # type: ignore
PeftModel = None  # type: ignore

REQUIRED_COLUMNS = ["instruction", "input", "output"]

MODEL_ALIASES = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
    "Mixtral-8x22B": "mistralai/Mixtral-8x22B-v0.1",
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "ChatGLM3-6B": "THUDM/chatglm3-6b",
    "InternLM2-7B": "internlm/internlm2-7b",
    "Baichuan2-7B": "baichuan-inc/Baichuan2-7B-Base",
}

# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------
@dataclass
class TrainConfig:
    base_model: str
    dataset_path: str
    finetune_method: str = "LoRA"  # LoRA | QLoRA | Full Fine-tuning
    output_dir: str = "./outputs/finetuned_model"

    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    lora_rank: int = 8
    max_length: int = 512

    seed: int = 42
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 1
    eval_split_ratio: float = 0.1


@dataclass
class TrainResult:
    success: bool
    message: str
    output_dir: str = ""
    model_name: str = ""
    finetune_method: str = ""
    num_train_samples: int = 0
    num_eval_samples: int = 0
    loss_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ensure_dependencies() -> None:
    global torch
    global Dataset
    global AutoModelForCausalLM
    global AutoTokenizer
    global DataCollatorForLanguageModeling
    global Trainer
    global TrainingArguments
    global TrainerCallback
    global set_seed
    global BitsAndBytesConfig
    global LoraConfig
    global TaskType
    global get_peft_model
    global PeftModel

    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing required ML dependency: torch.") from e

    if Dataset is None:
        try:
            from datasets import Dataset as DatasetClass
            Dataset = DatasetClass  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing required ML dependency: datasets.") from e

    if Trainer is None or TrainingArguments is None:
        try:
            from transformers import (
                AutoModelForCausalLM as AutoModelForCausalLMClass,
                AutoTokenizer as AutoTokenizerClass,
                DataCollatorForLanguageModeling as DataCollatorClass,
                Trainer as TrainerClass,
                TrainingArguments as TrainingArgumentsClass,
                TrainerCallback as TrainerCallbackClass,
                set_seed as set_seed_fn,
                BitsAndBytesConfig as BitsAndBytesConfigClass,
            )
            AutoModelForCausalLM = AutoModelForCausalLMClass  # type: ignore
            AutoTokenizer = AutoTokenizerClass  # type: ignore
            DataCollatorForLanguageModeling = DataCollatorClass  # type: ignore
            Trainer = TrainerClass  # type: ignore
            TrainingArguments = TrainingArgumentsClass  # type: ignore
            TrainerCallback = TrainerCallbackClass  # type: ignore
            set_seed = set_seed_fn  # type: ignore
            BitsAndBytesConfig = BitsAndBytesConfigClass  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing required ML dependency: transformers.") from e

    if get_peft_model is None:
        try:
            from peft import LoraConfig as LoraConfigClass
            from peft import PeftModel as PeftModelClass
            from peft import TaskType as TaskTypeClass
            from peft import get_peft_model as get_peft_model_fn
            LoraConfig = LoraConfigClass  # type: ignore
            TaskType = TaskTypeClass  # type: ignore
            get_peft_model = get_peft_model_fn  # type: ignore
            PeftModel = PeftModelClass  # type: ignore
        except Exception:
            pass

    if torch is None or Dataset is None or Trainer is None or TrainingArguments is None:
        raise RuntimeError(
            "Missing required ML dependencies. Please install torch, datasets, transformers."
        )


def detect_device() -> tuple[Any, Any, str]:
    if torch is None:
        return None, None, "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16, "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16, "mps"
    return torch.device("cpu"), torch.float32, "cpu"


def normalize_model_name(name: str) -> str:
    candidate = Path(name)
    if candidate.exists():
        return str(candidate)
    return MODEL_ALIASES.get(name, name)


def resolve_local_model_path(path_str: str) -> str:
    """
    Accept either:
    1) actual snapshot folder with config.json / model.safetensors
    2) a parent folder containing snapshots/<hash>/...
    3) a normal HF model id
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    if not p.exists():
        return path_str

    if p.is_file():
        return str(p)

    if (p / "config.json").exists():
        return str(p)

    snapshots = p / "snapshots"
    if snapshots.exists() and snapshots.is_dir():
        for child in sorted(snapshots.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                return str(child)

    return str(p)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_training_arguments(**kwargs: Any):
    if TrainingArguments is None:
        raise RuntimeError("TrainingArguments is not available.")

    signature = inspect.signature(TrainingArguments.__init__)
    accepted = set(signature.parameters)
    accepted.discard("self")

    normalized = dict(kwargs)
    if "evaluation_strategy" in normalized and "eval_strategy" in accepted:
        normalized["eval_strategy"] = normalized.pop("evaluation_strategy")
    if "eval_strategy" in normalized and "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        normalized["evaluation_strategy"] = normalized.pop("eval_strategy")
    if "warmup_ratio" in normalized and "warmup_ratio" not in accepted:
        normalized.pop("warmup_ratio")

    return TrainingArguments(**{k: v for k, v in normalized.items() if k in accepted})


def _build_trainer(tokenizer: Any, **kwargs: Any):
    if Trainer is None:
        raise RuntimeError("Trainer is not available.")

    signature = inspect.signature(Trainer.__init__)
    accepted = set(signature.parameters)
    accepted.discard("self")

    normalized = dict(kwargs)
    if "processing_class" in accepted:
        normalized["processing_class"] = tokenizer
    elif "tokenizer" in accepted:
        normalized["tokenizer"] = tokenizer

    return Trainer(**{k: v for k, v in normalized.items() if k in accepted})


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse failed at line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} must be an object.")
            records.append(obj)
    return records


def _sample_keys(records: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for item in records:
        keys.update(item.keys())
    return sorted(keys)


def _validate_record(record: Dict[str, Any], idx: int) -> None:
    missing = [k for k in REQUIRED_COLUMNS if k not in record]
    if missing:
        raise ValueError(f"Sample {idx} is missing fields: {', '.join(missing)}")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_sft_text(example: Dict[str, Any]) -> str:
    instruction = _normalize_text(example.get("instruction"))
    input_text = _normalize_text(example.get("input"))
    output_text = _normalize_text(example.get("output"))

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    return prompt + output_text


def _guess_lora_target_modules(model_name: str) -> List[str]:
    name = model_name.lower()
    if any(x in name for x in ["qwen", "llama", "mistral", "gemma", "phi", "yi", "glm", "internlm", "baichuan"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _normalize_metric_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize_for_metric(text: str) -> List[str]:
    text = _normalize_metric_text(text)
    if not text:
        return []
    if re.search(r"\s", text):
        return text.split()
    return list(text)


def _exact_match(reference: str, prediction: str) -> bool:
    return _normalize_metric_text(reference) == _normalize_metric_text(prediction)


def _token_accuracy(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize_for_metric(reference)
    pred_tokens = _tokenize_for_metric(prediction)

    denom = max(len(ref_tokens), len(pred_tokens), 1)
    matches = 0
    for i in range(min(len(ref_tokens), len(pred_tokens))):
        if ref_tokens[i] == pred_tokens[i]:
            matches += 1
    return matches / denom


def build_generation_prompt(instruction: str, input_text: str = "", template: str = "alpaca") -> str:
    instruction = _normalize_text(instruction)
    input_text = _normalize_text(input_text)

    if template.lower() in {"alpaca", "qwen", "llama2", "chatml"}:
        if input_text:
            return f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        return f"Instruction: {instruction}\nResponse:"

    if input_text:
        return f"{instruction}\n\n{input_text}\n\nAnswer:"
    return f"{instruction}\n\nAnswer:"


def build_chat_messages_from_context(
    question: str,
    evaluation_context: Dict[str, Any],
) -> List[Dict[str, str]]:
    summary = {
        "metrics": evaluation_context.get("metrics", {}),
        "examples": evaluation_context.get("examples", [])[:3],
        "generation_config": evaluation_context.get("generation_config", {}),
        "model": evaluation_context.get("model", {}),
        "split_name": evaluation_context.get("split_name", ""),
        "message": evaluation_context.get("message", ""),
        "training_status": evaluation_context.get("training_status", {}),
        "dataset_report": evaluation_context.get("dataset_report", {}),
        "validation_result": evaluation_context.get("validation_result", {}),
    }

    system_prompt = (
        "You are an expert AI assistant for LLM fine-tuning workflows.\n"
        "Use any available context: dataset report, validation result, training status, "
        "training metrics, evaluation metrics, examples, generation config, and model config.\n"
        "If evaluation metrics are missing, do not say the whole context is empty when "
        "training or dataset context exists. Instead, explain what can be concluded from "
        "the available information and what still requires running evaluation.\n"
        "Be concise, specific, and actionable. Suggest changes to data, prompt template, "
        "decoding params, or training settings when useful."
    )

    user_prompt = (
        f"Workflow context:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"User question:\n{question}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ------------------------------------------------------------
# Validation / config
# ------------------------------------------------------------
def validate_dataset(dataset_path: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    if not path.exists():
        return {
            "valid": False,
            "message": f"Dataset not found: {dataset_path}",
            "num_samples": 0,
            "sample_keys": [],
            "preview": [],
        }

    if path.suffix.lower() != ".jsonl":
        return {
            "valid": False,
            "message": "Dataset must be a .jsonl file after backend normalization.",
            "num_samples": 0,
            "sample_keys": [],
            "preview": [],
        }

    try:
        records = _read_jsonl(path)
    except Exception as e:
        return {
            "valid": False,
            "message": str(e),
            "num_samples": 0,
            "sample_keys": [],
            "preview": [],
        }

    if not records:
        return {
            "valid": False,
            "message": "Dataset is empty.",
            "num_samples": 0,
            "sample_keys": [],
            "preview": [],
        }

    try:
        for idx, rec in enumerate(records, start=1):
            _validate_record(rec, idx)
    except Exception as e:
        return {
            "valid": False,
            "message": str(e),
            "num_samples": len(records),
            "sample_keys": _sample_keys(records),
            "preview": records[:3],
        }

    return {
        "valid": True,
        "message": f"Dataset valid with {len(records)} samples.",
        "num_samples": len(records),
        "sample_keys": _sample_keys(records),
        "preview": records[:3],
    }


def build_train_config(payload: Dict[str, Any]) -> TrainConfig:
    required = ["base_model", "dataset_path"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    finetune_method = str(payload.get("finetune_method", "LoRA")).strip()
    if finetune_method not in {"LoRA", "QLoRA", "Full Fine-tuning"}:
        raise ValueError("finetune_method must be one of: LoRA, QLoRA, Full Fine-tuning")

    dataset_path = Path(str(payload["dataset_path"]))
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    output_dir = str(payload.get("output_dir", "./outputs/finetuned_model"))
    _safe_mkdir(Path(output_dir))

    normalized_model = normalize_model_name(str(payload["base_model"]))

    return TrainConfig(
        base_model=normalized_model,
        dataset_path=str(dataset_path),
        finetune_method=finetune_method,
        output_dir=output_dir,
        learning_rate=float(payload.get("learning_rate", 2e-4)),
        batch_size=int(payload.get("batch_size", 4)),
        epochs=int(payload.get("epochs", 3)),
        lora_rank=int(payload.get("lora_rank", 8)),
        max_length=int(payload.get("max_length", 512)),
        seed=int(payload.get("seed", 42)),
        warmup_ratio=float(payload.get("warmup_ratio", 0.03)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        logging_steps=int(payload.get("logging_steps", 1)),
        eval_split_ratio=float(payload.get("eval_split_ratio", 0.1)),
    )


# ------------------------------------------------------------
# Dataset / assistant loading
# ------------------------------------------------------------
def _load_model_and_tokenizer(config: TrainConfig):
    _ensure_dependencies()

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not available.")

    device, dtype, device_name = detect_device()
    model_id = normalize_model_name(config.base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(CACHE_DIR),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    effective_method = config.finetune_method
    use_cuda = device_name == "cuda"

    if effective_method == "QLoRA" and not use_cuda:
        effective_method = "LoRA"

    if effective_method == "QLoRA":
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is required for QLoRA.")
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes / BitsAndBytesConfig is required for QLoRA.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        lora_targets = _guess_lora_target_modules(model_id)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=max(config.lora_rank * 2, 8),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_targets,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer, effective_method, device_name

    if effective_method == "LoRA":
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is required for LoRA.")

        load_kwargs: Dict[str, Any] = {
            "cache_dir": str(CACHE_DIR),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device_name in {"cuda", "mps"} and dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.config.use_cache = False

        if device is not None:
            model.to(device)

        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

        lora_targets = _guess_lora_target_modules(model_id)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=max(config.lora_rank * 2, 8),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_targets,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer, effective_method, device_name

    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device_name in {"cuda", "mps"} and dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False

    if device is not None:
        model.to(device)

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, tokenizer, effective_method, device_name


def _prepare_datasets(
    records: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    seed: int,
    eval_split_ratio: float,
):
    if Dataset is None:
        raise RuntimeError("datasets is not available.")

    texts = [_format_sft_text(r) for r in records]
    raw_ds = Dataset.from_dict({"text": texts})

    if len(raw_ds) <= 1:
        train_ds = raw_ds
        eval_ds = None
        return train_ds, eval_ds

    test_size = max(min(eval_split_ratio, 0.3), 0.05)
    split = raw_ds.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    return train_ds, eval_ds


def create_loss_history_callback():
    base_callback = TrainerCallback if TrainerCallback is not None else object

    class LossHistoryCallback(base_callback):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.loss_history: List[Dict[str, Any]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if not logs:
                return control
            if "loss" in logs:
                try:
                    loss_value = float(logs["loss"])
                except Exception:
                    return control
                self.loss_history.append({"step": int(state.global_step), "loss": loss_value})
            return control

    return LossHistoryCallback()


# ------------------------------------------------------------
# Mock fallback
# ------------------------------------------------------------
def _mock_training(config: TrainConfig, reason: str) -> TrainResult:
    loss_history: List[Dict[str, Any]] = []
    loss = 1.2
    steps = max(3, config.epochs * 4)

    for step in range(1, steps + 1):
        loss = max(0.05, loss * random.uniform(0.80, 0.95))
        loss_history.append({"step": step, "loss": round(loss, 4)})
        time.sleep(0.15)

    return TrainResult(
        success=True,
        message="Mock training completed.",
        output_dir=config.output_dir,
        model_name=config.base_model,
        finetune_method=config.finetune_method,
        num_train_samples=0,
        num_eval_samples=0,
        loss_history=loss_history,
        metrics={
            "train_runtime": round(steps * 0.15, 2),
            "final_loss": loss_history[-1]["loss"] if loss_history else None,
        },
        note=f"Mock mode used: {reason}",
    )


# ------------------------------------------------------------
# Main training entry
# ------------------------------------------------------------
def start_training(config: TrainConfig) -> TrainResult:
    try:
        if set_seed is not None:
            set_seed(config.seed)

        if torch is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        dataset_path = Path(config.dataset_path)
        records = _read_jsonl(dataset_path)
        if not records:
            return TrainResult(
                success=False,
                message="Training dataset is empty.",
                output_dir=config.output_dir,
                model_name=config.base_model,
                finetune_method=config.finetune_method,
            )

        for idx, rec in enumerate(records, start=1):
            _validate_record(rec, idx)

        try:
            _ensure_dependencies()
        except Exception as e:
            return _mock_training(config, str(e))

        if set_seed is not None:
            set_seed(config.seed)

        if torch is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        model, tokenizer, effective_method, device_name = _load_model_and_tokenizer(config)

        train_ds, eval_ds = _prepare_datasets(
            records=records,
            tokenizer=tokenizer,
            max_length=config.max_length,
            seed=config.seed,
            eval_split_ratio=config.eval_split_ratio,
        )

        if DataCollatorForLanguageModeling is None:
            return _mock_training(config, "DataCollatorForLanguageModeling unavailable")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        loss_callback = create_loss_history_callback()

        use_cuda = device_name == "cuda"
        fp16 = bool(use_cuda)
        bf16 = False

        optim = "adamw_torch"
        if effective_method == "QLoRA" and use_cuda:
            optim = "paged_adamw_8bit"

        training_args = _build_training_arguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=max(config.logging_steps, 1),
            save_strategy="epoch",
            eval_strategy="epoch" if eval_ds is not None and len(eval_ds) > 0 else "no",
            report_to=[],
            fp16=fp16,
            bf16=bf16,
            optim=optim,
            remove_unused_columns=False,
            dataloader_pin_memory=use_cuda,
            seed=config.seed,
            save_total_limit=2,
        )

        trainer = _build_trainer(
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds if eval_ds is not None and len(eval_ds) > 0 else None,
            data_collator=data_collator,
            callbacks=[loss_callback],
        )

        train_start = time.time()
        train_output = trainer.train()
        train_runtime = time.time() - train_start

        out_dir = Path(config.output_dir)
        _safe_mkdir(out_dir)

        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        metrics = {
            "train_runtime": round(train_runtime, 2),
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds is not None else 0,
            "device": device_name,
            "effective_method": effective_method,
        }

        try:
            train_metrics = train_output.metrics or {}
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
                else:
                    metrics[k] = v
        except Exception:
            pass

        try:
            if eval_ds is not None and len(eval_ds) > 0:
                eval_metrics = trainer.evaluate()
                for k, v in eval_metrics.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
                    else:
                        metrics[k] = v
        except Exception:
            pass

        note = "Real training run completed."
        if config.finetune_method == "QLoRA" and device_name != "cuda":
            note = "QLoRA requested, but this device is not CUDA; fell back to LoRA."

        return TrainResult(
            success=True,
            message="Training completed successfully.",
            output_dir=config.output_dir,
            model_name=config.base_model,
            finetune_method=effective_method,
            num_train_samples=len(train_ds),
            num_eval_samples=len(eval_ds) if eval_ds is not None else 0,
            loss_history=loss_callback.loss_history,
            metrics=metrics,
            note=note,
        )

    except Exception as e:
        return TrainResult(
            success=False,
            message=f"Training failed: {e}",
            output_dir=config.output_dir,
            model_name=config.base_model,
            finetune_method=config.finetune_method,
            loss_history=[],
            metrics={},
            note="See exception details in message.",
        )


# ------------------------------------------------------------
# Real evaluation
# ------------------------------------------------------------
def _load_model_for_generation(
    base_model: str,
    model_path: str,
):
    _ensure_dependencies()
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not available.")

    device, dtype, device_name = detect_device()
    base_model_id = normalize_model_name(base_model)
    model_path = resolve_local_model_path(model_path)
    model_dir = Path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=str(CACHE_DIR),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    load_kwargs: Dict[str, Any] = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device_name in {"cuda", "mps"} and dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    if model_dir.exists() and (model_dir / "adapter_config.json").exists():
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapter checkpoints.")
        base_model_obj = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
        model = PeftModel.from_pretrained(base_model_obj, str(model_dir))
    elif model_dir.exists() and (model_dir / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)

    if device is not None:
        model.to(device)

    model.eval()
    return model, tokenizer, device_name


def _generate_prediction(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    template: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
    do_sample: bool,
    num_beams: int,
    seed: int,
) -> Tuple[str, str]:
    prompt = build_generation_prompt(instruction, input_text, template=template)

    if set_seed is not None:
        set_seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": max(temperature, 1e-5),
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        prediction = decoded[len(prompt):].strip()
    else:
        prediction = decoded.strip()

    return prompt, prediction


def evaluate_dataset_split(
    split_path: str,
    base_model: str,
    model_path: str,
    template: str = "alpaca",
    max_eval_samples: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
    num_beams: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"Split file is empty: {split_path}")

    evaluated_records = records[: max(1, max_eval_samples)]
    model, tokenizer, device_name = _load_model_for_generation(base_model, model_path)

    examples: List[Dict[str, Any]] = []
    exact_match_count = 0
    token_acc_total = 0.0
    pred_len_total = 0.0
    ref_len_total = 0.0

    for idx, sample in enumerate(evaluated_records):
        instruction = _normalize_text(sample.get("instruction"))
        input_text = _normalize_text(sample.get("input"))
        reference = _normalize_text(sample.get("output"))

        prompt, prediction = _generate_prediction(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            input_text=input_text,
            template=template,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_beams=num_beams,
            seed=seed + idx,
        )

        em = _exact_match(reference, prediction)
        ta = _token_accuracy(reference, prediction)

        exact_match_count += int(em)
        token_acc_total += ta
        pred_len_total += len(_tokenize_for_metric(prediction))
        ref_len_total += len(_tokenize_for_metric(reference))

        examples.append(
            {
                "index": idx,
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "exact_match": em,
                "token_accuracy": round(ta, 4),
                "prediction_length": len(_tokenize_for_metric(prediction)),
                "reference_length": len(_tokenize_for_metric(reference)),
                "prompt": prompt,
            }
        )

    n = len(evaluated_records)
    metrics = {
        "exact_match_accuracy": round(exact_match_count / max(n, 1), 4),
        "token_accuracy": round(token_acc_total / max(n, 1), 4),
        "avg_prediction_length": round(pred_len_total / max(n, 1), 2),
        "avg_reference_length": round(ref_len_total / max(n, 1), 2),
        "evaluated_samples": n,
        "device": device_name,
    }

    return {
        "success": True,
        "split_name": path.stem,
        "total_split_samples": len(records),
        "evaluated_samples": n,
        "truncated": len(evaluated_records) < len(records),
        "metrics": metrics,
        "examples": examples,
        "message": "Evaluation completed successfully.",
        "generation_config": {
            "template": template,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "seed": seed,
            "max_eval_samples": max_eval_samples,
        },
        "model": {
            "base_model": base_model,
            "model_path": model_path,
        },
    }


# ------------------------------------------------------------
# DeepSeek API assistant / inference
# ------------------------------------------------------------

def _resolve_deepseek_api_key(api_key: str = "") -> str:
    resolved = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not resolved:
        raise RuntimeError(
            "Missing DeepSeek API key. Enter it in the frontend or set DEEPSEEK_API_KEY."
        )
    return resolved


def _deepseek_chat_completion(
    messages: List[Dict[str, str]],
    api_key: str = "",
    model: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 180,
) -> Dict[str, Any]:
    resolved_api_key = _resolve_deepseek_api_key(api_key)
    resolved_model = (model or DEEPSEEK_MODEL).strip()
    base_url = DEEPSEEK_API_BASE_URL.rstrip("/")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    context = None
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"DeepSeek API request failed: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"DeepSeek API request failed: {e}") from e
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek API returned no choices.")

    message = choices[0].get("message") or {}
    answer = str(message.get("content") or "").strip()

    return {
        "success": True,
        "answer": answer,
        "provider": "deepseek",
        "model": resolved_model,
        "usage": data.get("usage", {}),
    }


def chat_with_assistant(
    messages: List[Dict[str, str]],
    evaluation_context: Dict[str, Any] | None = None,
    max_new_tokens: int = 256,
    api_key: str = "",
    model: str = "",
    temperature: float = 0.3,
) -> Dict[str, Any]:
    if evaluation_context is None:
        evaluation_context = {}

    question = messages[-1]["content"] if messages else ""
    chat_messages = build_chat_messages_from_context(
        question=question,
        evaluation_context=evaluation_context,
    )

    if messages and messages[-1].get("role") == "user":
        chat_messages[-1] = {
            "role": "user",
            "content": (
                f"Workflow context:\n{json.dumps(evaluation_context, ensure_ascii=False, indent=2)}\n\n"
                f"User question:\n{messages[-1].get('content', '')}"
            ),
        }

    return _deepseek_chat_completion(
        messages=chat_messages,
        api_key=api_key,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def run_deepseek_inference(
    prompt: str,
    api_key: str = "",
    model: str = "",
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    return _deepseek_chat_completion(
        messages=messages,
        api_key=api_key,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
