from __future__ import annotations

import csv
import io
import json
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.core.trainer import (
    TrainConfig,
    build_train_config,
    evaluate_dataset_split,
    start_training,
    validate_dataset,
    chat_with_assistant,
    run_deepseek_inference,
)

app = FastAPI(title="LLM Fine-tuning Backend", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["instruction", "input", "output"]

training_state: Dict[str, Any] = {
    "status": "idle",
    "message": "Ready.",
    "config": None,
    "result": None,
}
state_lock = threading.Lock()

dataset_state: Dict[str, Any] = {
    "source_filename": None,
    "source_format": None,
    "raw_count": 0,
    "split_done": False,
    "split_ratio": {"train": 0.7, "validation": 0.15, "test": 0.15},
    "split_counts": {"train": 0, "validation": 0, "test": 0},
    "split_paths": {"train": None, "validation": None, "test": None},
    "report": None,
}
dataset_lock = threading.Lock()


class DatasetValidateRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset file")


class TrainRequest(BaseModel):
    base_model: str
    dataset_path: str
    finetune_method: str = "LoRA"
    output_dir: str = "./outputs/finetuned_model"

    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    lora_rank: int = 8
    max_length: int = 512


class EvaluateRequest(BaseModel):
    split_name: str = Field(default="validation", description="train | validation | test")
    base_model: str
    model_path: str
    template: str = "alpaca"

    max_eval_samples: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 256
    repetition_penalty: float = 1.05
    do_sample: bool = True
    num_beams: int = 1
    seed: int = 42


class AssistantMessage(BaseModel):
    role: str
    content: str


class AssistantChatRequest(BaseModel):
    messages: List[AssistantMessage]
    evaluation_context: Dict[str, Any] = Field(default_factory=dict)
    max_new_tokens: int = 256
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.3


class InferenceRequest(BaseModel):
    prompt: str
    api_key: str = ""
    model: str = "deepseek-chat"
    system_prompt: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.7


def set_state(**kwargs: Any) -> None:
    with state_lock:
        training_state.update(kwargs)


def get_state() -> Dict[str, Any]:
    with state_lock:
        return dict(training_state)


def set_dataset_state(**kwargs: Any) -> None:
    with dataset_lock:
        dataset_state.update(kwargs)


def get_dataset_state() -> Dict[str, Any]:
    with dataset_lock:
        return dict(dataset_state)


def _normalize_record(record: Dict[str, Any], instruction_override: str = "") -> Dict[str, str]:
    return {
        "instruction": instruction_override or str(record.get("instruction", "")).strip(),
        "input": str(record.get("input", "")).strip(),
        "output": str(record.get("output", "")).strip(),
    }


def _required_columns_for_upload(instruction_override: str = "") -> List[str]:
    if instruction_override:
        return ["input", "output"]
    return REQUIRED_COLUMNS


def _validate_required_fields(
    record: Dict[str, Any],
    idx: int,
    instruction_override: str = "",
) -> None:
    required_columns = _required_columns_for_upload(instruction_override)
    missing = [col for col in required_columns if col not in record]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Sample {idx} is missing fields: {', '.join(missing)}",
        )


def load_dataset_records(
    filename: str,
    content: bytes,
    instruction_override: str = "",
) -> Tuple[List[Dict[str, str]], str]:
    suffix = Path(filename).suffix.lower()
    required_columns = _required_columns_for_upload(instruction_override)

    if suffix == ".csv":
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))

        if reader.fieldnames is None:
            raise HTTPException(status_code=400, detail="CSV file has no header.")

        missing_cols = [c for c in required_columns if c not in reader.fieldnames]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"CSV is missing required columns: {', '.join(missing_cols)}",
            )

        records: List[Dict[str, str]] = []
        for row in reader:
            records.append(_normalize_record(row, instruction_override))

        return records, "csv"

    if suffix == ".jsonl":
        text = content.decode("utf-8-sig")
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"JSONL parse failed at line {line_no}: {e}",
                )
            if not isinstance(obj, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"JSONL line {line_no} must be an object.",
                )
            _validate_required_fields(obj, line_no, instruction_override)
            records.append(_normalize_record(obj, instruction_override))
        return records, "jsonl"

    if suffix == ".json":
        text = content.decode("utf-8-sig")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON parse failed: {e}")

        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
            records = obj["data"]
        else:
            raise HTTPException(
                status_code=400,
                detail='JSON must be a list[object] or {"data": [...]}',
            )

        if not all(isinstance(x, dict) for x in records):
            raise HTTPException(status_code=400, detail="Each JSON sample must be an object.")

        normalized: List[Dict[str, str]] = []
        for idx, rec in enumerate(records, start=1):
            _validate_required_fields(rec, idx, instruction_override)
            normalized.append(_normalize_record(rec, instruction_override))

        return normalized, "json"

    raise HTTPException(status_code=400, detail="Only .csv / .json / .jsonl files are supported.")


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _unique_upload_path(original_name: str) -> Path:
    candidate = UPLOAD_DIR / original_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        new_candidate = UPLOAD_DIR / f"{stem}_{idx}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        idx += 1


def split_records(
    records: List[Dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    if not records:
        return {"train": [], "validation": [], "test": []}

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 1:
        return {"train": shuffled, "validation": [], "test": []}

    train_n = max(1, int(n * train_ratio))
    val_n = max(0, int(n * val_ratio))
    test_n = n - train_n - val_n
    if test_n < 0:
        test_n = 0

    if train_n + val_n + test_n < n:
        train_n += n - (train_n + val_n + test_n)

    train = shuffled[:train_n]
    val = shuffled[train_n:train_n + val_n]
    test = shuffled[train_n + val_n:]

    if len(train) == 0 and n > 0:
        train = [shuffled[0]]
        rest = shuffled[1:]
        val = rest[:1]
        test = rest[1:]

    return {"train": train, "validation": val, "test": test}


def build_dataset_report(
    records: List[Dict[str, str]],
    source_format: str,
    split_done: bool,
    split_counts: Dict[str, int],
    split_paths: Dict[str, str | None],
    split_ratio: Dict[str, float],
) -> Dict[str, Any]:
    total = len(records)

    avg_lengths = {}
    empty_counts = {}
    for key in REQUIRED_COLUMNS:
        lens = [len((r.get(key) or "").strip()) for r in records]
        avg_lengths[key] = round(sum(lens) / len(lens), 2) if lens else 0
        empty_counts[key] = sum(1 for r in records if not (r.get(key) or "").strip())

    preview = records[:3]

    return {
        "source_format": source_format,
        "num_samples": total,
        "split_done": split_done,
        "split_ratio": split_ratio,
        "split_counts": split_counts,
        "split_paths": split_paths,
        "columns": REQUIRED_COLUMNS,
        "required_columns": REQUIRED_COLUMNS,
        "avg_field_lengths": avg_lengths,
        "empty_field_counts": empty_counts,
        "preview": preview,
        "note": "Dataset has been normalized and randomly split into train / validation / test.",
    }


def run_training_in_background(config: TrainConfig) -> None:
    try:
        set_state(
            status="running",
            message="Training started.",
            config=config.__dict__,
            result=None,
        )

        result = start_training(config)

        if result.success:
            set_state(
                status="success",
                message=result.message,
                result=result.__dict__,
            )
        else:
            set_state(
                status="failed",
                message=result.message,
                result=result.__dict__,
            )
    except Exception as e:
        set_state(
            status="failed",
            message=f"Unexpected error: {e}",
            result=None,
        )


@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    instruction: str = Form(default=""),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl"}:
        raise HTTPException(
            status_code=400,
            detail="Only .csv / .json / .jsonl files are supported.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    instruction_override = instruction.strip()
    records, source_format = load_dataset_records(
        file.filename,
        content,
        instruction_override=instruction_override,
    )

    if not records:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    split_result = split_records(records, 0.7, 0.15, 0.15, seed=42)

    base_name = Path(file.filename).stem
    train_path = _unique_upload_path(f"{base_name}_train.jsonl")
    val_path = _unique_upload_path(f"{base_name}_validation.jsonl")
    test_path = _unique_upload_path(f"{base_name}_test.jsonl")

    write_jsonl(train_path, split_result["train"])
    write_jsonl(val_path, split_result["validation"])
    write_jsonl(test_path, split_result["test"])

    split_counts = {
        "train": len(split_result["train"]),
        "validation": len(split_result["validation"]),
        "test": len(split_result["test"]),
    }
    split_paths = {
        "train": str(train_path.resolve()),
        "validation": str(val_path.resolve()),
        "test": str(test_path.resolve()),
    }
    split_ratio = {"train": 0.7, "validation": 0.15, "test": 0.15}

    report = build_dataset_report(
        records=records,
        source_format=source_format,
        split_done=True,
        split_counts=split_counts,
        split_paths=split_paths,
        split_ratio=split_ratio,
    )

    set_dataset_state(
        source_filename=file.filename,
        source_format=source_format,
        raw_count=len(records),
        split_done=True,
        split_ratio=split_ratio,
        split_counts=split_counts,
        split_paths=split_paths,
        report=report,
    )

    return {
        "success": True,
        "message": "Dataset uploaded successfully.",
        "filename": file.filename,
        "source_format": source_format,
        "dataset_report": report,
        "split_paths": split_paths,
        "dataset_path": split_paths["train"],
        "size_bytes": len(content),
    }


@app.get("/dataset-splits")
def api_dataset_splits():
    return get_dataset_state()


@app.post("/validate-dataset")
def api_validate_dataset(req: DatasetValidateRequest):
    return validate_dataset(req.dataset_path)


@app.post("/train")
def api_train(req: TrainRequest):
    current = get_state()
    if current["status"] == "running":
        raise HTTPException(status_code=409, detail="A training job is already running.")

    try:
        config = build_train_config(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dataset_check = validate_dataset(config.dataset_path)
    if not dataset_check["valid"]:
        raise HTTPException(status_code=400, detail=dataset_check["message"])

    thread = threading.Thread(
        target=run_training_in_background,
        args=(config,),
        daemon=True,
    )
    thread.start()

    set_state(
        status="running",
        message="Training job submitted.",
        config=config.__dict__,
        result=None,
    )

    return {
        "success": True,
        "message": "Training started.",
        "status": "running",
        "config": config.__dict__,
    }


@app.get("/training-status")
def api_training_status():
    return get_state()


@app.post("/evaluate")
def api_evaluate(req: EvaluateRequest):
    ds = get_dataset_state()
    split_paths = ds.get("split_paths") or {}
    split_path = split_paths.get(req.split_name)

    if req.split_name not in {"train", "validation", "test"}:
        raise HTTPException(status_code=400, detail="split_name must be train / validation / test.")

    if not split_path:
        raise HTTPException(status_code=400, detail=f"No dataset split found for {req.split_name}.")

    try:
        result = evaluate_dataset_split(
            split_path=split_path,
            base_model=req.base_model,
            model_path=req.model_path,
            template=req.template,
            max_eval_samples=req.max_eval_samples,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty,
            do_sample=req.do_sample,
            num_beams=req.num_beams,
            seed=req.seed,
        )
        result["split_name"] = req.split_name
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


@app.post("/assistant/chat")
def api_assistant_chat(req: AssistantChatRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        result = chat_with_assistant(
            messages=messages,
            evaluation_context=req.evaluation_context,
            max_new_tokens=req.max_new_tokens,
            api_key=req.api_key,
            model=req.model,
            temperature=req.temperature,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant failed: {e}")


@app.post("/inference/deepseek")
def api_deepseek_inference(req: InferenceRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")

    try:
        return run_deepseek_inference(
            prompt=req.prompt,
            api_key=req.api_key,
            model=req.model,
            system_prompt=req.system_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
