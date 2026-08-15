from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def _h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _format_report_rows(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "<p class='muted'>No data.</p>"

    header = "".join(f"<th>{_h(col)}</th>" for col in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_h(row.get(col, ''))}</td>" for col in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_training_report_html(
    status_payload: Dict[str, Any],
    dataset_report: Dict[str, Any],
    evaluation_output: Dict[str, Any] | None,
) -> str:
    result = status_payload.get("result") or {}
    config = status_payload.get("config") or {}
    metrics = result.get("metrics") or {}
    split_counts = dataset_report.get("split_counts") or {}
    split_ratio = dataset_report.get("split_ratio") or {}
    loss_history = result.get("loss_history") or []
    eval_result = evaluation_output or {}
    eval_metrics = eval_result.get("metrics") or {}
    eval_examples = eval_result.get("examples") or []
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    loss_rows = [
        {"step": item.get("step", ""), "loss": item.get("loss", "")}
        for item in loss_history
    ]
    eval_rows = [
        {
            "index": ex.get("index", ""),
            "input": ex.get("input", ""),
            "reference": ex.get("reference", ""),
            "prediction": ex.get("prediction", ""),
            "exact_match": ex.get("exact_match", ""),
            "token_accuracy": ex.get("token_accuracy", ""),
        }
        for ex in eval_examples[:20]
    ]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>LLM SFT Training Report</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2937;
      background: #f7f7f4;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 28px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fbfbfa; }}
    .metric strong {{ display: block; font-size: 12px; color: #6b7280; margin-bottom: 4px; }}
    .metric span {{ font-size: 18px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>LLM SFT Training Report</h1>
  <p class="muted">Generated at {_h(generated_at)}</p>

  <h2>Training Summary</h2>
  <div class="grid">
    <div class="metric"><strong>Status</strong><span>{_h(status_payload.get("status", "-"))}</span></div>
    <div class="metric"><strong>Model</strong><span>{_h(result.get("model_name") or config.get("base_model") or "-")}</span></div>
    <div class="metric"><strong>Method</strong><span>{_h(result.get("finetune_method") or config.get("finetune_method") or "-")}</span></div>
    <div class="metric"><strong>Output</strong><span>{_h(result.get("output_dir") or config.get("output_dir") or "-")}</span></div>
  </div>
  <p>{_h(status_payload.get("message", ""))}</p>

  <h2>Dataset</h2>
  <div class="grid">
    <div class="metric"><strong>Total Samples</strong><span>{_h(dataset_report.get("num_samples", "-"))}</span></div>
    <div class="metric"><strong>Train</strong><span>{_h(split_counts.get("train", "-"))}</span></div>
    <div class="metric"><strong>Validation</strong><span>{_h(split_counts.get("validation", "-"))}</span></div>
    <div class="metric"><strong>Test</strong><span>{_h(split_counts.get("test", "-"))}</span></div>
  </div>
  <p class="muted">Split ratio: train={_h(split_ratio.get("train", "-"))}, validation={_h(split_ratio.get("validation", "-"))}, test={_h(split_ratio.get("test", "-"))}</p>

  <h2>Training Metrics</h2>
  <div class="grid">
    <div class="metric"><strong>Train Runtime</strong><span>{_h(metrics.get("train_runtime", "-"))}</span></div>
    <div class="metric"><strong>Train Samples</strong><span>{_h(result.get("num_train_samples", metrics.get("train_samples", "-")))}</span></div>
    <div class="metric"><strong>Eval Samples</strong><span>{_h(result.get("num_eval_samples", metrics.get("eval_samples", "-")))}</span></div>
    <div class="metric"><strong>Device</strong><span>{_h(metrics.get("device", "-"))}</span></div>
  </div>

  <h2>Loss History</h2>
  {_format_report_rows(loss_rows, ["step", "loss"])}

  <h2>Evaluation</h2>
  <div class="grid">
    <div class="metric"><strong>Exact Match</strong><span>{_h(eval_metrics.get("exact_match_accuracy", "-"))}</span></div>
    <div class="metric"><strong>Token Accuracy</strong><span>{_h(eval_metrics.get("token_accuracy", "-"))}</span></div>
    <div class="metric"><strong>Evaluated Samples</strong><span>{_h(eval_metrics.get("evaluated_samples", "-"))}</span></div>
    <div class="metric"><strong>Eval Device</strong><span>{_h(eval_metrics.get("device", "-"))}</span></div>
  </div>
  {_format_report_rows(eval_rows, ["index", "input", "reference", "prediction", "exact_match", "token_accuracy"])}
</main>
</body>
</html>"""


def save_training_report(html_text: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="LLM Fine-tuning Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Style
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1.2rem;
        }

        .topbar {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(57, 92, 255, 0.12), rgba(0, 180, 216, 0.10));
            border: 1px solid rgba(120, 120, 120, 0.18);
            margin-bottom: 1rem;
        }

        .topbar-title {
            margin: 0;
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 800;
        }

        .topbar-subtitle {
            margin: 0.25rem 0 0 0;
            opacity: 0.78;
            font-size: 0.96rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid rgba(120, 120, 120, 0.22);
            vertical-align: middle;
            margin-left: 0.35rem;
        }
        .pill-idle { background: rgba(120,120,120,0.10); }
        .pill-running { background: rgba(255,193,7,0.14); }
        .pill-success { background: rgba(46,204,113,0.14); }
        .pill-failed { background: rgba(231,76,60,0.14); }

        .section-card {
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(120, 120, 120, 0.18);
            background: rgba(255, 255, 255, 0.03);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
            height: 100%;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }

        .small-note {
            font-size: 0.86rem;
            opacity: 0.75;
            margin-top: 0.4rem;
        }

        .hint-box {
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            background: rgba(58, 133, 255, 0.10);
            border: 1px solid rgba(58, 133, 255, 0.18);
            margin-top: 0.6rem;
        }

        .sidebar-step {
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            border: 1px solid rgba(120,120,120,0.14);
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
        }

        .sidebar-step strong {
            display: block;
            margin-bottom: 0.15rem;
        }

        .divider-soft {
            height: 1px;
            background: rgba(120,120,120,0.18);
            margin: 0.9rem 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(120, 120, 120, 0.14);
            border-radius: 16px;
            padding: 0.7rem 0.9rem;
        }

        /* Hide the internal Streamlit trigger button */
        div[data-testid="stButton"] button.internal-ai-trigger {
            display: none !important;
        }

        /* Floating AI launcher */
        .ai-fab-wrap {
            position: fixed;
            right: 22px;
            bottom: 22px;
            z-index: 99999;
        }
        .ai-fab {
            border: none;
            border-radius: 999px;
            padding: 0.85rem 1.05rem;
            font-weight: 800;
            background: linear-gradient(135deg, #395cff, #00b4d8);
            color: white;
            cursor: pointer;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        }
        .ai-fab:hover {
            opacity: 0.95;
        }

        .chat-bubble-user {
            background: rgba(57, 92, 255, 0.10);
            border: 1px solid rgba(57, 92, 255, 0.16);
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            margin-bottom: 0.5rem;
        }
        .chat-bubble-assistant {
            background: rgba(0, 180, 216, 0.08);
            border: 1px solid rgba(0, 180, 216, 0.14);
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def init_state() -> None:
    defaults = {
        "uploaded_dataset_path": "",
        "uploaded_dataset_name": "",
        "dataset_report": None,
        "split_paths": None,
        "upload_result": None,
        "validation_result": None,
        "train_submit_result": None,
        "training_status": None,
        "last_poll_time": 0.0,
        "last_error": "",
        "inference_output": "",
        "evaluation_output": None,
        "show_instruction": False,
        "show_ai_assistant": False,
        "backend_url": DEFAULT_BACKEND_URL,
        "assistant_messages": [],
        "assistant_answer": "",
        "dataset_instruction": "",
        "deepseek_api_key": "",
        "deepseek_model": "deepseek-chat",
        "inference_system_prompt": "",
        "training_report_html": "",
        "training_report_path": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# API helpers
# ============================================================
class ApiRequestError(Exception):
    pass


class ApiHTTPError(ApiRequestError):
    def __init__(self, status_code: int, detail: Any, body: str = "") -> None:
        super().__init__(str(detail or body or f"HTTP {status_code}"))
        self.status_code = status_code
        self.detail = detail
        self.body = body


def _parse_error_body(body: str) -> Any:
    try:
        data = json.loads(body)
        return data.get("detail", data)
    except Exception:
        return body


def _read_json_response(req: urllib.request.Request, timeout: int) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ApiHTTPError(e.code, _parse_error_body(body), body) from e
    except urllib.error.URLError as e:
        raise ApiRequestError(str(e)) from e

    return json.loads(text)


def api_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _read_json_response(req, timeout)


def api_get(url: str, timeout: int = 30) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    return _read_json_response(req, timeout)


def _encode_multipart(fields: Dict[str, str], files: Dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----llm-sft-{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def upload_dataset_api(
    backend_url: str,
    uploaded_file,
    dataset_instruction: str = "",
) -> Dict[str, Any]:
    url = f"{backend_url}/upload-dataset"
    body, boundary = _encode_multipart(
        fields={"instruction": dataset_instruction.strip()},
        files={
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
        },
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _read_json_response(req, 60)


def validate_dataset_api(backend_url: str, dataset_path: str) -> Dict[str, Any]:
    return api_post(f"{backend_url}/validate-dataset", {"dataset_path": dataset_path})


def train_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/train", payload)


def training_status_api(backend_url: str) -> Dict[str, Any]:
    return api_get(f"{backend_url}/training-status")


def evaluate_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/evaluate", payload)


def assistant_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/assistant/chat", payload, timeout=180)


def inference_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/inference/deepseek", payload, timeout=180)


def refresh_status(backend_url: str) -> None:
    try:
        st.session_state.training_status = training_status_api(backend_url)
        st.session_state.last_poll_time = time.time()
        st.session_state.last_error = ""
    except ApiRequestError as e:
        st.session_state.last_error = str(e)


# ============================================================
# Instruction dialog
# ============================================================
@st.dialog("instruction_dialog")
def instruction_dialog():
    st.markdown(
        """
### What is this platform for?
This is an LLM fine-tuning platform designed for non-technical users. Think of it as a "Model Training and Testing Workbench."

---

### Workflow
1. Upload Dataset  
2. System automatically checks and randomly splits data into train / validation / test  
3. Check Dataset Overview to ensure data accuracy  
4. Click Check Dataset for basic validation  
5. Set training parameters and Start Training  
6. Go to Monitor to track training status and loss curves  
7. Go to Evaluation to test model performance on validation/test sets  
8. Use the AI Assistant in the bottom-right corner to ask for advice after evaluation  

---

### Data Format Requirements
Supported formats:
- CSV
- JSON
- JSONL

CSV must contain these columns:
- input
- output

Instruction can be entered in the sidebar. If the uploaded file also contains
an instruction column, the sidebar value will override it when provided.
        """
    )

    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.show_instruction = False
        st.rerun()


# ============================================================
# AI Assistant dialog
# ============================================================
@st.dialog("AI Assistant")
def ai_assistant_dialog():
    st.markdown("### AI Assistant")
    st.caption("Ask about the latest evaluation result, error patterns, or next-step tuning suggestions.")

    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask something about the evaluation...")
    if user_prompt:
        st.session_state.assistant_messages.append({"role": "user", "content": user_prompt})

        refresh_status(st.session_state.backend_url)
        eval_result = st.session_state.get("evaluation_output") or {}
        training_status = st.session_state.get("training_status") or {}
        dataset_report = st.session_state.get("dataset_report") or {}
        validation_result = st.session_state.get("validation_result") or {}
        assistant_payload = {
            "messages": st.session_state.assistant_messages,
            "evaluation_context": {
                "metrics": eval_result.get("metrics", {}),
                "examples": eval_result.get("examples", [])[:3],
                "generation_config": eval_result.get("generation_config", {}),
                "model": eval_result.get("model", {}),
                "split_name": eval_result.get("split_name", ""),
                "message": eval_result.get("message", ""),
                "training_status": training_status,
                "dataset_report": dataset_report,
                "validation_result": validation_result,
            },
            "max_new_tokens": 256,
            "api_key": st.session_state.deepseek_api_key,
            "model": st.session_state.deepseek_model,
            "temperature": 0.3,
        }

        try:
            resp = assistant_api(st.session_state.backend_url, assistant_payload)
            answer = resp.get("answer", "")
            st.session_state.assistant_messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except ApiRequestError as e:
            st.error(f"Assistant request failed: {e}")

    if st.button("Close", use_container_width=True):
        st.session_state.show_ai_assistant = False
        st.rerun()


# ============================================================
# Floating launcher
# ============================================================
def open_ai_assistant():
    st.session_state.show_ai_assistant = True

def render_floating_launcher() -> None:
    st.button(
        "AI Assistant",
        key="ai_assistant_launcher",
        on_click=open_ai_assistant,
    )


# ============================================================
# Header
# ============================================================
status_payload = st.session_state.training_status or {}
status_value = status_payload.get("status", "idle")
status_class = f"pill-{status_value}" if status_value in {"idle", "running", "success", "failed"} else "pill-idle"

top_left, top_right = st.columns([0.82, 0.18], vertical_alignment="center")

with top_left:
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">LLM Fine-tuning Workbench <span class="status-pill {status_class}">{status_value.upper()}</span></div>
                <div class="topbar-subtitle">Upload, split, validate, train, evaluate, infer, and ask the assistant.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.write("")
    if st.button("Instruction", use_container_width=True):
        st.session_state.show_instruction = True

if st.session_state.show_instruction:
    instruction_dialog()

render_floating_launcher()

if st.session_state.show_ai_assistant:
    ai_assistant_dialog()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Control Panel")
    backend_url = st.text_input("Backend URL", key="backend_url")

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-step"><strong>DeepSeek API</strong>Used by AI Assistant and Inference.</div>',
        unsafe_allow_html=True,
    )
    deepseek_api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        key="deepseek_api_key",
        help="Keep blank only if DEEPSEEK_API_KEY is set in your terminal environment.",
    )
    deepseek_model = st.selectbox(
        "DeepSeek Model",
        ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"],
        key="deepseek_model",
    )

    st.markdown(
        '<div class="sidebar-step"><strong>Step 1 · Dataset</strong>Upload your dataset here.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "json", "jsonl"],
        help="Files can contain input and output fields. JSON can be a list of objects or {data:[...]}.",
    )
    dataset_instruction = st.text_area(
        "Dataset Instruction",
        value=st.session_state.dataset_instruction,
        placeholder="Example: Judge the sentiment of the input text. Only answer positive or negative.",
        help="This instruction is applied to every uploaded sample. Leave blank only if your file already has an instruction field.",
        height=96,
    )
    upload_btn = st.button("Upload to Backend", use_container_width=True)

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-step"><strong>Step 2 · Training</strong>Choose the base model and training parameters, then start training.</div>',
        unsafe_allow_html=True,
    )

    MODEL_CHOICES = {
        "Qwen2.5-0.5B (Base)": "Qwen/Qwen2.5-0.5B",
        "Qwen2.5-0.5B (Instruct)": "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
        "Qwen2.5-1.5B (Instruct)": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
        "Qwen2.5-3B (Instruct)": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
        "Qwen2.5-7B (Instruct)": "Qwen/Qwen2.5-7B-Instruct",
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "Llama-3.1-8B (Instruct)": "meta-llama/Llama-3.1-8B-Instruct",
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Mistral-7B (Instruct)": "mistralai/Mistral-7B-Instruct-v0.3",
        "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "ChatGLM3-6B": "THUDM/chatglm3-6b",
        "InternLM2-7B": "internlm/internlm2-7b",
        "Baichuan2-7B": "baichuan-inc/Baichuan2-7B-Base",
    }

    model_label = st.selectbox("Base Model", list(MODEL_CHOICES.keys()))
    base_model = MODEL_CHOICES[model_label]

    finetune_method = st.selectbox("Fine-tuning Method", ["LoRA", "QLoRA", "Full Fine-tuning"])
    output_dir = st.text_input("Output Directory", value="./outputs/finetuned_model")

    c1, c2 = st.columns(2)
    with c1:
        learning_rate = st.number_input("Learning Rate", value=2e-4, format="%.6f")
        epochs = st.number_input("Epochs", min_value=1, value=3, step=1)
    with c2:
        batch_size = st.number_input("Batch Size", min_value=1, value=4, step=1)
        lora_rank = st.number_input("LoRA Rank", min_value=1, value=8, step=1)

    max_length = st.slider("Max Length", min_value=64, max_value=4096, value=512, step=64)

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        validate_btn = st.button("Check Dataset", use_container_width=True)
    with action_col2:
        train_btn = st.button("Start Fine-tuning", type="primary", use_container_width=True)

    refresh_btn = st.button("Refresh Status", use_container_width=True)

    st.caption("Tip: upload first, then validate, then train.")


# ============================================================
# Sidebar actions
# ============================================================
if upload_btn:
    if uploaded_file is None:
        st.warning("Please choose a dataset file first.")
    else:
        try:
            st.session_state.dataset_instruction = dataset_instruction
            result = upload_dataset_api(backend_url, uploaded_file, dataset_instruction)
            st.session_state.upload_result = result
            st.session_state.uploaded_dataset_path = result["dataset_path"]
            st.session_state.uploaded_dataset_name = result["filename"]
            st.session_state.dataset_report = result.get("dataset_report")
            st.session_state.split_paths = result.get("split_paths")
            st.session_state.validation_result = None
            st.success("Dataset uploaded successfully.")
        except ApiRequestError as e:
            st.session_state.last_error = str(e)
            st.error(f"Upload failed: {e}")

if validate_btn:
    if not st.session_state.uploaded_dataset_path:
        st.warning("Please upload a dataset first.")
    else:
        try:
            result = validate_dataset_api(backend_url, st.session_state.uploaded_dataset_path)
            st.session_state.validation_result = result
            st.success("Dataset validation finished.")
        except ApiRequestError as e:
            st.session_state.last_error = str(e)
            st.error(f"Validation request failed: {e}")

if train_btn:
    if not st.session_state.uploaded_dataset_path:
        st.warning("Please upload a dataset first.")
    else:
        payload = {
            "base_model": base_model,
            "dataset_path": st.session_state.uploaded_dataset_path,
            "finetune_method": finetune_method,
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "lora_rank": lora_rank,
            "max_length": max_length,
        }
        try:
            result = train_api(backend_url, payload)
            st.session_state.train_submit_result = result
            st.session_state.training_status = result
            st.session_state.training_report_html = ""
            st.session_state.training_report_path = ""
            st.success("Training job submitted.")
        except ApiHTTPError as e:
            detail = e.detail
            st.error(f"Training request rejected: {detail or e}")
        except ApiRequestError as e:
            st.session_state.last_error = str(e)
            st.error(f"Training request failed: {e}")

if refresh_btn or st.session_state.training_status is None:
    refresh_status(backend_url)


# ============================================================
# Top metrics
# ============================================================
status_payload = st.session_state.training_status or {}
status_value = status_payload.get("status", "idle")

metric_cols = st.columns(4)
metric_cols[0].metric("Dataset", st.session_state.uploaded_dataset_name or "None")
metric_cols[1].metric("Model", model_label)
metric_cols[2].metric("Method", finetune_method)
metric_cols[3].metric("Status", status_value)


# ============================================================
# Tabs
# ============================================================
setup_tab, monitor_tab, eval_tab, infer_tab = st.tabs(["Setup", "Monitor", "Evaluation", "Inference"])


# ============================================================
# Setup tab
# ============================================================
with setup_tab:
    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        report = st.session_state.get("dataset_report")

        if report:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", report.get("num_samples", 0))
            c2.metric("Source Format", report.get("source_format", "unknown"))
            c3.metric("Columns", len(report.get("columns", [])))

            st.write("**Detected columns:** " + ", ".join(report.get("columns", [])))

            split_counts = report.get("split_counts", {})
            split_ratio = report.get("split_ratio", {})
            st.write("**Split completed:** Yes")
            st.write(
                f"**Split counts:** train={split_counts.get('train', 0)}, "
                f"validation={split_counts.get('validation', 0)}, "
                f"test={split_counts.get('test', 0)}"
            )
            st.write(
                f"**Split ratio:** train={split_ratio.get('train', 0.7)}, "
                f"validation={split_ratio.get('validation', 0.15)}, "
                f"test={split_ratio.get('test', 0.15)}"
            )

            if report.get("note"):
                st.markdown(
                    f'<div class="hint-box">{report.get("note")}</div>',
                    unsafe_allow_html=True,
                )

            avg_lengths = report.get("avg_field_lengths", {})
            if avg_lengths:
                with st.expander("Average Field Lengths", expanded=False):
                    st.json(avg_lengths)

            empty_counts = report.get("empty_field_counts", {})
            if empty_counts:
                with st.expander("Empty Field Counts", expanded=False):
                    st.json(empty_counts)

            with st.expander("Preview", expanded=False):
                st.json(report.get("preview", []))

            with st.expander("Split Paths", expanded=False):
                st.json(report.get("split_paths", {}))
        else:
            st.info("No dataset uploaded yet.")

        if st.session_state.upload_result:
            with st.expander("Upload Result", expanded=False):
                st.json(st.session_state.upload_result)

        st.markdown(
            '<div class="small-note">Uploaded data is normalized to JSONL and randomly split into train / validation / test on the backend.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Training Configuration</div>', unsafe_allow_html=True)

        report = st.session_state.get("dataset_report") or {}
        config_preview = {
            "base_model": base_model,
            "finetune_method": finetune_method,
            "dataset_format": report.get("source_format", ""),
            "num_samples": report.get("num_samples", ""),
            "split_done": report.get("split_done", False),
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "lora_rank": lora_rank,
            "max_length": max_length,
        }
        st.json(config_preview)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Validation</div>', unsafe_allow_html=True)

        if st.session_state.validation_result:
            vr = st.session_state.validation_result
            if vr.get("valid"):
                st.success(vr.get("message", "Dataset is valid."))
            else:
                st.error(vr.get("message", "Dataset is invalid."))

            m1, m2, m3 = st.columns(3)
            m1.metric("Valid", str(vr.get("valid")))
            m2.metric("Samples", vr.get("num_samples", 0))
            m3.metric("Keys", len(vr.get("sample_keys", [])))

            st.caption(", ".join(vr.get("sample_keys", [])) or "No keys returned.")
        else:
            st.info("Click **Check Dataset** after uploading a file.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Submission Preview</div>', unsafe_allow_html=True)
        if st.session_state.train_submit_result:
            st.json(st.session_state.train_submit_result)
        else:
            st.caption("No training request submitted yet.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Monitor tab
# ============================================================
with monitor_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Training Monitor</div>', unsafe_allow_html=True)

    if status_value == "running":
        st.info("训练中")
        st.progress(0.5, text="Fine-tuning is running...")
        st.caption(status_payload.get("message", "Training is running."))
        time.sleep(2)
        refresh_status(backend_url)
        st.rerun()
    elif status_value == "success":
        st.success("训练完成")
        result = status_payload.get("result") or {}
        summary_cols = st.columns(4)
        summary_cols[0].metric("Model", result.get("model_name", "-"))
        summary_cols[1].metric("Method", result.get("finetune_method", "-"))
        summary_cols[2].metric("Train Samples", result.get("num_train_samples", 0))
        summary_cols[3].metric("Eval Samples", result.get("num_eval_samples", 0))

        if st.button("生成报告", type="primary", use_container_width=True):
            html_report = build_training_report_html(
                status_payload=status_payload,
                dataset_report=st.session_state.get("dataset_report") or {},
                evaluation_output=st.session_state.get("evaluation_output"),
            )
            report_path = save_training_report(html_report)
            st.session_state.training_report_html = html_report
            st.session_state.training_report_path = str(report_path)
            try:
                webbrowser.open(report_path.resolve().as_uri())
            except Exception:
                pass
            st.success(f"报告已生成：{report_path}")

        if st.session_state.training_report_html:
            st.caption(st.session_state.training_report_path)
            st.download_button(
                "下载 HTML 报告",
                data=st.session_state.training_report_html,
                file_name=Path(st.session_state.training_report_path).name or "training_report.html",
                mime="text/html",
                use_container_width=True,
            )
            import streamlit.components.v1 as components
            components.html(st.session_state.training_report_html, height=780, scrolling=True)
    elif status_value == "failed":
        st.error("训练失败")
        st.write(status_payload.get("message", "Training failed."))
    else:
        st.info("等待开始训练")
        st.caption("点击左侧 Start Fine-tuning 后，这里会切换为训练中。")

    if st.session_state.last_error:
        st.error(f"Last error: {st.session_state.last_error}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Evaluation tab
# ============================================================
with eval_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evaluation</div>', unsafe_allow_html=True)

    dataset_report = st.session_state.get("dataset_report") or {}

    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.subheader("Evaluation Settings")

        split_options = ["validation", "test", "train"]
        selected_split = st.selectbox("Split", split_options, index=0)

        max_eval_samples = st.slider("Max Eval Samples", min_value=1, max_value=100, value=20, step=1)
        template = st.selectbox("Template", ["alpaca", "qwen", "llama2", "chatml", "custom"], index=0)

        st.write("### Generation Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.01)
        top_k = st.number_input("Top K", min_value=0, value=50, step=1)
        max_new_tokens = st.number_input("Max New Tokens", min_value=16, value=256, step=16)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.05, 0.01)
        do_sample = st.checkbox("Do Sample", value=True)
        num_beams = st.number_input("Num Beams", min_value=1, value=1, step=1)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

        eval_btn = st.button("Run Evaluation", type="primary", use_container_width=True)

    with right:
        st.subheader("Evaluation Result")

        if eval_btn:
            if not dataset_report.get("split_done"):
                st.warning("Please upload and split a dataset first.")
            else:
                payload = {
                    "split_name": selected_split,
                    "base_model": base_model,
                    "model_path": output_dir,
                    "template": template,
                    "max_eval_samples": int(max_eval_samples),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "max_new_tokens": int(max_new_tokens),
                    "repetition_penalty": float(repetition_penalty),
                    "do_sample": bool(do_sample),
                    "num_beams": int(num_beams),
                    "seed": int(seed),
                }
                try:
                    result = evaluate_api(backend_url, payload)
                    st.session_state.evaluation_output = result
                    st.session_state.assistant_messages = []
                    st.session_state.assistant_answer = ""
                except ApiRequestError as e:
                    st.session_state.last_error = str(e)
                    st.error(f"Evaluation failed: {e}")

        eval_result = st.session_state.get("evaluation_output")
        if eval_result:
            st.success(eval_result.get("message", "Evaluation finished."))

            metrics = eval_result.get("metrics", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exact Match", f"{metrics.get('exact_match_accuracy', 0):.4f}")
            m2.metric("Token Accuracy", f"{metrics.get('token_accuracy', 0):.4f}")
            m3.metric("Avg Pred Len", f"{metrics.get('avg_prediction_length', 0):.2f}")
            m4.metric("Eval Samples", metrics.get("evaluated_samples", 0))

            m5, m6 = st.columns(2)
            m5.metric("Avg Ref Len", f"{metrics.get('avg_reference_length', 0):.2f}")
            m6.metric("Device", metrics.get("device", "-"))

            st.write(
                f"**Split:** {eval_result.get('split_name', '-')}"
                f" | **Total in split:** {eval_result.get('total_split_samples', 0)}"
                f" | **Evaluated:** {eval_result.get('evaluated_samples', 0)}"
                f" | **Truncated:** {eval_result.get('truncated', False)}"
            )

            examples = eval_result.get("examples", [])
            if examples:
                st.markdown("### Evaluation Examples")
                for ex in examples[:10]:
                    st.markdown(
                        f"**Sample {ex.get('index', 0)}** "
                        f"| exact_match={ex.get('exact_match', False)} "
                        f"| token_accuracy={ex.get('token_accuracy', 0):.4f}"
                    )
                    st.write("Reference")
                    st.code(ex.get("reference", ""))
                    st.write("Prediction")
                    st.code(ex.get("prediction", ""))
                    st.divider()

                error_examples = [ex for ex in examples if not ex.get("exact_match", False)]
                if error_examples:
                    with st.expander("Error Examples", expanded=False):
                        for ex in error_examples[:5]:
                            st.markdown(f"**Sample {ex.get('index', 0)}**")
                            st.write("Instruction")
                            st.code(ex.get("instruction", ""))
                            if ex.get("input"):
                                st.write("Input")
                                st.code(ex.get("input", ""))
                            st.write("Reference")
                            st.code(ex.get("reference", ""))
                            st.write("Prediction")
                            st.code(ex.get("prediction", ""))
                            st.divider()

            with st.expander("Generation Config", expanded=False):
                st.json(eval_result.get("generation_config", {}))

            with st.expander("Model Config", expanded=False):
                st.json(eval_result.get("model", {}))

            st.markdown("### AI Assistant")
            st.caption("Click the floating button in the lower-right corner to ask follow-up questions.")
        else:
            st.info("Run evaluation to see accuracy and error analysis.")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Inference tab
# ============================================================
with infer_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">DeepSeek Inference Test</div>', unsafe_allow_html=True)

    system_prompt = st.text_area(
        "System Prompt",
        key="inference_system_prompt",
        placeholder="Optional: You are a helpful assistant.",
        height=90,
    )

    prompt = st.text_area(
        "Enter a prompt",
        placeholder="Type a test prompt here...",
        height=120,
    )

    ip1, ip2 = st.columns(2)
    with ip1:
        inference_temperature = st.slider("Inference Temperature", 0.0, 2.0, 0.7, 0.05)
    with ip2:
        inference_max_tokens = st.number_input(
            "Inference Max Tokens",
            min_value=16,
            max_value=4096,
            value=512,
            step=16,
        )

    ic1, ic2 = st.columns([1, 1])
    run_infer = ic1.button("Run Inference", type="primary", use_container_width=True)
    clear_logs = ic2.button("Clear Display", use_container_width=True)

    if clear_logs:
        st.session_state.upload_result = None
        st.session_state.validation_result = None
        st.session_state.train_submit_result = None
        st.session_state.training_status = None
        st.session_state.last_error = ""
        st.session_state.inference_output = ""
        st.session_state.evaluation_output = None
        st.session_state.assistant_messages = []
        st.session_state.assistant_answer = ""
        st.rerun()

    if run_infer:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            payload = {
                "prompt": prompt,
                "api_key": st.session_state.deepseek_api_key,
                "model": st.session_state.deepseek_model,
                "system_prompt": system_prompt,
                "max_new_tokens": int(inference_max_tokens),
                "temperature": float(inference_temperature),
            }
            try:
                result = inference_api(backend_url, payload)
                st.session_state["inference_output"] = result.get("answer", "")
            except ApiHTTPError as e:
                detail = e.detail
                st.error(f"Inference failed: {detail or e}")
            except ApiRequestError as e:
                st.session_state.last_error = str(e)
                st.error(f"Inference request failed: {e}")

    if st.session_state.get("inference_output"):
        st.markdown("### Inference Output")
        st.markdown(st.session_state["inference_output"])

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.caption("Backend: FastAPI | Frontend: Streamlit | Training: local Hugging Face model | Assistant/Inference: DeepSeek API")
