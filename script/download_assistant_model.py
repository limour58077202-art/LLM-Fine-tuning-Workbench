from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "models_cache" / "models--Qwen--Qwen2.5-0.5B-Instruct"

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {CACHE_DIR}")

if __name__ == "__main__":
    main()