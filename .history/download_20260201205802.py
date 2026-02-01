from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="C:/hf_models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=1   # ⚠️ 关键：降低并发，减少失败
)