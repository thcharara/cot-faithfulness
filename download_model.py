from huggingface_hub import snapshot_download
import os

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
local_dir = os.path.join(os.getcwd(), "models", "DeepSeek-R1-Distill-Llama-8B")

print(f"Downloading {model_id} to {local_dir}...")
print("This will be ~16GB. It may take a while.")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"\nDone! Files downloaded to: {local_dir}")

# List what was downloaded
for f in sorted(os.listdir(local_dir)):
    size_mb = os.path.getsize(os.path.join(local_dir, f)) / 1e6
    print(f"  {f}: {size_mb:.1f} MB")
