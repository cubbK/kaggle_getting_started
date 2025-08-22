import os
from huggingface_hub import run_uv_job

job = run_uv_job(
    "dinov2_clothes_big_script.py",
    env={"HF_TOKEN": os.environ["HF_TOKEN"]},
    flavor="l4x1",
    timeout=7200,
)

print(job.url)