from huggingface_hub import hf_hub_download
import concurrent.futures

with concurrent.futures.ProcessPoolExecutor() as executor:
    base_future = executor.submit(
        hf_hub_download,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        local_dir="models/checkpoints",
    )
    refiner_future = executor.submit(
        hf_hub_download,
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        filename="sd_xl_refiner_1.0.safetensors",
        local_dir="models/checkpoints",
    )

base_future.result()
refiner_future.result()
