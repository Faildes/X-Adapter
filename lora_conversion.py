import os
import datetime

from scripts.inference_lora import lora_c

def lora_convert(lora_path, output_path):
    base = "runwayml/stable-diffusion-v1-5"
    sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = "madebyollin/sdxl-vae-fp16-fix"
    adapter = "./checkpoint/X-Adapter/X_Adapter_v1.bin"
    lora_c(lora_path, base, sdxl, vae, adapter, output_path)
    print("Done!")
