import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter, DiffusionPipeline
from diffusers.utils import export_to_gif
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import re
# Verhindert das Anzeigen von Symlink-Warnungen
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def log(message):
    print(f"[DEBUG] {message}")

def can_handle(model_name):
    """
    Prüft, ob das Modell ein AnimateLCM-Modell ist.
    Überprüft den Modellnamen auf das Vorhandensein von 'animatelcm'.
    """
    try:
        return "animatelcm" in model_name.lower()
    except Exception as e:
        log(f"Konnte Model-Infos nicht abrufen: {e}")
        return False

def generate(model_name, prompt, output_file, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **kwargs):
    """
    Generiert Animationen für das Modell wangfuyun/AnimateLCM
    """
    if not can_handle(model_name):
        log(f"Modell {model_name} wird nicht unterstützt")
        return

    try:
        log(f"Initialisiere AnimateLCM-Pipeline für {model_name}")

        # Lade den Motion Adapter und das Modell
        adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype)
        pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=dtype)
        
        # Setze den Scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

        # Lade LoRA-Gewichte
        pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
        pipe.set_adapters(["lcm-lora"], [0.8])

        # VAE Slicing und CPU Offloading aktivieren
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

        # Definiere die Anzahl der Frames
        num_frames = 30  # Beispielwert

        # Generiere die Frames
        output = pipe(
            prompt=prompt,
            negative_prompt="bad quality, worse quality, low resolution",
            num_frames=num_frames,
            guidance_scale=2.0,
            num_inference_steps=6,
            generator=torch.Generator(device).manual_seed(0),
        )

        frames = output.frames[0]

        # Speichere das GIF
        output_file = output_file if output_file.endswith(".gif") else output_file + ".gif"
        export_to_gif(frames, output_file)
        log(f"Animation gespeichert unter: {output_file}")
    
    except Exception as e:
        log(f"Fehler bei der Generierung: {str(e)}")
        raise
