import torch
import os
import re
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import model_info
from .utils import log, sanitize_output

# Cache-Konfiguration
HF_CACHE_HOME = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).absolute()
os.environ["HF_HUB_OFFLINE"] = "1"

def can_handle(model_name):
    """Überprüft ob das Modell Hyper-SD oder SDXL-basiert ist"""
    return "hyper-sd" in model_name.lower() or "sdxl" in model_name.lower()

def generate(model_name, prompt, output_file, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **kwargs):  # Änderung hier: **kwargs hinzufügen
    
    output_file = sanitize_output(output_file, "png")
    
    try:
        # 1. Lade das Basis-SDXL-Modell
        base_model_path = HF_CACHE_HOME / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
        base_snapshot = sorted(base_model_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_snapshot.as_posix(),
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True
        ).to(device)

        # 2. Parameter aus kwargs verarbeiten
        steps = kwargs.get('steps')
        guidance = kwargs.get('guidance_scale')
        seed = kwargs.get('seed')

        # 3. LoRA-Handling mit automatischer Step-Erkennung
        if "hyper-sd" in model_name.lower():
            lora_path = HF_CACHE_HOME / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
            lora_snapshot = sorted(lora_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[0]
            
            pipeline.load_lora_weights(lora_snapshot.as_posix())
            pipeline.fuse_lora()

            # Automatische Step-Erkennung wenn nicht in kwargs angegeben
            if not steps:
                steps = int(re.search(r"(\d+)steps", [f.name for f in lora_snapshot.iterdir() if "steps" in f.name][0]).group(1))
            
            # Guidance-Scale nur setzen wenn nicht in kwargs
            if not guidance:
                guidance = 3.5 if steps < 10 else 7.5

        # 4. Fallback-Werte für SDXL
        steps = steps or 25
        guidance = guidance or 7.5

        # 5. Generator für Reproduzierbarkeit
        generator = None
        if seed:
            generator = torch.Generator(device).manual_seed(int(seed))

        # 6. Generiere Bild mit allen Parametern
        image = pipeline(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        image.save(output_file)
        log(f"Bild erfolgreich generiert: {output_file}")

    except Exception as e:
        log(f"Fehler: {str(e)}")
        raise RuntimeError(f"Generierung fehlgeschlagen: {e}") from e