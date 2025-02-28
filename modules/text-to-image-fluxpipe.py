import torch
import warnings
import sys
import io
from diffusers import DiffusionPipeline
from transformers import logging
from .utils import log, sanitize_output

# Encoding fuer Windows-Konsolen erzwingen
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
#sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Warnungen unterdruecken
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

def can_handle(model_name):
    return "flux.1-schnell" in model_name.lower()

def generate(model_name, prompt, output_file, 
            device="cuda",
            dtype=torch.float16,
            resolution=(512, 512),
            **kwargs):

    try:
        # Debug-Output mit ASCII-Zeichen
        log(f"[PARAMETER] Model: {model_name}")
        log(f"[PARAMETER] Prompt: {prompt}")
        log(f"[PARAMETER] Output File: {output_file}")
        log(f"[PARAMETER] Device: {device}")
        log(f"[PARAMETER] Resolution: {resolution[0]}x{resolution[1]}")
        log(f"[PARAMETER] Kwargs: {kwargs}")

        # Device-Handling
        if ":" in device:
            parts = device.split(":", 1)
            if parts[0] == "cuda" and parts[1] == "cpu":
                log("[WARNUNG] Ungueltige Device-Kombination - verwende CPU")
                device = "cpu"
            else:
                device = f"{parts[0]}:{parts[1].split(':')[0]}"
        else:
            device = device.lower()

        # CUDA-Fallback
        if device.startswith("cuda") and not torch.cuda.is_available():
            log("[WARNUNG] CUDA nicht verfuegbar - wechsle zu CPU")
            device = "cpu"

        log(f"[INFO] Aktives Device: {device}")

        # Pipeline-Initialisierung
        output_file = sanitize_output(output_file, "png")
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            use_safetensors=True
        )

        # Tokenizer-Anpassungen
        if hasattr(pipe, 'tokenizer') and pipe.tokenizer:
            log("[INFO] Passe Tokenizer an")
            for attr in ['add_prefix_space', 'use_fast']:
                if hasattr(pipe.tokenizer, attr):
                    setattr(pipe.tokenizer, attr, False)

        # Device-Konfiguration
        try:
            pipe = pipe.to(device)
            if device.startswith("cuda"):
                pipe.enable_sequential_cpu_offload()
                pipe.enable_attention_slicing(2)
        except Exception as e:
            log(f"[FEHLER] Device-Konfiguration fehlgeschlagen: {str(e)}")
            device = "cpu"
            pipe = pipe.to(device)

        # Callback-Handling
        callback_params = {}
        if hasattr(pipe, 'callback_on_step_end'):
            callback_params = {
                'callback_on_step_end': lambda pipe, step, *_: log(f"Schritt {step+1}/{kwargs.get('steps',20)}"),
                'callback_on_step_end_tensor_inputs': ['latents'],
                'step_callback_interval': 1
            }
        else:
            log("[WARNUNG] Callbacks nicht unterstuetzt")

        # Bildgenerierung
        generator = None
        if 'seed' in kwargs:
            generator = torch.Generator(device=device)
            generator.manual_seed(kwargs['seed'])

        log("[INFO] Starte Bildgenerierung...")
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            image = pipe(
                prompt=prompt,
                num_inference_steps=kwargs.get('steps', 20),
                guidance_scale=kwargs.get('guidance_scale', 3.0),
                height=resolution[0],
                width=resolution[1],
                generator=generator,
                **callback_params
            ).images[0]

        image.save(output_file)
        log(f"[ERFOLG] Bild gespeichert: {output_file}")

    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        log(f"[KRITISCHER FEHLER] {error_msg}")
        raise RuntimeError(f"FLUX-Fehler: {error_msg}") from e