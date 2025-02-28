import re
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import model_info
# Import korrigieren (kein relativer Import)
from modules.utils import log, sanitize_output

def can_handle(model_name):
    """
    Prüft, ob das Modell ein Text-zu-Bild-Modell ist.
    """
    try:
        info = model_info(model_name)
        tags = getattr(info, "tags", [])
        return any(tag in tags for tag in ["text-to-image", "stable-diffusion", "diffusion"])
    except Exception as e:
        log(f"Model-Info nicht abrufbar: {e}")
        return False

def generate(model_name, prompt, output_file, device, dtype, **kwargs):
    """
    Generiert ein Bild aus einem Text-Prompt mit optionalen Parametern.
    """
    try:
        # Parameter aus kwargs mit Defaults
        steps = kwargs.get('steps', 25)
        seed = kwargs.get('seed', None)
        sampler = kwargs.get('sampler', 'euler_a')

        log(f"Starte Generierung: {model_name} auf {device}")
        log(f"Parameter: {steps} Schritte, Sampler: {sampler}, Seed: {seed or 'auto'}")

        # Dateinamen bereinigen
        output_file = sanitize_output(kwargs.get('output', output_file), "png")

        # Pipeline mit dtype initialisieren
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(device)

        # Generator für Reproduzierbarkeit
        generator = None
        if seed:
            generator = torch.Generator(device).manual_seed(seed)

        # Generierung mit Parametern
        image = pipeline(
            prompt,
            num_inference_steps=steps,
            generator=generator
        ).images[0]

        image.save(output_file)
        log(f"Erfolgreich gespeichert: {output_file}")

    except Exception as e:
        log(f"Generierungsfehler: {str(e)}")
        raise