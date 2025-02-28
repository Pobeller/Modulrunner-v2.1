import os
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from huggingface_hub import model_info
from PIL import Image
import numpy as np

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import re
# Verhindert das Anzeigen von Symlink-Warnungen
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



import cv2
from .utils import log, sanitize_output

def can_handle(model_name):
    """
    Prüft, ob das Modell ein Text-zu-Bild-Modell ist, das ControlNet unterstützt.
    Überprüft Hugging Face Cache auf relevante Tags.
    """
    try:
        # Modell-Informationen abrufen
        info = model_info(model_name)
        tags = info.tags if hasattr(info, "tags") else []
        
        # Überprüfen, ob "ControlNet" oder "MistoLine" in den Tags enthalten ist
        return any(tag in tags for tag in ["ControlNet", "MistoLine"])
    
    except Exception as e:
        log(f"Konnte Model-Infos nicht abrufen: {e}")
        return False


def generate(model_name, prompt, output_file, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16 if torch.cuda.is_available() else torch.float32, image_url=None):
    """
    Generiert ein Bild aus einem Text-Prompt unter Verwendung von ControlNet und StableDiffusionXLControlNetPipeline.
    """
    log(f"Starte Bildgenerierung mit {model_name} auf {device}")

    output_file = sanitize_output(output_file, "png")  # Standardformat PNG setzen

    try:
        # Wenn ein Bild-URL übergeben wird, das Bild laden und für die Pipeline vorbereiten
        if image_url:
            image = load_image(image_url)  # Bild von der URL laden
        else:
            # Beispiel: Standardbild für Bildgenerierung verwenden (hier wird das Bild von der URL geladen)
            image = load_image("input.jpg")
        
        # ControlNet und VAE Modelle laden
        controlnet = ControlNetModel.from_pretrained(
            "TheMistoAI/MistoLine",
            torch_dtype=dtype,
            variant="fp16",
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        
        # Pipeline mit ControlNet und VAE erstellen
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
        )
        pipe.enable_model_cpu_offload()

        # Bildvorverarbeitung (z.B. Kantenextraktion)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)  # RGB-Konvertierung
        image = Image.fromarray(image)

        # Bildgenerierung mit ControlNet
        controlnet_conditioning_scale = 0.5
        negative_prompt = 'low quality, bad quality, sketches'
        
        images = pipe(
            prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        # Bild speichern
        images[0].save(output_file)
        log(f"Bild gespeichert in {output_file}")
        
    except Exception as e:
        log(f"Fehler bei der Bildgenerierung: {e}")
