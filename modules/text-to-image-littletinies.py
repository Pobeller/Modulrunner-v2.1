import torch
from diffusers import StableDiffusionXLPipeline
from .utils import log, sanitize_output

def can_handle(model_name):
    return "littletinies" in model_name.lower()

def generate(
    model_name,
    prompt,
    output_file,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    **kwargs
):
    """
    Generiert Bilder mit Basis-Modell + dynamischem LoRA
    """
    try:
        # Basismodell prüfen und setzen
        log(f"Empfange kwargs: {kwargs}")
        base_model = kwargs.get('basemodel', 'stabilityai/stable-diffusion-xl-base-1.0')  # Fallback-Modell
        log(f"Verwendetes Basis-Modell: {base_model}")

        # Parameter aus kwargs
        steps = kwargs.get('steps', 40)
        guidance = kwargs.get('guidance_scale', 7.5)
        lora_scale = kwargs.get('lora_scale', 0.7)
        seed = kwargs.get('seed')

        log(f"Konfiguration: {base_model} + {model_name} (LoRA)")

        # Pipeline initialisieren
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)

        # LoRA mergen
        pipeline.load_lora_weights(model_name)
        pipeline.fuse_lora(lora_scale=lora_scale)

        # Generator für Reproduzierbarkeit
        generator = torch.Generator(device).manual_seed(int(seed)) if seed else None

        # Generierung
        image = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            cross_attention_kwargs={"scale": lora_scale}
        ).images[0]

        # Korrektur der Output-Datei
        if not output_file.endswith(".png"):
            output_file = sanitize_output(output_file, "png")

        # Speichern
        image.save(output_file)
        log(f"Erfolgreich generiert: {output_file}")

    except Exception as e:
        log(f"FEHLER: {str(e)}")
        raise
