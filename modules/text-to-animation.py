import torch
import re
from diffusers import AnimateDiffPipeline, MotionAdapter, TextToVideoSDPipeline, DiffusionPipeline
from diffusers.utils import export_to_gif
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

def log(message):
    print(f"[DEBUG] {message}")

def can_handle(model_name):
    """Prüft auf unterstützte Text-to-Animation-Modelle"""
    patterns = [
        r"animatediff",
        r"zeroscope",
        r"text-to-video",
        r"toonmaker"
    ]
    return any(re.search(p, model_name, re.IGNORECASE) for p in patterns)

def generate(
    model_name, 
    prompt, 
    output_file, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    num_frames=16,
    height=512,
    width=512,
    **kwargs  # Nimmt alle zusätzlichen Parameter entgegen
):
    """
    Generiert Animationen mit verschiedenen Architekturen
    """
    try:
        # Gemeinsame Parameter
        seed = kwargs.get('seed')
        generator = torch.Generator(device).manual_seed(int(seed)) if seed else None
        guidance = kwargs.get('guidance_scale', 7.5)
        steps = kwargs.get('steps')

        # AnimateDiff-Modelle
        if "animatediff" in model_name.lower():
            log(f"Initialisiere AnimateDiff-Pipeline für {model_name}")
            
            # Parameter mit Fallbacks
            steps = steps or 25
            adapter = MotionAdapter().to(device, dtype)
            
            # Modell laden
            adapter.load_state_dict(load_file(
                hf_hub_download("ByteDance/AnimateDiff-Lightning", "diffusion_pytorch_model.safetensors"),
                device=device
            ))
            
            pipe = AnimateDiffPipeline.from_pretrained(
                "emilianJR/epiCRealism", 
                motion_adapter=adapter, 
                torch_dtype=dtype
            ).to(device)
            
            output = pipe(
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator
            )

        # Zeroscope-Modelle
        elif "zeroscope" in model_name.lower():
            log(f"Initialisiere Zeroscope-Pipeline für {model_name}")
            steps = steps or 40
            
            pipe = TextToVideoSDPipeline.from_pretrained(
                model_name, 
                torch_dtype=dtype
            ).to(device)
            
            video_frames = pipe(
                prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                generator=generator
            ).frames
            
            # Zweite Stufe für höhere Auflösung
            if "xl" in model_name.lower():
                log("Führe Upscaling durch...")
                pipe = DiffusionPipeline.from_pretrained(
                    "cerspense/zeroscope-v2-xl",
                    torch_dtype=dtype
                ).to(device)
                video_frames = [pipe(frame, generator=generator).images[0] for frame in video_frames]

            output = type('', (), {'frames': [video_frames]})()

        # Andere Video-Modelle
        else:
            log(f"Initialisiere generische Pipeline für {model_name}")
            steps = steps or 50
            
            pipe = TextToVideoSDPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype
            ).to(device)
            
            output = pipe(
                prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            )

        # Ergebnis speichern
        output_file = output_file if output_file.endswith(".gif") else output_file + ".gif"
        export_to_gif(output.frames[0], output_file)
        log(f"Animation gespeichert unter: {output_file}")

    except Exception as e:
        log(f"Fehler bei der Generierung: {str(e)}")
        raise