import re
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import warnings

# Warnungen unterdrücken
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

def log(message):
    print(f"[DEBUG] {message}")

def can_handle(model_name):
    return "animatediff-lightning" in model_name.lower()

def generate(model_name, prompt, output_file, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **kwargs):
    """
    Generiert eine Animation mit AnimateDiff-Lightning.
    """
    try:
        if not can_handle(model_name):
            log(f"Modell {model_name} nicht unterstützt")
            return False

        log(f"Initialisiere AnimateDiff-Lightning für {model_name}")
        
        # Parameter aus kwargs mit Fallbacks
        steps = kwargs.get('steps')
        seed = kwargs.get('seed')
        guidance = kwargs.get('guidance_scale', 1.0)

        # Schritte aus Modellname oder kwargs
        if not steps:
            match = re.search(r'(\d+)step', model_name.lower())
            steps = int(match.group(1)) if match else 4
        log(f"Nutze Schritte: {steps}")

        # Generator für Reproduzierbarkeit
        generator = None
        if seed:
            generator = torch.Generator(device).manual_seed(int(seed))

        # Modellkomponenten laden
        base_model = "emilianJR/epiCRealism"
        ckpt = f"animatediff_lightning_{steps}step_diffusers.safetensors"
        
        log(f"Lade MotionAdapter: {ckpt}")
        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file(
            hf_hub_download("ByteDance/AnimateDiff-Lightning", ckpt),
            device=device
        ))

        # Pipeline initialisieren
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model, 
            motion_adapter=adapter, 
            torch_dtype=dtype
        ).to(device)

        # Scheduler konfigurieren
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        # Dateiendung sicherstellen
        if not output_file.lower().endswith(".gif"):
            output_file += ".gif"

        # Generierung
        log(f"Starte Animation: '{prompt}'")
        output = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator
        )

        export_to_gif(output.frames[0], output_file)
        log(f"Animation gespeichert: {output_file}")
        return True

    except Exception as e:
        log(f"Fehler: {str(e)}")
        raise