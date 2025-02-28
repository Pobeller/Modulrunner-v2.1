import os
import torch
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from .utils import log, sanitize_output

QUANT_PRIORITY = ["Q4_K_M", "Q5_K_M", "Q5_K_S", "Q4_K_S", "Q3_K_M", "Q2_K"]

def can_handle(model_name):
    return any([
        model_name.endswith(".gguf"),
        "gguf" in model_name.lower(),
        "llama" in model_name.lower()
    ])

def handle_parameters(raw_params):
    """Verarbeitet Parameter in beiden Formaten (CLI-String und Dict)"""
    if isinstance(raw_params, dict):
        return {
            'max_tokens': int(raw_params.get('max_length', 512)),
            'temperature': float(raw_params.get('temperature', 0.7).replace(',', '.')),
            'n_ctx': int(raw_params.get('nctx', 2048))
        }
    
    if isinstance(raw_params, str):
        params = {}
        for part in raw_params.split("--"):
            if not part.strip(): continue
            key_val = part.strip().split(" ", 1)
            if len(key_val) == 2:
                key, val = key_val
                params[key] = val.replace(",", ".")
        return {
            'max_tokens': int(params.get('max_length', 512)),
            'temperature': float(params.get('temperature', 0.7)),
            'n_ctx': int(params.get('nctx', 2048))
        }
    
    return {'max_tokens': 512, 'temperature': 0.7, 'n_ctx': 2048}

def get_model_path(model_name):
    """Handhabt Modellpfad mit Cache und Download"""
    if model_name.endswith(".gguf") and os.path.isfile(model_name):
        return model_name
    
    try:
        # Cache durchsuchen
        cache_dir = Path(os.path.expanduser('~'))/'.cache'/'huggingface'/'hub'
        repo_part = model_name.replace('/', '--')
        
        for quant in QUANT_PRIORITY:
            for path in cache_dir.glob(f"**/*{quant}*.gguf"):
                if repo_part in str(path):
                    return str(path)
        
        # Fallback: Download
        return hf_hub_download(
            repo_id=model_name,
            filename=f"{model_name.split('/')[-1]}.{QUANT_PRIORITY[0]}.gguf",
            revision="main"
        )
    
    except Exception as e:
        log(f"Modellzugriffsfehler: {str(e)}")
        raise

def generate(
    model_name,
    prompt,
    output_file,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    **kwargs
):
    try:
        # Debug: Originalparameter loggen
        log(f"[DEBUG] Eingangsparameter - output_file: {output_file}")
        log(f"[DEBUG] Alle kwargs: {kwargs}")

        # Parameterverarbeitung
        params = handle_parameters(kwargs.get('user_args', ''))
        log(f"[DEBUG] Verarbeitete Parameter: {params}")

        # Modell laden
        model_path = get_model_path(model_name)
        log(f"[DEBUG] Modellpfad: {model_path}")

        # Initialisierung
        llm = Llama(
            model_path=model_path,
            n_ctx=params['n_ctx'],
            n_gpu_layers=-1 if device == "cuda" else 0,
            verbose=False
        )

        # Generierung
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=params['max_tokens'],
            temperature=params['temperature'],
            stream=False
        )

 

        # Pfadbearbeitung
        output_path = Path(output_file)
    

        output_path = output_path.with_suffix('.txt')
        

        # Dateischreiben
        log(f"[DEBUG] Schreibe in: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            content = response["choices"][0]["text"].strip()
            f.write(content)
            log(f"[DEBUG] Inhalt: {content[:50]}...")  # Erste 50 Zeichen loggen

        log(f"Erfolgreich generiert: {output_path}")

        

    except Exception as e:
        log(f"[DEBUG] Fehler aufgetreten in: {__file__}")
        log(f"[DEBUG] Exception Typ: {type(e).__name__}")
        log(f"KRITISCHER FEHLER: {str(e)}")
        raise