import re
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

def sanitize_output(output_file, default_extension="jpg"):
    """
    Überprüft und bereinigt den Ausgabe-Dateinamen, indem die richtige Erweiterung hinzugefügt wird,
    falls keine gültige Erweiterung angegeben wurde.
    """
    if not re.search(r'\.(jpg|jpeg|png|gif)$', output_file, re.IGNORECASE):
        # Falls keine gültige Erweiterung vorhanden ist, die Standard-Erweiterung hinzufügen
        output_file += f".{default_extension}"
    return output_file


def log(message):
    print(f"[{__name__}] {message}")

def load_file(model_name, ckpt):
    """ Lädt eine Datei von Hugging Face und lädt sie in Torch """
    try:
        log(f"Starte Download von {ckpt} aus dem Modell {model_name}...")
        file_path = hf_hub_download(model_name, ckpt)
        log(f"Datei erfolgreich heruntergeladen: {file_path}")
        
        log(f"Lade Modell von {file_path} auf Gerät {device}...")
        model = torch.load(file_path, map_location=device)
        log(f"Modell erfolgreich geladen von {file_path}")
        
        return model
    except Exception as e:
        log(f"Fehler beim Laden der Datei {ckpt} für Modell {model_name}: {e}")
        raise


def export_to_gif(frames, output_file, duration=100):
    """ Speichert eine Liste von Frames als GIF """
    if isinstance(frames, torch.Tensor):
        frames = [Image.fromarray(frame.cpu().numpy()) for frame in frames]

    frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    log(f"GIF gespeichert unter {output_file}")
