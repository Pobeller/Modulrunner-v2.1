import torch
from pathlib import Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from .utils import log  # sanitize_output wird nicht mehr benötigt

VOICE_MAPPING = {
    'default': 0,
    'male': 12,
    'female': 8
}

def can_handle(model_name):
    return model_name.lower() == "sjdata/speecht5_finetuned_common_voice_11_de"

def generate(model_name, prompt, output_file, device=None, dtype=None, **kwargs):
    try:
        log(f"Starte TTS-Generierung mit Parametern: {kwargs}")
        
        # Parameter verarbeiten
        voice = kwargs.get('--voice', 'default').lower()
        speed = float(str(kwargs.get('--speed', 1.0)).replace(',', '.'))
        pitch = int(kwargs.get('--pitch', 0))
        
        # Ausgabepfad erzwingen
        output_path = Path(output_file).with_suffix('.wav')
        log(f"Finaler Ausgabepfad: {output_path}")
        
        # Geräteerkennung
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Modelle laden
        processor = SpeechT5Processor.from_pretrained(model_name)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        
        # Sprecher-Embedding
        speaker_idx = VOICE_MAPPING.get(voice, 0)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(
            embeddings_dataset[speaker_idx]["xvector"]
        ).unsqueeze(0).to(device)
        
        # Textverarbeitung
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            speaking_rate=float(speed),
            pitch_shift=int(pitch)
        ).to(device)
        
        # Generierung
        with torch.no_grad():
            spectrogram = model.generate(**inputs, speaker_embeddings=speaker_embeddings)
            waveform = vocoder(spectrogram)
        
        # Datei schreiben
        sf.write(
            str(output_path),  # Path-Objekt in String konvertieren
            waveform.cpu().numpy().squeeze(),
            samplerate=16000
        )
        
        log(f"Erfolgreich generierte WAV-Datei: {output_path}")

    except KeyError as e:
        log(f"Ungültige Stimme: {voice}. Verfügbare Optionen: {list(VOICE_MAPPING.keys())}")
        raise
    except Exception as e:
        log(f"Fehler: {str(e)}")
        raise