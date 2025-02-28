# Modulrunner v2.1

Willkommen zu **Modulrunner v2.1** 🚀

Vielen Dank, dass du den Weg hierher gefunden hast und dir die Zeit nimmst, diese Zeilen zu lesen! Dieses Projekt ist das Ergebnis vieler Stunden Arbeit, unzähliger Tests und einer ordentlichen Portion Leidenschaft für KI, Automatisierung und kreative Tools.

## Was ist Modulrunner?

**Modulrunner** ist ein flexibler, modularer Runner für verschiedene KI-Module, der sich darauf konzentriert, die Arbeit mit Machine-Learning-Pipelines zu vereinfachen. Der Fokus liegt dabei auf Text-zu-Bild-Generierung, Bildbearbeitung und anderen multimodalen Aufgaben.

Das System basiert auf **PyTorch** und nutzt **LoRA**-Adapter, um verschiedene Modelle zu kombinieren und individuelle Konfigurationen zu ermöglichen.

## Features

- Unterstützung für CUDA und CPU (je nach Hardware)
- Dynamische Modulauswahl (z.B. Text-zu-Bild mit Littletinies)
- Flexible **kwargs**-Übermittlung zwischen Modulen
- Debug-Modus für detaillierte Logausgaben
- Unterstützung für mehrere Basismodelle in einer Pipeline
- Kombinierbare LoRA-Adapter


## Beispielaufruf

Der Aufruf funktioniert aktuell nur für **Windows-Nutzer** mit einer **CUDA-fähigen GPU** und korrekt installierten Treibern.

```powershell
python main.py alvdansen/littletinies "Bitte generiere mir..." "output_datei" --basemodel fluently/Fluently-XL-v2
```

### Beispiel-Output

```plaintext
========================================
=== Modulrunner v2.1 ========================
========================================

[SYSTEM] Gerät: CUDA
[SYSTEM] Torch-Version: 2.6.0+cu126
[INFO] Modellname: alvdansen/littletinies
[INFO] Prompt: Bitte generiere mir...
[INFO] Output-Dateiname: output_datei
[INFO] Empfange kwargs: --device cuda --steps 25 --nctx 2048 --format auto --sampler euler_a --debug False --basemodel fluently/Fluently-XL-v2
[INFO] Basemodel: fluently/Fluently-XL-v2

[MODUL] Aktives Modul: text-to-image-littletinies
[AUSGABE] Datei: output_datei.png
[modules.utils] Empfange kwargs: {'user_args': {'device': 'cuda', 'steps': 25, 'seed': None, 'nctx': 2048, 'format': 'auto', 'sampler': 'euler_a', 'debug': False}, 'basemodel': 'fluently/Fluently-XL-v2'}
[modules.utils] Verwendetes Basis-Modell: fluently/Fluently-XL-v2
[modules.utils] Konfiguration: fluently/Fluently-XL-v2 + alvdansen/littletinies (LoRA)
Loading pipeline components...: 100%|████████████████████████████████████████████████████| 7/7 [00:00<00:00,  7.68it/s]
```

## Community?

Ich freue mich über jeden, der Interesse an diesem Projekt hat – sei es durch Feedback, Pull Requests oder einfach nur ein freundliches "Hallo" in den Issues! Wenn du eine passende Community kennst, die sich mit **generativer KI** oder **Rust-Modding** (denn irgendwo kommt das sicher auch noch dazu 😏) beschäftigt, dann lass es mich bitte wissen!

## ToDo

- Mehr Module integrieren (Audio, Video, Text)
- Interaktive Web-GUI
- Config-Dateien für Pipelines
- Bessere Fehlerbehandlung
- Anbindung an Discord oder Telegram

## Mitmachen

Ich bin offen für jede Art von Beitrag – sei es Code, Doku oder einfach nur Ideen. Schreibt mich einfach an!

---

Bleibt neugierig und probiert immer alles aus – nur so entstehen die besten Ideen! ❤️

Alex (aka **Brain2k12**)

---

"Warum selber denken, wenn die KI das eh besser kann?" 😉

