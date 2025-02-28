import sys
import importlib
import torch
from pathlib import Path

def print_header():
    print("\n" + "=" * 40)
    print("=== Modulrunner v2.1 " + "=" * 24)
    print("=" * 40 + "\n")

def parse_arguments():
    if len(sys.argv) < 4:
        print("[FEHLER] Falsche Parameteranzahl!")
        print("Nutzung: python main.py <Modell> <Prompt> <Output> [Optionen]")
        sys.exit(1)

    args = {
        'model': sys.argv[1],
        'prompt': sys.argv[2],
        'output': sys.argv[3],
        'device': 'auto',
        'steps': 25,
        'seed': None,
        'nctx': 2048,
        'format': 'auto',
        'sampler': 'euler_a',
        'debug': False,
        'basemodel': None  # basemodel wird hier korrekt initialisiert
    }

    i = 4
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--basemodel':
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                args['basemodel'] = sys.argv[i + 1]
                i += 2
            else:
                args['basemodel'] = True  # Nur das Flag setzen
                i += 1
        elif arg.startswith('--'):
            key = arg.lstrip('--')
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                args[key] = sys.argv[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            print(f"[WARNUNG] Unbekannter Parameter ignoriert: {arg}")
            i += 1

    if args['device'] == 'auto':
        args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def main():
    print_header()
    args = parse_arguments()
    
    print(f"[SYSTEM] Gerät: {args['device'].upper()}")
    print(f"[SYSTEM] Torch-Version: {torch.__version__}")
    print(f"[INFO] Modellname: {args['model']}")
    print(f"[INFO] Prompt: {args['prompt']}")
    print(f"[INFO] Output-Dateiname: {args['output']}")
    
    # Hier wird basemodel explizit in der Ausgabe berücksichtigt
    kwargs_display = ' '.join(f"--{k} {v}" if v is not True else f"--{k}" for k, v in args.items() if k not in ['model', 'prompt', 'output'] and v is not None)
    if args['basemodel']:  # basemodel nur anzeigen, wenn es gesetzt ist
        kwargs_display += f" --basemodel {args['basemodel']}"
    print(f"[INFO] Empfange kwargs: {kwargs_display}")

    modules_dir = Path(__file__).parent / "modules"
    modules = [f.stem for f in modules_dir.glob("*.py") if not f.name.startswith(('_', 'utils'))]

    if args['basemodel']:
        print(f"[INFO] Basemodel: {args['basemodel']}")

    for module_name in modules:
        try:
            module = importlib.import_module(f"modules.{module_name}")
            if hasattr(module, "can_handle") and module.can_handle(args['model']):
                print(f"\n[MODUL] Aktives Modul: {module_name}")
                output_ext = getattr(module, 'DEFAULT_EXTENSION', '.png')
                output_file = f"{args['output']}{output_ext}"
                print(f"[AUSGABE] Datei: {output_file}")
                
                gen_args = {
                    'model_name': args['model'],
                    'prompt': args['prompt'],
                    'output_file': output_file,
                    'device': args['device'],
                    'dtype': torch.float16 if args['device'] == 'cuda' else torch.float32,
                    'user_args': {k: v for k, v in args.items() if k not in ['model', 'prompt', 'output', 'basemodel']}
                }

                # Basemodel zu den gen_args hinzufügen
                if args['basemodel']:
                    gen_args['basemodel'] = args['basemodel']

                module.generate(**gen_args)
                print("\n[ERFOLG] Generierung abgeschlossen")
                sys.exit(0)
                
        except Exception as e:
            print(f"\n[FEHLER] Modul {module_name}: {str(e)}")

    print(f"\n[FEHLER] Kein passendes Modul für: {args['model']}")
    print("[MODULE] Verfügbare Module:")
    for mod in modules:
        print(f" - {mod}")

if __name__ == "__main__":
    main()
