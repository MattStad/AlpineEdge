#!/usr/bin/env python3
"""
Debug-Script: Prüft Ollama-Status und verfügbare Modelle
"""

import sys

print("=" * 70)
print("OLLAMA DEBUG - STATUS CHECK")
print("=" * 70)

# 1. Prüfe ob ollama package installiert ist
print("\n1. Prüfe ollama Python-Package...")
try:
    import ollama

    print("   ✓ ollama package installiert")
except ImportError:
    print("   ✗ ollama package NICHT installiert!")
    print("   Installation: pip install ollama")
    sys.exit(1)

# 2. Prüfe Ollama Server
print("\n2. Prüfe Ollama Server-Verbindung...")
try:
    response = ollama.list()
    print("   ✓ Ollama Server läuft")
    print(f"\n   Raw Response: {response}")
except Exception as e:
    print(f"   ✗ Ollama Server nicht erreichbar!")
    print(f"   Fehler: {e}")
    print("\n   Lösung:")
    print("   - Starte Server: ollama serve")
    print("   - Oder in neuem Terminal: ollama list")
    sys.exit(1)

# 3. Parse verfügbare Modelle
print("\n3. Parse verfügbare Modelle...")
models = response.get('models', [])

if not models:
    print("   ✗ KEINE Modelle installiert!")
    print("\n   Installiere ein Modell:")
    print("   ollama pull llama3.2:3b")
    print("   ollama pull llama3.1:8b")
    print("   ollama pull mistral:7b")
else:
    print(f"   ✓ {len(models)} Modell(e) gefunden\n")

    for i, model in enumerate(models, 1):
        # Neue ollama API verwendet 'model' Attribut statt 'name'
        name = None
        if hasattr(model, 'model'):
            name = model.model
        elif hasattr(model, 'name'):
            name = model.name
        elif isinstance(model, dict):
            name = model.get('model') or model.get('name', 'N/A')
        else:
            name = 'N/A'

        size = getattr(model, 'size', None) or (model.get('size', 0) if isinstance(model, dict) else 0)
        size_gb = size / (1024 ** 3) if size else 0

        details = getattr(model, 'details', None) or (model.get('details', {}) if isinstance(model, dict) else {})

        print(f"   [{i}] {name}")
        print(f"       Größe: {size_gb:.2f} GB")
        if details:
            print(f"       Details: {details}")
        print()

# 4. Teste Auto-Detection
print("\n4. Teste Auto-Detection...")
try:
    from news_classifier import NewsClassifier

    print("   Versuche Classifier zu initialisieren...")
    classifier = NewsClassifier()

    print(f"   ✓ Auto-Detection erfolgreich!")
    print(f"   Gewähltes Modell: {classifier.model_name}")

except Exception as e:
    print(f"   ✗ Auto-Detection fehlgeschlagen!")
    print(f"   Fehler: {e}")
    import traceback

    traceback.print_exc()

# 5. Empfehlungen
print("\n" + "=" * 70)
print("EMPFEHLUNGEN")
print("=" * 70)

if not models:
    print("\n⚠️  KEINE MODELLE INSTALLIERT")
    print("\nInstalliere jetzt:")
    print("  ollama pull llama3.2:3b    # Schnell, 2GB")
    print("  ollama pull llama3.1:8b    # Präziser, 4.7GB")
    print("  ollama pull mistral:7b     # Finanz-News, 4.1GB")
else:
    print("\n✓ Setup scheint OK")
    print("\nVerfügbare Modelle:")
    for model in models:
        # Extrahiere Namen
        name = None
        if hasattr(model, 'model'):
            name = model.model
        elif hasattr(model, 'name'):
            name = model.name
        elif isinstance(model, dict):
            name = model.get('model') or model.get('name')

        if name:
            print(f"  • {name}")

    # Erstes Modell für Beispiel
    first_model = None
    if models:
        m = models[0]
        if hasattr(m, 'model'):
            first_model = m.model
        elif hasattr(m, 'name'):
            first_model = m.name
        elif isinstance(m, dict):
            first_model = m.get('model') or m.get('name')

    print("\nVerwendung:")
    print("  from news_classifier import classify_unclassified_news")
    print("  classify_unclassified_news()  # Auto-detect")
    if first_model:
        print(f"  # oder explizit: model_name='{first_model}'")

print("\n" + "=" * 70)
