import os

# ==============================================================================
# 1. CHEMINS DE FICHIERS ET DOSSIERS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
LEXICON_FILE_PATH = os.path.join(BASE_DIR, "warhammer_lexicon.txt")

# ==============================================================================
# 2. CONFIGURATION LLM (OLLAMA) ET RAG
# ==============================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "cogitator"
OLLAMA_TIMEOUT_SECONDS = 60         # Timeout de la requête HTTP vers Ollama
RAG_NUM_RESULTS = 2                 # Nombre d'entrées du lexique à injecter

# ==============================================================================
# 3. SYNTHÈSE VOCALE (SHERPA-ONNX / PIPER)
# ==============================================================================
TTS_SPEED = 1.1                     # Vitesse de diction (1.0 = normal, >1.0 = plus rapide)
AUDIO_SAMPLE_RATE_DEFAULT = 22050   # Fréquence d'échantillonnage de secours (Hz)
ACTIVE_MODEL_FOLDER = "en_GB-alan-medium"
SELECTED_MODEL_DIR = os.path.join(MODELS_DIR, ACTIVE_MODEL_FOLDER)

# ==============================================================================
# 4. MODULATION SYNTHÉTIQUE (RING MODULATOR)
# ==============================================================================
RING_MOD_FREQ_HZ = 45.0             # Fréquence de la sinusoïde (30-50 Hz = métallique)
RING_MOD_DRY_MIX = 0.85             # Ratio du signal vocal d'origine (85%)
RING_MOD_WET_MIX = 0.15             # Ratio de modulation sinusoïdale (15%)

# ==============================================================================
# 5. TRAITEMENT NUMÉRIQUE DU SIGNAL (PEDALBOARD DSP)
# ==============================================================================
DSP_PITCH_SHIFT_SEMITONES = 3.5     # Modification de hauteur (+3.5 demi-tons = plus aigu)
DSP_BIT_DEPTH = 12.0                # Résolution audio (12-bit = grain numérique propre)
DSP_DISTORTION_DRIVE_DB = 2.0       # Gain de saturation (dB)
DSP_HIGHPASS_CUTOFF_HZ = 500        # Filtre passe-haut (coupe sous 500 Hz)
DSP_LOWPASS_CUTOFF_HZ = 3800        # Filtre passe-bas (coupe au-dessus de 3.8 kHz)

# ==============================================================================
# 6. RECONNAISSANCE VOCALE (STT / WHISPER)
# ==============================================================================
WHISPER_MODEL_SIZE = "small"          # "tiny", "base", "small" (base = bon compromis vitesse/précision)
STT_SAMPLE_RATE = 16000              # Fréquence d'échantillonnage standard pour Whisper (16kHz)
MIC_RECORD_DURATION = 5.0            # Durée d'écoute fixe en secondes (ajustable)
# --- DÉTECTION D'ACTIVITÉ VOCALE (SILERO VAD) ---
VAD_SILENCE_DURATION = 1.5          # Durée de silence requise pour couper (en secondes)
VAD_THRESHOLD = 0.5                 # Seuil de probabilité de parole (0.0 à 1.0)
VAD_CHUNK_SIZE = 512                # Taille de bloc requise par Silero pour 16kHz (512 échantillons = ~32ms)
VAD_MAX_RECORD_TIME = 20.0          # Sécurité : durée maximale d'un enregistrement (en secondes)