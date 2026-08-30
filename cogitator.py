import os
import time
import wave
import tempfile
import requests
import torch
import numpy as np
import sounddevice as sd
from pedalboard import (
    Pedalboard, 
    PitchShift, 
    Distortion, 
    HighpassFilter, 
    LowpassFilter, 
    Bitcrush,
    GSMFullRateCompressor
)

# Imports des modules locaux
import config as cfg
from rag_engine import LexiconRAG

# ==============================================================================
# 1. INITIALISATION DES MOTEURS DU COGITATEUR
# ==============================================================================

def init_rag():
    """Initialise la base de données vectorielle RAG."""
    print("[INIT] Initializing Sanctified RAG Engine...")
    try:
        return LexiconRAG()
    except Exception as e:
        print(f"[WARNING] Could not initialize RAG Engine: {e}")
        return None

def init_vad():
    """Charge le modèle de détection d'activité vocale Silero VAD."""
    print("[INIT] Loading Silero VAD Model...")
    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        print("[INIT] Silero VAD loaded successfully.")
        return vad_model, vad_utils
    except Exception as e:
        print(f"[WARNING] Could not load Silero VAD: {e}")
        return None, None

def init_stt():
    """Charge le modèle Faster-Whisper pour la reconnaissance vocale."""
    print("[INIT] Loading Faster-Whisper STT Model...")
    try:
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(cfg.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print(f"[INIT] Faster-Whisper '{cfg.WHISPER_MODEL_SIZE}' loaded successfully.")
        return stt_model
    except Exception as e:
        print(f"[WARNING] Could not load Faster-Whisper: {e}")
        return None

def init_tts():
    """Charge le modèle Sherpa-ONNX pour la synthèse vocale."""
    print("[INIT] Loading Sherpa-ONNX Voice Model...")
    try:
        import sherpa_onnx
        
        if not os.path.exists(cfg.SELECTED_MODEL_DIR):
            raise FileNotFoundError(f"Model directory '{cfg.SELECTED_MODEL_DIR}' does not exist.")

        model_files = [f for f in os.listdir(cfg.SELECTED_MODEL_DIR) if f.endswith(".onnx")]
        tokens_files = [f for f in os.listdir(cfg.SELECTED_MODEL_DIR) if f == "tokens.txt" or f.endswith(".txt")]
        lexicon_files = [f for f in os.listdir(cfg.SELECTED_MODEL_DIR) if f == "lexicon.txt"]
        
        data_dir_path = os.path.join(cfg.SELECTED_MODEL_DIR, "espeak-ng-data")
        if not os.path.exists(data_dir_path):
            data_dir_path = ""

        if not model_files or not tokens_files:
            raise FileNotFoundError(f"Missing .onnx or tokens.txt in '{cfg.SELECTED_MODEL_DIR}'")
            
        model_path = os.path.join(cfg.SELECTED_MODEL_DIR, model_files[0])
        tokens_path = os.path.join(cfg.SELECTED_MODEL_DIR, tokens_files[0])
        lexicon_path = os.path.join(cfg.SELECTED_MODEL_DIR, lexicon_files[0]) if lexicon_files else ""

        print(f"[INIT] Model: {model_files[0]}")
        print(f"[INIT] Lexicon: {lexicon_files[0] if lexicon_files else 'None'}")
        print(f"[INIT] Data Dir: {data_dir_path if data_dir_path else 'None'}")

        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model_path,
            tokens=tokens_path,
            lexicon=lexicon_path,
            data_dir=data_dir_path
        )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits_config)
        )
        
        tts = sherpa_onnx.OfflineTts(tts_config)
        print("[INIT] Sherpa-ONNX TTS loaded successfully.")
        return tts
    except Exception as e:
        print(f"[WARNING] Could not load Sherpa-ONNX TTS: {e}")
        return None

def init_dsp_effects():
    """Configure la chaîne d'effets audio Pedalboard."""
    return Pedalboard([
        PitchShift(semitones=cfg.DSP_PITCH_SHIFT_SEMITONES),
        Bitcrush(bit_depth=cfg.DSP_BIT_DEPTH),
        GSMFullRateCompressor(),
        Distortion(drive_db=cfg.DSP_DISTORTION_DRIVE_DB),
        HighpassFilter(cutoff_frequency_hz=cfg.DSP_HIGHPASS_CUTOFF_HZ),
        LowpassFilter(cutoff_frequency_hz=cfg.DSP_LOWPASS_CUTOFF_HZ)
    ])


# ==============================================================================
# 2. CHARGEMENT GLOBAL DES INSTANCES
# ==============================================================================
rag = init_rag()
vad_model, vad_utils = init_vad()
get_speech_timestamps = vad_utils[0] if vad_utils else None
stt_model = init_stt()
tts = init_tts()
servitor_effects = init_dsp_effects()


# ==============================================================================
# 3. FONCTIONS DU PIPELINE AUDIO & COGNITIF
# ==============================================================================

def record_audio_with_vad(sample_rate=cfg.STT_SAMPLE_RATE):
    """Enregistre l'audio du microphone en continu et s'arrête dès qu'un silence est détecté."""
    if not vad_model:
        print(f"\n[MIC] Listening (Fallback mode, {cfg.MIC_RECORD_DURATION}s)...")
        audio_data = sd.rec(int(cfg.MIC_RECORD_DURATION * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return temp_file.name

    print("\n[MIC] Listening...")
    
    audio_buffer = []
    speech_started = False
    silence_start_time = None
    start_time = time.time()
    chunk_samples = cfg.VAD_CHUNK_SIZE
    
    vad_model.reset_states()

    def callback(indata, frames, time_info, status):
        nonlocal speech_started, silence_start_time, audio_buffer
        if status:
            print(f"[MIC WARN] Audio status: {status}")

        audio_chunk = indata.copy().flatten()
        audio_tensor = torch.from_numpy(audio_chunk).float() / 32768.0
        speech_prob = vad_model(audio_tensor, sample_rate).item()

        if speech_prob > cfg.VAD_THRESHOLD:
            if not speech_started:
                print("[MIC] Voice detected. Recording...")
                speech_started = True
            silence_start_time = None
        else:
            if speech_started and silence_start_time is None:
                silence_start_time = time.time()

        if speech_started:
            audio_buffer.append(audio_chunk)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', 
                         blocksize=chunk_samples, callback=callback):
        while True:
            sd.sleep(50)
            
            if speech_started and silence_start_time:
                elapsed_silence = time.time() - silence_start_time
                if elapsed_silence >= cfg.VAD_SILENCE_DURATION:
                    print(f"[MIC] Silence detected ({elapsed_silence:.1f}s). Stopping recording.")
                    break

            if time.time() - start_time > cfg.VAD_MAX_RECORD_TIME:
                print("[MIC] Max recording duration reached.")
                break

    if not audio_buffer:
        return None

    recorded_np = np.concatenate(audio_buffer, axis=0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recorded_np.tobytes())

    return temp_file.name

def transcribe_audio(audio_path):
    """Transcrit l'audio WAV en texte via Faster-Whisper avec guidage phonétique."""
    if not stt_model:
        return ""
    
    segments, _ = stt_model.transcribe(
        audio_path, 
        beam_size=5,
        initial_prompt=getattr(cfg, "STT_INITIAL_PROMPT", "")
    )
    transcription = " ".join([segment.text for segment in segments]).strip()
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    return transcription

def speak_servitor(text):
    """Synthétise la voix du Serviteur et applique le traitement DSP."""
    if not tts:
        print(f"[AUDIO MOCK] Servitor says: {text}")
        return

    print(f"[DEBUG] Generating speech audio with Sherpa-ONNX...")
    try:
        audio = tts.generate(text, sid=0, speed=cfg.TTS_SPEED)
        raw_samples = np.array(audio.samples, dtype=np.float32)
        sr = getattr(audio, 'sample_rate', cfg.AUDIO_SAMPLE_RATE_DEFAULT)

        # Ring Modulator
        t = np.linspace(0, len(raw_samples) / sr, len(raw_samples), endpoint=False)
        carrier = np.sin(2 * np.pi * cfg.RING_MOD_FREQ_HZ * t)
        robotic_samples = raw_samples * (cfg.RING_MOD_DRY_MIX + cfg.RING_MOD_WET_MIX * carrier)

        # Effets Pedalboard
        print("[DEBUG] Applying Pedalboard audio effects...")
        processed_audio = servitor_effects(robotic_samples, sr)

        # Restitution sonore
        print("[DEBUG] Playing audio output...")
        sd.play(processed_audio, sr)
        sd.wait()
        print("[DEBUG] Playback complete.")
    except Exception as e:
        print(f"[ERROR] TTS / Audio playback failed: {e}")

def query_llm(user_input):
    """Interroge Ollama avec le contexte extrait de la base RAG."""
    context = rag.search(user_input, n_results=cfg.RAG_NUM_RESULTS)
    
    if context:
        print(f"[RAG] Context Injected:\n{context}")
        full_prompt = f"Sanctified Lexicon Archives:\n{context}\n\nUser Query: {user_input}\nRespond as a Servitor:"
    else:
        full_prompt = user_input

    payload = {
        "model": cfg.MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(
            cfg.OLLAMA_URL, 
            json=payload, 
            timeout=cfg.OLLAMA_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return "Cognitive link disrupted. Machine spirit unresponsive."

# ==============================================================================
# 4. BOUCLE PRINCIPALE (VOICE LOOP)
# ==============================================================================
def main():
    print("\n=======================================================")
    print("   COGITATOR / SERVITOR AUDIO INTERFACE INITIALIZED")
    print("   Praise the Omnissiah. Press Ctrl+C to terminate.")
    print("=======================================================\n")
    
    speak_servitor("Cogitator online. State designated query.")

    # Liste des variations phonétiques tolérées pour l'arrêt
    shutdown_phrases = ["exit", "terminate", "quit", "stop", "shut down"]

    while True:
        try:
            audio_file = record_audio_with_vad()
            
            if not audio_file:
                continue

            user_input = transcribe_audio(audio_file)

            if not user_input:
                print("[WARN] No speech detected or transcription empty.")
                continue

            print(f"\n[USER (STT)] > {user_input}")

            # Interception d'arrêt tolérante aux erreurs phonétiques
            if any(phrase in user_input.lower() for phrase in shutdown_phrases):
                speak_servitor("Terminating cognitive session.")
                break

            response_text = query_llm(user_input)
            print(f"\n[SERVITOR] > {response_text}\n")

            speak_servitor(response_text)

        except KeyboardInterrupt:
            print("\n[INIT] Session aborted by user.")
            break

if __name__ == "__main__":
    main()