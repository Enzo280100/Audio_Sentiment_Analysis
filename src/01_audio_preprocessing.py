import os
import re
from tqdm import tqdm

import numpy as np
from pydub import AudioSegment

import librosa
import soundfile as sf
import noisereduce as nr

import torch

# ======================================================
# ================ LOAD SILERO VAD =====================
# ======================================================

def load_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils
    return model, get_speech_timestamps, read_audio


vad_model, get_speech_timestamps, read_audio = load_vad_model()


# ======================================================
# ================ AUDIO CLEANING =======================
# ======================================================

def clean_audio(input_path, target_sr=16000):
    """
    Loads audio, resamples, converts to mono, reduces noise, and normalizes.
    Returns numpy array + sample rate.
    """
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Noise reduction
    reduced = nr.reduce_noise(y=audio, sr=target_sr)

    # Normalize amplitude
    normalized = librosa.util.normalize(reduced)

    return normalized, target_sr


# ======================================================
# ================ SAVE AUDIO ===========================
# ======================================================

def save_audio(output_path, audio, sr):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, sr)


# ======================================================
# ================ VAD SEGMENTATION =====================
# ======================================================

def extract_voice_segments(input_wav_path, output_dir, filename):
    """
    Extracts speech-only segments using Silero VAD.
    Saves each segment as a WAV file.
    """
    wav = read_audio(input_wav_path)
    speech_timestamps = get_speech_timestamps(wav, vad_model)

    if len(speech_timestamps) == 0:
        return 0

    audio = AudioSegment.from_wav(input_wav_path)

    os.makedirs(output_dir, exist_ok=True)

    for i, ts in enumerate(speech_timestamps):
        # Convert samples → ms (16kHz)
        start_ms = ts["start"] * 1000 / 16000
        end_ms = ts["end"] * 1000 / 16000

        segment = audio[start_ms:end_ms]
        segment.export(
            os.path.join(output_dir, f"{filename}_seg_{i+1}.wav"),
            format="wav"
        )

    return len(speech_timestamps)


# ======================================================
# ================ MAIN PROCESSING ======================
# ======================================================

def preprocess_audio_folder(input_dir, output_clean_dir, output_segments_dir):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    for filename in tqdm(files, desc="Procesando audios"):

        input_path = os.path.join(input_dir, filename)
        clean_path = os.path.join(output_clean_dir, filename)

        try:
            print(f"🔧 Limpieza básica: {filename}")

            # 1. Clean audio (resample + mono + denoise + normalize)
            cleaned_audio, sr = clean_audio(input_path)
            save_audio(clean_path, cleaned_audio, sr)

            # 2. VAD segmentation (optional)
            print(f"🎙️ Segmentación por voz: {filename}")
            num_segments = extract_voice_segments(
                clean_path,
                output_segments_dir,
                filename.replace(".wav", "")
            )

            print(f"   → {num_segments} segmentos extraídos.")

        except Exception as e:
            print(f"⚠️ Error procesando {filename}: {e}")
            continue
