"""
API de classificação de comandos de voz — SEM LIBROSA.

Pipeline: MFCC+delta+delta2 (156) + espectral (4) + onset (4) + LSTM (64) = 228 features
Modelo: MLP Pipeline sklearn (modelo_mlp_sem_chroma.pkl)
        — já inclui StandardScaler interno, não escalar externamente.

Execute: uvicorn api_voice:app --reload
"""

from fastapi import FastAPI, UploadFile, File
import numpy as np
import torch
import joblib
import io
from scipy.io import wavfile

from feature_extraction import (
    extrair_features, LSTMSupervisionado,
    SR, N_MFCC,
)

# ========================
# LOAD MODELOS
# ========================

modelo        = joblib.load("modelo_mlp_sem_chroma.pkl")       # MLP Pipeline (scaler embutido)
label_encoder = joblib.load("label_encoder_mlp_sem_chroma.pkl")

ckpt        = torch.load("lstm_supervisionada_best.pt", map_location="cpu", weights_only=False)
HIDDEN_SIZE = ckpt["hidden_size"]
NUM_LAYERS  = ckpt["num_layers"]
num_classes = len(ckpt["classes"])
norm_mean   = ckpt["mean"]
norm_std    = ckpt["std"]

encoder = LSTMSupervisionado(N_MFCC, HIDDEN_SIZE, NUM_LAYERS, num_classes)
encoder.load_state_dict(ckpt["model_state"])
encoder.eval()

print(f"Modelo MLP carregado | classes: {list(label_encoder.classes_)}")
print(f"Encoder LSTM carregado → {HIDDEN_SIZE} dims\n")


app = FastAPI()


# ========================
# CARREGAMENTO DE ÁUDIO (sem librosa)
# Espera WAV — mono ou estéreo, qualquer sample rate, mas o
# script de captura já envia 16kHz mono, então normalmente
# não há reamostragem aqui.
# ========================

def carregar_audio_bytes(audio_bytes, sr_destino=SR):
    sr_original, audio = wavfile.read(io.BytesIO(audio_bytes))

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr_original != sr_destino:
        duracao = len(audio) / sr_original
        n_amostras_novo = int(round(duracao * sr_destino))
        x_antigo = np.linspace(0, duracao, num=len(audio), endpoint=False)
        x_novo   = np.linspace(0, duracao, num=n_amostras_novo, endpoint=False)
        audio    = np.interp(x_novo, x_antigo, audio).astype(np.float32)

    return audio


# ========================
# ENDPOINT
# ========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio = carregar_audio_bytes(audio_bytes, sr_destino=SR)

        # (1, 228) — scaler já aplicado internamente pelo Pipeline
        X = extrair_features(audio, encoder, norm_mean, norm_std)

        pred  = modelo.predict(X)[0]
        label = label_encoder.inverse_transform([pred])[0]

        # Probabilidades por classe (MLPClassifier tem predict_proba nativo)
        proba     = modelo.predict_proba(X)[0]
        confianca = float(np.max(proba))

        if confianca <= 0.60:
            return {
                "command":    "desconhecido",
                "confidence": confianca,
            }
        else:
            return {
                "command":    str(label),
                "confidence": confianca,
            }

    except Exception as e:
        return {"error": str(e)}
