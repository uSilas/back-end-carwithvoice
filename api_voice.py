from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa
import joblib
import io

# ========================
# LOAD MODELO
# ========================
modelo = joblib.load("mlp_comandos_voz.pkl")
label_encoder = joblib.load("label_encoder_mlp.pkl")

app = FastAPI()

# ========================
# CONFIG
# ========================
SR = 16000
DURACAO = 3  # TEM que bater com o treino

# ========================
# FEATURE ENGINEERING (IGUAL AO TREINO)
# ========================
def extract_features(audio, sr=SR):
    audio = librosa.util.fix_length(audio, size=int(SR * DURACAO))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(feat):
        return np.concatenate([
            np.mean(feat, axis=1),
            np.std(feat, axis=1),
            np.max(feat, axis=1),
            np.min(feat, axis=1)
        ])

    mfcc_stats = stats(mfcc)
    delta_stats = stats(delta)
    delta2_stats = stats(delta2)

    energia = np.mean(audio ** 2)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    features = np.concatenate([
        mfcc_stats,
        delta_stats,
        delta2_stats,
        [energia, zcr, spectral_centroid, spectral_bandwidth]
    ])

    return features.reshape(1, -1)

# ========================
# ENDPOINT
# ========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ler áudio
        audio_bytes = await file.read()

        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SR
        )

        # extrair features
        X = extract_features(audio, sr)

        # prever
        pred = modelo.predict(X)[0]
        label = label_encoder.inverse_transform([pred])[0]

        return {
            "command": label
        }

    except Exception as e:
        return {
            "error": str(e)
        }