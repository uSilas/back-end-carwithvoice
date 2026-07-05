"""
feature_extraction.py — Pipeline de features para inferência em tempo real.

Deve ser IDÊNTICO ao script de treinamento (2_extracao_features_final.py),
com as mesmas opções usadas no treino do MLP final:
  - N_MFCC = 13
  - Chroma REMOVIDO

Vetor final: MFCC+delta+delta2 (156) + espectral (4) + onset (4) + LSTM (64) = 228
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct
from scipy.signal import savgol_filter

# ========================
# CONFIG — mesmos valores do treinamento
# ========================
SR         = 16000
DURACAO    = 2.50
N_MFCC     = 13        # ← 13, igual ao treino da LSTM e do MLP
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
FMIN       = 0.0
FMAX       = SR / 2


# ========================
# MODELO LSTM
# ========================

class LSTMSupervisionado(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout    = nn.Dropout(0.0)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        """Hidden state final da LSTM — usado como feature pelo MLP."""
        _, (hn, _) = self.lstm(x)
        return hn[-1]   # (batch, hidden_size)

    def forward(self, x):
        return self.classifier(self.dropout(self.encode(x)))


# ========================
# PRÉ-CÔMPUTO DE FILTROS (feito uma única vez ao importar o módulo)
# ========================

def _hz_para_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_para_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _construir_filtros_mel(sr, n_fft, n_mels, fmin, fmax):
    mel_min = _hz_para_mel(fmin)
    mel_max = _hz_para_mel(fmax)
    pontos_mel = np.linspace(mel_min, mel_max, n_mels + 2)
    pontos_hz  = _mel_para_hz(pontos_mel)
    bins = np.floor((n_fft + 1) * pontos_hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filtros = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_esq, f_centro, f_dir = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_esq, f_centro):
            if f_centro != f_esq:
                filtros[m - 1, k] = (k - f_esq) / (f_centro - f_esq)
        for k in range(f_centro, f_dir):
            if f_dir != f_centro:
                filtros[m - 1, k] = (f_dir - k) / (f_dir - f_centro)
    return filtros


# Filtros compartilhados — calculados uma vez na importação
_FILTROS_MEL    = _construir_filtros_mel(SR, N_FFT, N_MELS, FMIN, FMAX)
_JANELA_HAMMING = np.hamming(N_FFT)


# ========================
# ÁUDIO — normalização e ajuste de duração
# ========================

def normalizar_amplitude(audio: np.ndarray) -> np.ndarray:
    pico = np.max(np.abs(audio))
    return audio / pico if pico > 0 else audio


def ajustar_duracao(audio: np.ndarray, tamanho_fixo: int) -> np.ndarray:
    n = len(audio)
    if n > tamanho_fixo:
        return audio[:tamanho_fixo]
    if n < tamanho_fixo:
        return np.pad(audio, (0, tamanho_fixo - n), mode="constant")
    return audio


# ========================
# STFT / ESPECTRO DE POTÊNCIA
# ========================

def _espectro_potencia(audio: np.ndarray):
    """Retorna potencia (n_frames, n_fft//2+1)."""
    n_amostras = len(audio)
    n_frames   = 1 + (n_amostras - N_FFT) // HOP_LENGTH
    if n_frames < 1:
        n_frames = 1
        audio = np.pad(audio, (0, N_FFT - n_amostras), mode="constant")

    frames = np.zeros((n_frames, N_FFT))
    for i in range(n_frames):
        trecho = audio[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT]
        if len(trecho) < N_FFT:
            trecho = np.pad(trecho, (0, N_FFT - len(trecho)), mode="constant")
        frames[i] = trecho

    frames_jan = frames * _JANELA_HAMMING
    espectro   = np.fft.rfft(frames_jan, n=N_FFT, axis=1)
    potencia   = (np.abs(espectro) ** 2) / N_FFT
    return potencia, espectro


# ========================
# MFCC
# ========================

def _calcular_mfcc(audio: np.ndarray) -> np.ndarray:
    """Retorna (N_MFCC, n_frames)."""
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    potencia, _ = _espectro_potencia(audio_pre)

    esp_mel = np.maximum(potencia @ _FILTROS_MEL.T, 1e-10)
    log_mel = np.log(esp_mel)
    mfcc    = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    return mfcc.T   # (N_MFCC, n_frames)


# ========================
# DELTA (Savitzky-Golay)
# ========================

def _calcular_delta(feat: np.ndarray, ordem: int = 1) -> np.ndarray:
    """feat: (n_coef, n_frames)"""
    n_frames = feat.shape[1]
    largura  = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if largura < 3:
        return np.zeros_like(feat)
    return savgol_filter(feat, window_length=largura, polyorder=2,
                         deriv=ordem, axis=1, mode="interp")


# ========================
# FEATURES ESPECTRAIS
# ========================

def _calcular_zcr(audio: np.ndarray) -> float:
    n_amostras = len(audio)
    n_frames   = 1 + (n_amostras - N_FFT) // HOP_LENGTH
    if n_frames < 1:
        return 0.0

    frames = np.zeros((n_frames, N_FFT))
    for i in range(n_frames):
        trecho = audio[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT]
        if len(trecho) < N_FFT:
            trecho = np.pad(trecho, (0, N_FFT - len(trecho)), mode="constant")
        frames[i] = trecho

    sinais = np.sign(frames)
    sinais[sinais == 0] = 1
    cruzamentos = np.abs(np.diff(sinais, axis=1))
    zcr = np.sum(cruzamentos, axis=1) / (2 * N_FFT)
    return float(np.mean(zcr))


def _calcular_centroid_bw(audio: np.ndarray):
    potencia, _ = _espectro_potencia(audio)
    magnitude   = np.sqrt(potencia)
    freqs       = np.fft.rfftfreq(N_FFT, d=1.0 / SR)

    soma = np.sum(magnitude, axis=1) + 1e-10
    centroid  = np.sum(magnitude * freqs[np.newaxis, :], axis=1) / soma
    desvio    = freqs[np.newaxis, :] - centroid[:, np.newaxis]
    bandwidth = np.sqrt(np.sum(magnitude * desvio ** 2, axis=1) / soma)
    return float(np.mean(centroid)), float(np.mean(bandwidth))


# ========================
# ONSET STRENGTH + TEMPO
# ========================

def _calcular_onset_strength(audio: np.ndarray) -> np.ndarray:
    audio_pre   = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    potencia, _ = _espectro_potencia(audio_pre)

    esp_mel = np.maximum(potencia @ _FILTROS_MEL.T, 1e-10)
    log_mel = np.log(esp_mel)

    diff = np.maximum(np.diff(log_mel, axis=0), 0.0)
    onset_env = np.concatenate([[0.0], np.sum(diff, axis=1)])
    return onset_env


def _calcular_tempo(onset_env: np.ndarray) -> float:
    if len(onset_env) < 2 or np.all(onset_env == 0):
        return 0.0

    centrado = onset_env - onset_env.mean()
    autocorr = np.correlate(centrado, centrado, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]

    fps     = SR / HOP_LENGTH
    lag_min = int(fps * 60 / 300)
    lag_max = min(int(fps * 60 / 30), len(autocorr) - 1)

    if lag_max <= lag_min:
        return 0.0

    melhor_lag = lag_min + np.argmax(autocorr[lag_min : lag_max + 1])
    return float(60.0 * fps / melhor_lag) if melhor_lag > 0 else 0.0


# ========================
# ESTATÍSTICAS (mean, std, max, min)
# ========================

def _stats(feat: np.ndarray) -> np.ndarray:
    """feat: (n_coef, n_frames) → vetor 1-D de tamanho 4*n_coef."""
    return np.concatenate([
        np.mean(feat, axis=1),
        np.std(feat,  axis=1),
        np.max(feat,  axis=1),
        np.min(feat,  axis=1),
    ])


# ========================
# EXTRAÇÃO LSTM
# ========================

def _extrair_lstm(mfcc: np.ndarray, encoder, norm_mean, norm_std) -> np.ndarray:
    """
    mfcc: (N_MFCC, n_frames)
    Retorna o hidden state final como numpy (hidden_size,).
    """
    seq = torch.tensor(mfcc.T, dtype=torch.float32)                   # (n_frames, N_MFCC)
    seq = (seq - norm_mean.squeeze(0)) / norm_std.squeeze(0)          # Z-score
    seq = seq.unsqueeze(0)                                             # (1, n_frames, N_MFCC)
    with torch.no_grad():
        z = encoder.encode(seq)
    return z.squeeze(0).numpy()                                        # (hidden_size,)


# ========================
# FUNÇÃO PÚBLICA — chamada pela API / captura em tempo real
# ========================

def extrair_features(
    audio: np.ndarray,
    encoder: LSTMSupervisionado,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> np.ndarray:
    """
    Recebe áudio float32 mono 16 kHz (qualquer duração) e retorna
    o vetor de features (1, 228) pronto para o MLP Pipeline.

    Pipeline (ordem idêntica ao treinamento):
        MFCC+delta+delta2  → 156   (13 coef × 4 stats × 3)
        Espectral          →   4   (energia, zcr, centroid, bandwidth)
        Onset              →   4   (mean, std, max, tempo)
        LSTM               →  64
        Total              → 228

    NOTA: chroma não é incluído — removido no treino do MLP final.
    NOTA: o MLP pkl já contém o StandardScaler interno (Pipeline sklearn),
          portanto NÃO é necessário escalar o vetor externamente.
    """
    # --- pré-processamento ---
    audio = normalizar_amplitude(audio)
    audio = ajustar_duracao(audio, int(SR * DURACAO))

    # --- MFCC + deltas ---
    mfcc   = _calcular_mfcc(audio)                    # (13, n_frames)
    delta  = _calcular_delta(mfcc, ordem=1)
    delta2 = _calcular_delta(mfcc, ordem=2)
    feat_mfcc = np.concatenate([_stats(mfcc), _stats(delta), _stats(delta2)])  # 156

    # --- features espectrais ---
    energia  = float(np.mean(audio ** 2))
    zcr      = _calcular_zcr(audio)
    centroid, bw = _calcular_centroid_bw(audio)
    feat_espectral = np.array([energia, zcr, centroid, bw])                    # 4

    # --- onset + tempo ---
    onset_env  = _calcular_onset_strength(audio)
    tempo      = _calcular_tempo(onset_env)
    feat_onset = np.array([
        float(np.mean(onset_env)),
        float(np.std(onset_env)),
        float(np.max(onset_env)),
        tempo,
    ])                                                                          # 4

    # --- LSTM hidden state ---
    feat_lstm = _extrair_lstm(mfcc, encoder, norm_mean, norm_std)              # 64

    # --- concatena na mesma ordem do treino (sem chroma) ---
    feat_final = np.concatenate([
        feat_mfcc,       # 156
        feat_espectral,  # 4
        feat_onset,      # 4
        feat_lstm,       # 64
    ])

    return feat_final.reshape(1, -1)   # (1, 228) — formato esperado pelo MLP Pipeline