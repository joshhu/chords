"""
音源分離模組

使用 Demucs v4 將音訊分離成人聲與伴奏
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


@dataclass
class SeparatedAudio:
    """分離後的音訊資料"""
    vocals: np.ndarray      # 人聲
    accompaniment: np.ndarray  # 伴奏（除人聲外的所有音軌混合）
    sample_rate: int


def get_device() -> torch.device:
    """取得可用的運算裝置（優先使用 GPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def separate_audio(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str = "htdemucs"
) -> SeparatedAudio:
    """
    使用 Demucs 分離音訊中的人聲與伴奏

    參數:
        audio: 輸入音訊陣列，形狀為 (channels, samples) 或 (samples,)
        sample_rate: 取樣率
        model_name: Demucs 模型名稱，預設為 "htdemucs"

    回傳:
        SeparatedAudio: 包含人聲和伴奏的資料結構
    """
    device = get_device()
    print(f"使用裝置: {device}")

    # 載入 Demucs 模型
    print(f"載入 Demucs 模型: {model_name}...")
    model = get_model(model_name)
    model.to(device)

    # 確保音訊格式正確 (channels, samples)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)  # 單聲道轉立體聲
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
        audio = audio.T  # 轉置成 (channels, samples)

    # 如果取樣率不符合模型需求，需要重新取樣
    if sample_rate != model.samplerate:
        import librosa
        audio_resampled = np.array([
            librosa.resample(audio[ch], orig_sr=sample_rate, target_sr=model.samplerate)
            for ch in range(audio.shape[0])
        ])
        audio = audio_resampled
        original_sr = sample_rate
        sample_rate = model.samplerate
    else:
        original_sr = sample_rate

    # 轉換為 PyTorch tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, channels, samples)
    audio_tensor = audio_tensor.to(device)

    # 執行分離
    print("正在分離音源...")
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device)

    # Demucs htdemucs 輸出順序: drums, bass, other, vocals
    # 取得各音軌索引
    source_names = model.sources
    vocals_idx = source_names.index("vocals")

    # 提取人聲
    vocals = sources[0, vocals_idx].cpu().numpy()

    # 伴奏 = 所有非人聲音軌的總和
    accompaniment = np.zeros_like(vocals)
    for i, name in enumerate(source_names):
        if name != "vocals":
            accompaniment += sources[0, i].cpu().numpy()

    # 如果需要，重新取樣回原始取樣率
    if original_sr != sample_rate:
        import librosa
        vocals = np.array([
            librosa.resample(vocals[ch], orig_sr=sample_rate, target_sr=original_sr)
            for ch in range(vocals.shape[0])
        ])
        accompaniment = np.array([
            librosa.resample(accompaniment[ch], orig_sr=sample_rate, target_sr=original_sr)
            for ch in range(accompaniment.shape[0])
        ])
        sample_rate = original_sr

    print("音源分離完成！")

    return SeparatedAudio(
        vocals=vocals,
        accompaniment=accompaniment,
        sample_rate=sample_rate
    )
