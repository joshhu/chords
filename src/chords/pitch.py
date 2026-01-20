"""
音高偵測模組

使用 CREPE 深度學習模型偵測人聲音高
"""

from dataclasses import dataclass
from typing import Optional

import crepe
import numpy as np


@dataclass
class PitchData:
    """音高偵測結果"""
    time: np.ndarray        # 時間序列（秒）
    frequency: np.ndarray   # 頻率序列（Hz）
    confidence: np.ndarray  # 信心度（0-1）
    sample_rate: int


def detect_pitch(
    audio: np.ndarray,
    sample_rate: int,
    model_capacity: str = "medium",
    step_size: int = 10,
    confidence_threshold: float = 0.5
) -> PitchData:
    """
    使用 CREPE 偵測音訊中的音高

    參數:
        audio: 輸入音訊陣列，形狀為 (channels, samples) 或 (samples,)
        sample_rate: 取樣率
        model_capacity: CREPE 模型大小，可選 "tiny", "small", "medium", "large", "full"
        step_size: 音高偵測的時間步長（毫秒）
        confidence_threshold: 信心度閾值，低於此值的音高將被標記為無效

    回傳:
        PitchData: 包含時間、頻率、信心度的資料結構
    """
    print(f"使用 CREPE 模型 ({model_capacity}) 偵測音高...")

    # 確保音訊為單聲道
    if audio.ndim == 2:
        # 取平均轉為單聲道
        audio_mono = np.mean(audio, axis=0)
    else:
        audio_mono = audio

    # 確保音訊為 float32
    audio_mono = audio_mono.astype(np.float32)

    # 使用 CREPE 偵測音高
    time, frequency, confidence, _ = crepe.predict(
        audio_mono,
        sample_rate,
        model_capacity=model_capacity,
        step_size=step_size,
        viterbi=True  # 使用 Viterbi 解碼獲得更平滑的結果
    )

    # 將低信心度的頻率設為 0（表示無效）
    frequency_filtered = frequency.copy()
    frequency_filtered[confidence < confidence_threshold] = 0

    print(f"音高偵測完成！偵測到 {np.sum(confidence >= confidence_threshold)} 個有效音高點")

    return PitchData(
        time=time,
        frequency=frequency_filtered,
        confidence=confidence,
        sample_rate=sample_rate
    )


def frequency_to_midi(frequency: np.ndarray) -> np.ndarray:
    """
    將頻率轉換為 MIDI 音符編號

    參數:
        frequency: 頻率陣列（Hz），0 表示無效

    回傳:
        MIDI 音符編號陣列，無效頻率對應 0
    """
    midi = np.zeros_like(frequency)
    valid = frequency > 0
    midi[valid] = 69 + 12 * np.log2(frequency[valid] / 440.0)
    return midi


def midi_to_frequency(midi: np.ndarray) -> np.ndarray:
    """
    將 MIDI 音符編號轉換為頻率

    參數:
        midi: MIDI 音符編號陣列，0 表示無效

    回傳:
        頻率陣列（Hz），無效音符對應 0
    """
    frequency = np.zeros_like(midi, dtype=np.float32)
    valid = midi > 0
    frequency[valid] = 440.0 * (2 ** ((midi[valid] - 69) / 12))
    return frequency
