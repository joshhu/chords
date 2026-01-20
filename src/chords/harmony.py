"""
和聲生成模組

使用 pyrubberband 或 librosa 進行變調，生成和聲
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .analyzer import KeyInfo, get_harmony_intervals
from .pitch import PitchData, frequency_to_midi


# 檢查 rubberband 是否可用
_USE_RUBBERBAND = False
try:
    import pyrubberband as pyrb
    # 測試 rubberband 是否真的可以運作
    _test_audio = np.zeros(1000, dtype=np.float32)
    pyrb.pitch_shift(_test_audio, 44100, 1)
    _USE_RUBBERBAND = True
    print("使用 pyrubberband 進行高品質變調")
except Exception:
    print("pyrubberband 不可用，改用 librosa（品質稍低但仍可接受）")
    import librosa


@dataclass
class HarmonyTrack:
    """和聲音軌"""
    audio: np.ndarray       # 音訊資料
    harmony_type: str       # 和聲類型（"third" 或 "fifth"）
    semitones: int          # 變調半音數


def _pitch_shift_rubberband(audio: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
    """使用 pyrubberband 變調"""
    return pyrb.pitch_shift(
        audio,
        sample_rate,
        semitones,
        rbargs={"--fine": "", "--formant": ""}  # 保留共振峰
    )


def _pitch_shift_librosa(audio: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
    """使用 librosa 變調"""
    return librosa.effects.pitch_shift(
        y=audio,
        sr=sample_rate,
        n_steps=semitones
    )


def _pitch_shift(audio: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
    """選擇可用的變調方法"""
    if _USE_RUBBERBAND:
        return _pitch_shift_rubberband(audio, sample_rate, semitones)
    else:
        return _pitch_shift_librosa(audio, sample_rate, semitones)


def generate_harmony(
    vocals: np.ndarray,
    sample_rate: int,
    semitones: int,
    harmony_type: str = "third"
) -> HarmonyTrack:
    """
    對人聲進行變調生成和聲

    參數:
        vocals: 人聲音訊陣列，形狀為 (channels, samples) 或 (samples,)
        sample_rate: 取樣率
        semitones: 變調的半音數（正數為升調，負數為降調）
        harmony_type: 和聲類型標籤

    回傳:
        HarmonyTrack: 和聲音軌
    """
    print(f"生成 {harmony_type} 和聲（變調 {semitones:+d} 半音）...")

    # 處理立體聲
    if vocals.ndim == 2:
        # 分別處理每個聲道
        harmony_channels = []
        for ch in range(vocals.shape[0]):
            shifted = _pitch_shift(vocals[ch], sample_rate, semitones)
            harmony_channels.append(shifted)
        harmony_audio = np.stack(harmony_channels, axis=0)
    else:
        # 單聲道
        harmony_audio = _pitch_shift(vocals, sample_rate, semitones)

    return HarmonyTrack(
        audio=harmony_audio,
        harmony_type=harmony_type,
        semitones=semitones
    )


def generate_adaptive_harmony(
    vocals: np.ndarray,
    sample_rate: int,
    pitch_data: PitchData,
    key_info: KeyInfo,
    harmony_type: str = "third"
) -> HarmonyTrack:
    """
    根據調性生成自適應和聲

    此方法會根據旋律音符和調性，動態決定每個音符的和聲音程。
    （注意：此方法較為複雜，MVP 版本可先使用固定音程）

    參數:
        vocals: 人聲音訊
        sample_rate: 取樣率
        pitch_data: 音高偵測資料
        key_info: 調性資訊
        harmony_type: 和聲類型（"third" 或 "fifth"）

    回傳:
        HarmonyTrack: 和聲音軌
    """
    # 簡化版：使用最常見的音程
    if harmony_type == "fifth":
        semitones = 7  # 五度永遠是 7 個半音
    else:
        # 三度：根據調性決定預設值
        if key_info.mode == "major":
            semitones = 4  # 大三度
        else:
            semitones = 3  # 小三度

    return generate_harmony(vocals, sample_rate, semitones, harmony_type)


def generate_multi_harmony(
    vocals: np.ndarray,
    sample_rate: int,
    key_info: Optional[KeyInfo] = None,
    harmony_types: List[str] = ["third", "fifth"]
) -> List[HarmonyTrack]:
    """
    生成多聲部和聲

    參數:
        vocals: 人聲音訊
        sample_rate: 取樣率
        key_info: 調性資訊（可選，用於智慧判斷三度類型）
        harmony_types: 要生成的和聲類型列表

    回傳:
        List[HarmonyTrack]: 和聲音軌列表
    """
    harmonies = []

    for harmony_type in harmony_types:
        if harmony_type == "fifth":
            semitones = 7
        elif harmony_type == "third":
            # 根據調性選擇大三度或小三度
            if key_info and key_info.mode == "minor":
                semitones = 3  # 小三度
            else:
                semitones = 4  # 大三度
        elif harmony_type == "third_lower":
            # 低三度
            if key_info and key_info.mode == "minor":
                semitones = -4  # 低大三度
            else:
                semitones = -3  # 低小三度
        elif harmony_type == "fifth_lower":
            semitones = -7  # 低五度
        else:
            print(f"未知的和聲類型: {harmony_type}，跳過")
            continue

        harmony = generate_harmony(vocals, sample_rate, semitones, harmony_type)
        harmonies.append(harmony)

    return harmonies
