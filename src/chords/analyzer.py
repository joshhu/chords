"""
調性分析模組

使用 Librosa 分析音訊的調性（Key）
"""

from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np

# 大調音階的半音間隔模式
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]

# 小調音階的半音間隔模式
MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

# 音符名稱
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class KeyInfo:
    """調性資訊"""
    root: str           # 根音名稱（如 "C", "G#"）
    root_index: int     # 根音索引（0-11，C=0）
    mode: str           # 調式（"major" 或 "minor"）
    confidence: float   # 信心度（0-1）
    scale_notes: List[str]  # 音階中的音符


def analyze_key(
    audio: np.ndarray,
    sample_rate: int
) -> KeyInfo:
    """
    分析音訊的調性

    參數:
        audio: 輸入音訊陣列
        sample_rate: 取樣率

    回傳:
        KeyInfo: 調性資訊
    """
    print("正在分析調性...")

    # 確保音訊為單聲道
    if audio.ndim == 2:
        audio_mono = np.mean(audio, axis=0)
    else:
        audio_mono = audio

    # 計算色度圖（Chromagram）
    chroma = librosa.feature.chroma_cqt(y=audio_mono, sr=sample_rate)

    # 計算每個音高類別的平均能量
    chroma_mean = np.mean(chroma, axis=1)

    # 嘗試所有可能的大調和小調，找出最匹配的
    best_score = -1
    best_key = 0
    best_mode = "major"

    for root in range(12):
        # 計算大調分數
        major_score = _calculate_key_score(chroma_mean, root, MAJOR_SCALE_INTERVALS)
        if major_score > best_score:
            best_score = major_score
            best_key = root
            best_mode = "major"

        # 計算小調分數
        minor_score = _calculate_key_score(chroma_mean, root, MINOR_SCALE_INTERVALS)
        if minor_score > best_score:
            best_score = minor_score
            best_key = root
            best_mode = "minor"

    # 計算信心度（正規化分數）
    confidence = best_score / np.sum(chroma_mean)

    # 取得音階中的音符
    intervals = MAJOR_SCALE_INTERVALS if best_mode == "major" else MINOR_SCALE_INTERVALS
    scale_notes = [NOTE_NAMES[(best_key + interval) % 12] for interval in intervals]

    key_name = NOTE_NAMES[best_key]
    mode_suffix = "m" if best_mode == "minor" else ""
    print(f"偵測到調性: {key_name}{mode_suffix} (信心度: {confidence:.2%})")

    return KeyInfo(
        root=key_name,
        root_index=best_key,
        mode=best_mode,
        confidence=confidence,
        scale_notes=scale_notes
    )


def _calculate_key_score(
    chroma_mean: np.ndarray,
    root: int,
    intervals: List[int]
) -> float:
    """計算特定調性的匹配分數"""
    score = 0.0
    for interval in intervals:
        note_index = (root + interval) % 12
        score += chroma_mean[note_index]
    return score


def get_harmony_intervals(
    key_info: KeyInfo,
    melody_midi: int,
    harmony_type: str = "third"
) -> int:
    """
    根據調性計算和聲音程

    參數:
        key_info: 調性資訊
        melody_midi: 旋律音的 MIDI 編號
        harmony_type: 和聲類型，"third"（三度）或 "fifth"（五度）

    回傳:
        半音數（用於變調）
    """
    if harmony_type == "fifth":
        # 五度永遠是 7 個半音
        return 7

    # 三度需要根據調性判斷是大三度（4半音）還是小三度（3半音）
    # 計算旋律音在音階中的位置
    melody_note = melody_midi % 12
    root = key_info.root_index

    # 計算相對於根音的半音距離
    relative_semitones = (melody_note - root) % 12

    # 根據調性判斷三度類型
    if key_info.mode == "major":
        # 大調：1、4、5 級是大三度，其他是小三度
        major_third_positions = [0, 5, 7]  # I, IV, V
        if relative_semitones in major_third_positions:
            return 4  # 大三度
        else:
            return 3  # 小三度
    else:
        # 小調：3、6、7 級是大三度，其他是小三度
        major_third_positions = [3, 8, 10]  # III, VI, VII
        if relative_semitones in major_third_positions:
            return 4  # 大三度
        else:
            return 3  # 小三度
