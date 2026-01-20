"""
混音模組

合併人聲、和聲與伴奏
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pedalboard import Pedalboard, Reverb, Compressor, Gain
from pedalboard.io import AudioFile

from .harmony import HarmonyTrack


@dataclass
class MixSettings:
    """混音設定"""
    vocals_volume: float = 1.0      # 人聲音量
    harmony_volume: float = 0.6     # 和聲音量
    accompaniment_volume: float = 1.0  # 伴奏音量
    add_reverb: bool = True         # 是否為和聲加入殘響
    reverb_room_size: float = 0.3   # 殘響空間大小
    reverb_wet_level: float = 0.2   # 殘響濕度


def mix_audio(
    vocals: np.ndarray,
    harmonies: List[HarmonyTrack],
    accompaniment: np.ndarray,
    sample_rate: int,
    settings: Optional[MixSettings] = None
) -> np.ndarray:
    """
    混合所有音軌

    參數:
        vocals: 原始人聲
        harmonies: 和聲音軌列表
        accompaniment: 伴奏
        sample_rate: 取樣率
        settings: 混音設定

    回傳:
        混合後的音訊陣列
    """
    if settings is None:
        settings = MixSettings()

    print("正在混音...")

    # 確保所有音訊長度一致
    min_length = min(
        vocals.shape[-1],
        accompaniment.shape[-1],
        *[h.audio.shape[-1] for h in harmonies]
    )

    # 裁剪到相同長度
    vocals = vocals[..., :min_length]
    accompaniment = accompaniment[..., :min_length]

    # 調整人聲音量
    mixed = vocals * settings.vocals_volume

    # 處理和聲
    if harmonies:
        # 合併所有和聲
        harmony_mix = np.zeros_like(mixed)
        for harmony in harmonies:
            h_audio = harmony.audio[..., :min_length]
            harmony_mix += h_audio

        # 對和聲施加效果
        if settings.add_reverb:
            harmony_mix = _apply_harmony_effects(
                harmony_mix,
                sample_rate,
                settings
            )

        # 加入和聲
        mixed += harmony_mix * settings.harmony_volume

    # 加入伴奏
    mixed += accompaniment * settings.accompaniment_volume

    # 正規化以避免削波
    max_amplitude = np.max(np.abs(mixed))
    if max_amplitude > 1.0:
        mixed = mixed / max_amplitude * 0.95
        print(f"已正規化音訊（原始峰值: {max_amplitude:.2f}）")

    print("混音完成！")
    return mixed


def _apply_harmony_effects(
    harmony: np.ndarray,
    sample_rate: int,
    settings: MixSettings
) -> np.ndarray:
    """
    對和聲施加音效處理

    參數:
        harmony: 和聲音訊
        sample_rate: 取樣率
        settings: 混音設定

    回傳:
        處理後的和聲音訊
    """
    # 建立效果鏈
    board = Pedalboard([
        # 輕微壓縮讓和聲更穩定
        Compressor(
            threshold_db=-20,
            ratio=3,
            attack_ms=5,
            release_ms=100
        ),
        # 殘響讓和聲更自然融合
        Reverb(
            room_size=settings.reverb_room_size,
            wet_level=settings.reverb_wet_level,
            dry_level=1.0 - settings.reverb_wet_level
        )
    ])

    # Pedalboard 需要 (samples, channels) 格式
    if harmony.ndim == 2:
        harmony_transposed = harmony.T
    else:
        harmony_transposed = harmony.reshape(-1, 1)

    # 施加效果
    processed = board(harmony_transposed, sample_rate)

    # 轉回原始格式
    if harmony.ndim == 2:
        return processed.T
    else:
        return processed.flatten()


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int
) -> None:
    """
    儲存音訊到檔案

    參數:
        audio: 音訊陣列
        output_path: 輸出檔案路徑
        sample_rate: 取樣率
    """
    print(f"正在儲存音訊到: {output_path}")

    # Pedalboard 需要 (samples, channels) 格式
    if audio.ndim == 2:
        audio_to_save = audio.T
    else:
        audio_to_save = audio.reshape(-1, 1)

    # 根據副檔名決定格式
    if output_path.lower().endswith(".mp3"):
        with AudioFile(output_path, "w", sample_rate, audio_to_save.shape[1]) as f:
            f.write(audio_to_save)
    else:
        # 預設使用 WAV
        import soundfile as sf
        sf.write(output_path, audio_to_save, sample_rate)

    print(f"儲存完成: {output_path}")


def load_audio(input_path: str) -> tuple:
    """
    載入音訊檔案

    參數:
        input_path: 輸入檔案路徑

    回傳:
        (audio, sample_rate): 音訊陣列 (channels, samples) 和取樣率
    """
    print(f"正在載入音訊: {input_path}")

    with AudioFile(input_path, "r") as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    # Pedalboard 讀取的格式已經是 (channels, samples)，不需要轉置
    print(f"載入完成！取樣率: {sample_rate}, 長度: {audio.shape[-1] / sample_rate:.2f} 秒")

    return audio, sample_rate
