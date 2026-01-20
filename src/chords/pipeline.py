"""
主處理流程模組

整合所有模組，完成從輸入到輸出的完整處理
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .analyzer import KeyInfo, analyze_key
from .harmony import HarmonyTrack, generate_multi_harmony
from .mixer import MixSettings, load_audio, mix_audio, save_audio
from .pitch import PitchData, detect_pitch
from .separator import SeparatedAudio, separate_audio

console = Console()


@dataclass
class ProcessingResult:
    """處理結果"""
    output_path: str
    key_info: KeyInfo
    harmony_tracks: List[HarmonyTrack]
    sample_rate: int


def process_audio(
    input_path: str,
    output_path: str,
    harmony_types: List[str] = ["third", "fifth"],
    harmony_volume: float = 0.6,
    add_reverb: bool = True,
    voice_type: Optional[str] = None,
    skip_separation: bool = False
) -> ProcessingResult:
    """
    處理音訊並生成和聲

    參數:
        input_path: 輸入音訊檔案路徑
        output_path: 輸出音訊檔案路徑
        harmony_types: 要生成的和聲類型列表
        harmony_volume: 和聲音量（0.0 - 1.0）
        add_reverb: 是否為和聲加入殘響效果
        voice_type: 和聲聲音類型（"male" 降八度、"female" 升八度、None 不調整）
        skip_separation: 是否跳過音源分離（用於測試）

    回傳:
        ProcessingResult: 處理結果
    """
    console.print(f"[bold blue]開始處理: {input_path}[/bold blue]")
    console.print(f"輸出路徑: {output_path}")
    console.print(f"和聲類型: {', '.join(harmony_types)}")
    console.print()

    # Step 1: 載入音訊
    console.print("[bold]Step 1/6: 載入音訊[/bold]")
    audio, sample_rate = load_audio(input_path)
    console.print()

    # Step 2: 音源分離
    if skip_separation:
        console.print("[bold yellow]跳過音源分離（測試模式）[/bold yellow]")
        vocals = audio
        accompaniment = audio * 0  # 靜音伴奏
    else:
        console.print("[bold]Step 2/6: 音源分離[/bold]")
        separated = separate_audio(audio, sample_rate)
        vocals = separated.vocals
        accompaniment = separated.accompaniment
    console.print()

    # Step 3: 音高偵測（用於調性分析）
    console.print("[bold]Step 3/6: 音高偵測[/bold]")
    pitch_data = detect_pitch(vocals, sample_rate, model_capacity="small")
    console.print()

    # Step 4: 調性分析
    console.print("[bold]Step 4/6: 調性分析[/bold]")
    key_info = analyze_key(vocals, sample_rate)
    console.print()

    # Step 5: 生成和聲
    console.print("[bold]Step 5/6: 生成和聲[/bold]")
    harmonies = generate_multi_harmony(
        vocals,
        sample_rate,
        key_info=key_info,
        harmony_types=harmony_types,
        voice_type=voice_type
    )
    console.print()

    # Step 6: 混音輸出
    console.print("[bold]Step 6/6: 混音輸出[/bold]")
    mix_settings = MixSettings(
        vocals_volume=1.0,
        harmony_volume=harmony_volume,
        accompaniment_volume=1.0,
        add_reverb=add_reverb
    )
    mixed = mix_audio(vocals, harmonies, accompaniment, sample_rate, mix_settings)

    # 儲存結果
    save_audio(mixed, output_path, sample_rate)

    console.print()
    console.print(f"[bold green]處理完成！[/bold green]")
    console.print(f"輸出檔案: {output_path}")

    return ProcessingResult(
        output_path=output_path,
        key_info=key_info,
        harmony_tracks=harmonies,
        sample_rate=sample_rate
    )


def check_gpu() -> bool:
    """檢查 GPU 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def print_system_info():
    """印出系統資訊"""
    import sys

    console.print("[bold]系統資訊[/bold]")
    console.print(f"Python 版本: {sys.version}")

    # 檢查 GPU
    if check_gpu():
        import torch
        console.print(f"[green]GPU 可用: {torch.cuda.get_device_name(0)}[/green]")
    else:
        console.print("[yellow]GPU 不可用，將使用 CPU（處理較慢）[/yellow]")

    console.print()
