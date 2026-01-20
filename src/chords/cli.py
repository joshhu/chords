"""
CLI 介面模組

提供命令列介面讓使用者操作
"""

from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .pipeline import check_gpu, print_system_info, process_audio

console = Console()


def parse_harmony_types(harmony_str: str) -> List[str]:
    """解析和聲類型字串"""
    return [h.strip() for h in harmony_str.split(",")]


@click.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="輸出檔案路徑（預設為 input_harmony.mp3）"
)
@click.option(
    "--harmony", "-h",
    default="third,fifth",
    help="和聲類型，以逗號分隔。可選: third, fifth, third_lower, fifth_lower（預設: third,fifth）"
)
@click.option(
    "--harmony-volume", "-v",
    default=0.6,
    type=float,
    help="和聲音量，範圍 0.0-1.0（預設: 0.6）"
)
@click.option(
    "--no-reverb",
    is_flag=True,
    help="不加入殘響效果"
)
@click.option(
    "--skip-separation",
    is_flag=True,
    hidden=True,
    help="跳過音源分離（僅供測試）"
)
@click.option(
    "--info",
    is_flag=True,
    help="顯示系統資訊"
)
@click.version_option(version=__version__)
def main(
    input_file: Optional[str],
    output: Optional[str],
    harmony: str,
    harmony_volume: float,
    no_reverb: bool,
    skip_separation: bool,
    info: bool
):
    """
    自動和聲生成工具

    為 MP3 歌曲自動添加多聲部和聲（三度 + 五度）

    \b
    使用範例:
        chords song.mp3                           # 基本用法
        chords song.mp3 -o output.mp3             # 指定輸出檔案
        chords song.mp3 --harmony third           # 只生成三度和聲
        chords song.mp3 --harmony-volume 0.4      # 調整和聲音量
    """
    # 印出標題
    console.print(Panel.fit(
        f"[bold blue]Chords[/bold blue] v{__version__}\n"
        "自動和聲生成工具",
        border_style="blue"
    ))
    console.print()

    # 顯示系統資訊
    if info:
        print_system_info()
        return

    # 檢查是否有提供輸入檔案
    if input_file is None:
        console.print("[red]錯誤: 請提供輸入音訊檔案[/red]")
        console.print("使用方式: chords <input_file> [OPTIONS]")
        console.print("使用 --help 查看更多選項")
        raise click.Abort()

    # 檢查 GPU
    if not check_gpu():
        console.print(
            "[yellow]警告: GPU 不可用，處理速度可能較慢[/yellow]"
        )
        console.print()

    # 決定輸出路徑
    if output is None:
        input_path = Path(input_file)
        output = str(input_path.parent / f"{input_path.stem}_harmony{input_path.suffix}")

    # 解析和聲類型
    harmony_types = parse_harmony_types(harmony)

    # 驗證和聲音量
    if not 0.0 <= harmony_volume <= 1.0:
        console.print("[red]錯誤: 和聲音量必須在 0.0 到 1.0 之間[/red]")
        raise click.Abort()

    try:
        # 執行處理
        result = process_audio(
            input_path=input_file,
            output_path=output,
            harmony_types=harmony_types,
            harmony_volume=harmony_volume,
            add_reverb=not no_reverb,
            skip_separation=skip_separation
        )

        # 印出結果摘要
        console.print()
        console.print(Panel(
            f"[bold green]處理成功！[/bold green]\n\n"
            f"偵測調性: [cyan]{result.key_info.root} {result.key_info.mode}[/cyan]\n"
            f"音階音符: {', '.join(result.key_info.scale_notes)}\n"
            f"生成和聲: {len(result.harmony_tracks)} 軌\n"
            f"輸出檔案: [blue]{result.output_path}[/blue]",
            title="結果",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[red]錯誤: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    main()
