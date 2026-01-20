# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

Chords 是一個自動和聲生成 CLI 工具，可為 MP3 歌曲自動添加多聲部和聲（三度、五度等）。

## 常用指令

```bash
# 安裝依賴（使用 uv）
uv sync

# 執行 CLI
uv run chords song.mp3                      # 基本用法
uv run chords song.mp3 -o output.mp3        # 指定輸出
uv run chords song.mp3 --harmony third      # 只生成三度和聲
uv run chords song.mp3 --harmony-volume 0.4 # 調整和聲音量
uv run chords --info                        # 顯示系統資訊

# 執行測試
uv run pytest                               # 執行所有測試
uv run pytest tests/test_pipeline.py -v    # 執行單一測試檔案
uv run pytest -k "test_frequency"          # 執行符合名稱的測試
```

## 架構概述

處理流程為六個步驟的 pipeline：

```
輸入音訊 → 音源分離 → 音高偵測 → 調性分析 → 和聲生成 → 混音輸出
```

### 核心模組 (`src/chords/`)

- **`cli.py`**: Click CLI 介面，解析參數並呼叫 `pipeline.process_audio()`
- **`pipeline.py`**: 主處理流程，整合所有模組並定義 `ProcessingResult` 資料結構
- **`separator.py`**: 使用 Demucs v4 分離人聲與伴奏，回傳 `SeparatedAudio`
- **`pitch.py`**: 使用 CREPE 深度學習模型偵測音高，回傳 `PitchData`
- **`analyzer.py`**: 使用 Librosa chromagram 分析調性，回傳 `KeyInfo`；包含大/小調音階邏輯
- **`harmony.py`**: 使用 pyrubberband（或 fallback 到 librosa）變調生成和聲，回傳 `HarmonyTrack`
- **`mixer.py`**: 使用 Pedalboard 施加效果（Compressor、Reverb）並混音輸出

### 關鍵資料結構

- `KeyInfo`: 調性資訊（根音、調式、音階音符、信心度）
- `PitchData`: 音高偵測結果（時間序列、頻率、信心度）
- `HarmonyTrack`: 和聲音軌（音訊資料、類型、半音數）
- `MixSettings`: 混音參數（各軌音量、殘響設定）

## 技術細節

- 音訊格式統一為 `(channels, samples)` 的 numpy array
- 支援 GPU 加速（PyTorch CUDA），沒有 GPU 時自動 fallback 到 CPU
- 變調優先使用 pyrubberband（品質較高），不可用時 fallback 到 librosa
