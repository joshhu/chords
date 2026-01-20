# Chords

自動為 MP3 歌曲添加多聲部和聲（三度 + 五度）的 CLI 工具。

## 功能特色

- 使用 Demucs v4 進行音源分離，提取人聲
- 使用 CREPE 深度學習模型偵測音高
- 使用 Librosa 自動分析調性（大調/小調）
- 根據調性智慧選擇和聲音程（大三度/小三度）
- 支援多種和聲類型：三度、五度、低三度、低五度
- 使用 Pedalboard 施加效果（壓縮、殘響）讓和聲更自然
- 支援 GPU 加速（CUDA）

## 安裝

需要 Python 3.12+ 和 [uv](https://github.com/astral-sh/uv)。

```bash
# 複製專案
git clone https://github.com/joshhu/chords.git
cd chords

# 安裝依賴
uv sync
```

### 系統需求

- **rubberband**（可選，用於高品質變調）：
  ```bash
  # Ubuntu/Debian
  sudo apt install rubberband-cli librubberband-dev

  # Fedora/RHEL
  sudo dnf install rubberband

  # Arch Linux
  sudo pacman -S rubberband

  # macOS
  brew install rubberband
  ```

- **libsndfile**（音訊 I/O）：
  ```bash
  # Ubuntu/Debian
  sudo apt install libsndfile1

  # Fedora/RHEL
  sudo dnf install libsndfile

  # Arch Linux
  sudo pacman -S libsndfile

  # macOS（通常已內建）
  brew install libsndfile
  ```

- **ffmpeg**（MP3 編解碼）：
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg

  # Fedora/RHEL
  sudo dnf install ffmpeg

  # Arch Linux
  sudo pacman -S ffmpeg

  # macOS
  brew install ffmpeg
  ```

## 使用方式

```bash
# 基本用法（生成三度+五度和聲）
uv run chords song.mp3

# 指定輸出檔案
uv run chords song.mp3 -o output.mp3

# 只生成三度和聲
uv run chords song.mp3 --harmony third

# 生成低八度和聲
uv run chords song.mp3 --harmony third_lower,fifth_lower

# 調整和聲音量（0.0-1.0）
uv run chords song.mp3 --harmony-volume 0.4

# 不加殘響效果
uv run chords song.mp3 --no-reverb

# 顯示系統資訊（GPU 狀態等）
uv run chords --info

# 查看所有選項
uv run chords --help
```

## 和聲類型

| 類型 | 說明 |
|------|------|
| `third` | 高三度（大調 +4 半音，小調 +3 半音） |
| `fifth` | 高五度（+7 半音） |
| `third_lower` | 低三度 |
| `fifth_lower` | 低五度（-7 半音） |

## 處理流程

```
輸入音訊 → 音源分離 → 音高偵測 → 調性分析 → 和聲生成 → 混音輸出
           (Demucs)   (CREPE)    (Librosa)  (Rubberband) (Pedalboard)
```

## 授權

MIT License
