"""
Pipeline 整合測試
"""

import numpy as np
import pytest


class TestAnalyzer:
    """調性分析模組測試"""

    def test_note_names(self):
        """測試音符名稱列表"""
        from chords.analyzer import NOTE_NAMES

        assert len(NOTE_NAMES) == 12
        assert NOTE_NAMES[0] == "C"
        assert NOTE_NAMES[9] == "A"

    def test_scale_intervals(self):
        """測試音階間隔"""
        from chords.analyzer import MAJOR_SCALE_INTERVALS, MINOR_SCALE_INTERVALS

        # 大調音階有 7 個音
        assert len(MAJOR_SCALE_INTERVALS) == 7
        # 小調音階有 7 個音
        assert len(MINOR_SCALE_INTERVALS) == 7


class TestPitch:
    """音高偵測模組測試"""

    def test_frequency_to_midi(self):
        """測試頻率轉 MIDI"""
        from chords.pitch import frequency_to_midi

        # A4 = 440 Hz = MIDI 69
        freq = np.array([440.0, 0, 880.0])
        midi = frequency_to_midi(freq)

        assert midi[0] == pytest.approx(69, abs=0.01)
        assert midi[1] == 0  # 無效頻率
        assert midi[2] == pytest.approx(81, abs=0.01)  # A5

    def test_midi_to_frequency(self):
        """測試 MIDI 轉頻率"""
        from chords.pitch import midi_to_frequency

        # MIDI 69 = A4 = 440 Hz
        midi = np.array([69, 0, 81])
        freq = midi_to_frequency(midi)

        assert freq[0] == pytest.approx(440.0, abs=0.1)
        assert freq[1] == 0  # 無效音符
        assert freq[2] == pytest.approx(880.0, abs=0.1)


class TestHarmony:
    """和聲生成模組測試"""

    def test_harmony_types(self):
        """測試和聲類型解析"""
        from chords.cli import parse_harmony_types

        result = parse_harmony_types("third,fifth")
        assert result == ["third", "fifth"]

        result = parse_harmony_types("third")
        assert result == ["third"]


class TestMixer:
    """混音模組測試"""

    def test_mix_settings_defaults(self):
        """測試混音設定預設值"""
        from chords.mixer import MixSettings

        settings = MixSettings()

        assert settings.vocals_volume == 1.0
        assert settings.harmony_volume == 0.6
        assert settings.accompaniment_volume == 1.0
        assert settings.add_reverb is True
