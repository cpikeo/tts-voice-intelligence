"""Real authorized XTTS integration, opt-in local assets only; never downloads weights.

Set all six TTS_XTTS_* variables below after human consent/license review.
This checks cross-text execution and signal validity, NOT target similarity/naturalness.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qa
import tts
from engines import XTTSEngine

KEYS = ("MODEL", "CONFIG", "LANGUAGE", "REFERENCE", "AUTHORIZATION", "PURPOSE")
VALUES = {k: os.environ.get("TTS_XTTS_" + k) for k in KEYS}
pytestmark = pytest.mark.skipif(not all(VALUES.values()),
                              reason="真实 XTTS 需本地模型/配置、语言、合法参考、授权与用途六个环境变量")


def test_real_authorized_clone_cross_text(tmp_path):
    engine = XTTSEngine(VALUES["MODEL"], VALUES["CONFIG"], VALUES["LANGUAGE"],
                        VALUES["REFERENCE"], VALUES["AUTHORIZATION"], VALUES["PURPOSE"])
    texts = (["这句话不同于参考录音。", "请听另一段内容的音色是否保持稳定。"]
             if VALUES["LANGUAGE"].lower().startswith("zh") else
             ["This text differs from the reference recording.", "Listen to the second passage for voice consistency."])
    for index, text in enumerate(texts):
        audio, sr, report = tts.synthesize_text(engine, text, processing="minimal", pauses={})
        wav = tts.write_audio(tmp_path / f"passage-{index}.wav", audio, sr)
        x, rate, info = qa.load_mono(wav)
        assert not [c for c in qa.run_signal_checks(x, rate, info) if c["status"] == "fail"]
        assert report["authorization"]["reference_sha256"]
        assert report["validation"]["human_listening"] == "not_performed"
