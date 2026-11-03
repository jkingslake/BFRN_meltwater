"""
Minimal offline voice-over for Manim scenes.

Narration is synthesised with Kokoro (an open-weights neural TTS model, run
through ONNX) using a British English voice and British (en-gb) pronunciation.
Clips are cached by a hash of their text, so re-rendering a scene does not
re-synthesise unchanged lines.

Usage inside a Scene::

    with voiceover(self, "The ice bends beneath the lake.") as vo:
        self.play(Create(beam), run_time=vo.duration)
    # the context manager waits out any remaining narration

Model files (about 350 MB) are looked for in $KOKORO_DIR (default ~/kokoro):
    kokoro-v1.0.onnx, voices-v1.0.bin
from https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import soundfile as sf

VOICE = os.environ.get("VO_VOICE", "bf_emma")      # British English, female
LANG = "en-gb"
SPEED = float(os.environ.get("VO_SPEED", "1.0"))
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "audio_cache")
SUBS = os.path.join(HERE, "subtitles")
KOKORO_DIR = os.environ.get("KOKORO_DIR", os.path.expanduser("~/kokoro"))

# words the phonemiser gets wrong -> respellings that it gets right
RESPELL = {
    r"\bisostasy\b": "eye-sosstasy",
    r"\bIsostasy\b": "Eye-sosstasy",
}

_kokoro = None


def _engine():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx"),
                         os.path.join(KOKORO_DIR, "voices-v1.0.bin"))
    return _kokoro


def speak(text: str) -> tuple[str, float]:
    """Return (wav path, duration in s) for ``text``, synthesising if needed."""
    spoken = text
    for pat, rep in RESPELL.items():
        spoken = re.sub(pat, rep, spoken)
    key = hashlib.sha1(f"{VOICE}|{SPEED}|{LANG}|{spoken}".encode()).hexdigest()[:16]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key}.wav")
    if not os.path.exists(path):
        samples, sr = _engine().create(spoken, voice=VOICE, speed=SPEED, lang=LANG)
        pad = np.zeros(int(0.12 * sr), dtype=samples.dtype)  # tiny breath at the end
        sf.write(path, np.concatenate([samples, pad]), sr)
    info = sf.info(path)
    return path, info.frames / info.samplerate


@dataclass
class Clip:
    text: str
    duration: float
    start: float


@contextmanager
def voiceover(scene, text: str, gap: float = 0.25):
    """Play narration over the animations inside the ``with`` block."""
    path, dur = speak(text)
    start = scene.time
    scene.add_sound(path)
    clip = Clip(text, dur, start)
    _log(scene, clip)
    yield clip
    remaining = dur - (scene.time - start)
    if remaining > 1e-3:
        scene.wait(remaining)
    if gap > 0:
        scene.wait(gap)


def _log(scene, clip: Clip):
    """Record narration timing so render.sh can build a subtitle file."""
    os.makedirs(SUBS, exist_ok=True)
    name = type(scene).__name__
    path = os.path.join(SUBS, f"{name}.json")
    log = getattr(scene, "_vo_log", None)
    if log is None:
        log = scene._vo_log = []
    log.append({"start": clip.start, "end": clip.start + clip.duration, "text": clip.text})
    with open(path, "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":  # quick test: python voiceover.py "Hello there"
    import sys
    p, d = speak(" ".join(sys.argv[1:]) or "Meltwater lakes bend ice shelves.")
    print(p, f"{d:.2f} s")
