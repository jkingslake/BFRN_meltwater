#!/usr/bin/env bash
# Render every scene of the explainer (in parallel) and stitch them together.
#
#   ./render.sh          # 1080p, 30 fps  (final)
#   ./render.sh low      # 480p, 15 fps   (quick check)
#
# Needs: manim (Community), a LaTeX install, ffmpeg, kokoro-onnx + model files
# (see voiceover.py).  Narration clips are cached in audio_cache/.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "low" ]]; then
  QFLAGS=(-ql --frame_rate 15); QDIR=480p15
else
  QFLAGS=(-qh --frame_rate 30); QDIR=1080p30
fi
JOBS=${JOBS:-4}

SCENES=$(python -c "from explainer import SCENES; print(' '.join(SCENES))")

# 1. synthesise all narration once (serially, so the TTS model loads only once)
python - <<'EOF'
import ast, re, explainer, voiceover, inspect
src = inspect.getsource(explainer)
for m in re.finditer(r'voiceover\(self,\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)', src):
    text = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', m.group(1)))
    voiceover.speak(text)
print("narration cached")
EOF

# 2. render scenes in parallel
printf '%s\n' $SCENES | xargs -P "$JOBS" -I{} \
  manim "${QFLAGS[@]}" --progress_bar none --disable_caching explainer.py {}

# 3. stitch + subtitles
python build_video.py "$QDIR"
