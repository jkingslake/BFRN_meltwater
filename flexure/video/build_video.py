"""
Stitch the rendered scenes into one video, with a subtitle track.

    python build_video.py [quality_dir]      # default quality_dir: 1080p30

Reads   media/videos/explainer/<quality_dir>/S??*.mp4   (rendered by render.sh)
        subtitles/S??*.json                             (written by voiceover.py)
Writes  ice_shelf_flexure_explainer.mp4  (+ .srt) and narration_script.md
"""
import json
import os
import subprocess
import sys

from explainer import SCENES

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = sys.argv[1] if len(sys.argv) > 1 else "1080p30"
SRC = os.path.join(HERE, "media", "videos", "explainer", QDIR)
OUT = os.path.join(HERE, "ice_shelf_flexure_explainer.mp4")
SRT = os.path.join(HERE, "ice_shelf_flexure_explainer.srt")
SCRIPT = os.path.join(HERE, "narration_script.md")


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())


def stamp(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def main():
    files = [os.path.join(SRC, f"{s}.mp4") for s in SCENES]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit(f"missing renders: {missing}")

    # subtitles + script
    offset, n, srt, md = 0.0, 1, [], ["# Narration script\n",
                                      "Generated from the voice-over lines in `explainer.py`.\n"]
    for scene, f in zip(SCENES, files):
        md.append(f"\n## {scene[3:]}\n")
        sub = os.path.join(HERE, "subtitles", f"{scene}.json")
        for clip in json.load(open(sub)) if os.path.exists(sub) else []:
            srt.append(f"{n}\n{stamp(offset + clip['start'])} --> {stamp(offset + clip['end'])}\n"
                       f"{clip['text']}\n")
            md.append(f"> {clip['text']}\n")
            n += 1
        offset += duration(f)
    open(SRT, "w").write("\n".join(srt))
    open(SCRIPT, "w").write("\n".join(md))

    # Each scene's narration ends before its closing fade, so its audio track is
    # shorter than its video.  Pad every audio track to its video length first,
    # otherwise the concat demuxer starts the next scene's voice too early.
    padded_dir = os.path.join(HERE, "media", "padded")
    os.makedirs(padded_dir, exist_ok=True)
    padded = []
    for f in files:
        out = os.path.join(padded_dir, os.path.basename(f))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f, "-c:v", "copy",
                        "-af", f"apad=whole_dur={duration(f):.3f}", "-c:a", "pcm_s16le",
                        "-t", f"{duration(f):.3f}", out.replace(".mp4", ".mov")], check=True)
        padded.append(out.replace(".mp4", ".mov"))

    # concatenate (re-encode once so every join is clean), add soft subtitles
    lst = os.path.join(HERE, "media", "concat.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{f}'\n" for f in padded)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-i", SRT, "-map", "0:v", "-map", "0:a", "-map", "1:s",
           "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-c:s", "mov_text",
           "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", OUT]
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT}  ({offset / 60:.1f} min, {os.path.getsize(OUT) / 1e6:.1f} MB)")
    print(f"wrote {SRT}\nwrote {SCRIPT}")


if __name__ == "__main__":
    main()
