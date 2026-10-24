"""Fit locally generated speech into explicit caption windows; validate durations."""
import json
import shutil
import subprocess
from pathlib import Path
from render import OUT


def main():
    segments = json.loads((OUT/"narration.json").read_text(encoding="utf-8-sig"))
    ffmpeg, probe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    filters, inputs, checks = [], [], []
    for i, seg in enumerate(segments):
        path = OUT/f"narration_{i}.wav"
        meta = json.loads(subprocess.check_output([probe, "-v", "error", "-show_format", "-of", "json", str(path)], text=True))
        duration = float(meta["format"]["duration"])
        tempo = max(1., duration/(seg["end"]-seg["start"]-.3))
        if tempo > 1.4:
            raise ValueError("Narration too long; shorten the script")
        command += ["-i", str(path)]
        filters.append(f"[{i}:a]atempo={tempo:.8f},adelay={round(seg['start']*1000)}:all=1[a{i}]")
        inputs.append(f"[a{i}]")
        checks.append({**seg, "raw_duration_s": duration, "tempo": tempo, "fitted_duration_s": duration/tempo})
    filters.append("".join(inputs)+f"amix=inputs={len(inputs)}:normalize=0,apad,atrim=duration=48[out]")
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "48000", "-ac", "1", str(OUT/"narration.wav")]
    subprocess.run(command, check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(OUT/"humanoid_joint_queue_silent.mp4"),
                    "-i", str(OUT/"narration.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-t", "48", "-movflags", "+faststart", str(OUT/"humanoid_joint_queue_narrated.mp4")], check=True)
    (OUT/"audio_validation.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
