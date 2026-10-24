"""Run checks and audit the encoded movie, not just the source frames."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from simulate import ROOT


def main():
    out = ROOT/"output"
    tests = subprocess.run([sys.executable, "-m", "unittest", "test_model", "-v"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
    summary = json.loads((out/"episode_results.json").read_text())
    assert summary["simulation_source_sha256"] == hashlib.sha256((ROOT/"simulate.py").read_bytes()).hexdigest()
    movie = out/"humanoid_joint_queue_narrated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(movie), "-f", "null", "-"], check=True)
    metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                                   "-of", "json", str(movie)], text=True))
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"]) == (1920, 1080)
    assert video["r_frame_rate"] == "25/1"
    assert abs(float(metadata["format"]["duration"])-48) < .05
    assert int(video["nb_frames"]) == 1200
    indices = json.loads((out/"movie_frame_index.json").read_text())
    assert len(indices) == 1200
    assert [r["frame"] for r in indices] == list(range(1200))
    assert all(0 <= r["replay_row"] < 600 for r in indices)
    for second in (6, 14, 23, 31, 43):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", str(movie), "-frames:v", "1",
                        str(out/f"encoded_{second:02d}.jpg")], check=True)
    report = {"tests_passed": True, "test_output": tests.stdout+tests.stderr,
              "full_movie_decode_passed": True, "video": {k: video[k] for k in ("width", "height", "r_frame_rate", "nb_frames", "codec_name")},
              "audio_codec": audio["codec_name"], "duration_s": float(metadata["format"]["duration"]),
              "movie_sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
              "movie_bytes": movie.stat().st_size, "render_backend": json.loads((out/"render_backend.json").read_text()),
              "replay_coverage": summary["replay_coverage"], "episode_count": len(summary["results"]),
              "matched_pairs": len(summary["results"])//2, "scope": "exploratory simulation only"}
    (out/"validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
