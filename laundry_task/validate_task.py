"""Validate retained experiment, encoded media and complete movie-to-trace links."""
import hashlib
import io
import json
import shutil
import subprocess
import unittest
import wave

import numpy as np

from laundry_task.render_task import OUT, ROOT, DURATION, FPS


def main():
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "test_model", "laundry_task.test_supervisor", "laundry_task.test_physical"])
    text = io.StringIO()
    result = unittest.TextTestRunner(stream=text, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(text.getvalue())
    video = OUT/"humanoid_laundry_task_narrated.mp4"
    probe = json.loads(subprocess.check_output([shutil.which("ffprobe"), "-v", "error", "-count_frames",
        "-show_streams", "-show_format", "-of", "json", str(video)], text=True))
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"]) == (1920,1080)
    assert v["avg_frame_rate"] == f"{FPS}/1"
    assert int(v["nb_read_frames"]) == DURATION*FPS
    assert abs(float(probe["format"]["duration"])-DURATION) < .1
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], check=True)
    frames = json.loads((OUT/"task_movie_frames.json").read_text())
    report = json.loads((OUT/"laundry_cuda_delayed.json").read_text())
    assert len(frames) == DURATION*FPS
    for n, f in enumerate(frames):
        assert f["video_frame"] == n and f["video_time_s"] == n/FPS
        assert f["simulation_time_s"] == report["frames"][f["trace_index"]]["time_s"]
    assert set(f["trace_index"] for f in frames) == set(range(len(report["frames"])))
    audio = json.loads((OUT/"task_audio_validation.json").read_text())
    for s in audio:
        assert s["fitted_duration_s"] <= s["end"]-s["start"]-.29
    with wave.open(str(OUT/"task_narration.wav")) as w:
        assert w.getsampwidth() == 2
        signal = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
        rms = float(np.sqrt(np.mean(signal**2)))
        assert rms > 100
    files = [video, OUT/"laundry_cuda_delayed.json", OUT/"laundry_cuda_delayed.npz",
             OUT/"laundry_cuda_immediate.json", OUT/"laundry_cuda_immediate.npz",
             OUT/"task_movie_frames.json", OUT/"task_render_provenance.json", OUT/"task_audio_validation.json"]
    validation = {"passed": True, "tests_run": result.testsRun, "test_output": text.getvalue(),
                  "movie": {"width": v["width"], "height": v["height"], "fps": FPS,
                            "frames": int(v["nb_read_frames"]), "duration_s": float(probe["format"]["duration"]),
                            "video_codec": v["codec_name"], "audio_codec": a["codec_name"],
                            "full_decode": "passed", "audio_rms_pcm16": rms},
                  "all_retained_stress_observations_in_movie": True,
                  "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  "scope": "Two final CUDA episodes, one calibrated setup. No policy superiority claim."}
    (OUT/"task_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({k:v for k,v in validation.items() if k != "test_output"}, indent=2))


if __name__ == "__main__":
    main()
