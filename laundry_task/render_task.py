"""One captioned movie, rendered strictly from the retained CUDA state trace."""
from __future__ import annotations
import argparse
import bisect
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import textwrap

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from laundry_task.physical import ROOT, build

OUT = ROOT / "laundry_task/output"
FPS, DURATION = 25, 66
WHITE, INK, MUTED = "#ffffff", "#17252b", "#52656c"
TEAL, ORANGE, RED = "#007d72", "#b86216", "#b23243"
SKILLS = ["open_door", "grasp_cloth", "insert_cloth", "release_cloth", "retract_hand"]
NAMES = {"idle": "Observe / retain", "open_door": "Open washer lid", "grasp_cloth": "Grasp washcloth",
         "insert_cloth": "Insert washcloth", "release_cloth": "Release washcloth", "retract_hand": "Retract hand"}


@lru_cache(None)
def font(size, bold=False):
    for file in (("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
                 ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(file, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def narration():
    return [
        dict(start=0, end=7, text="One humanoid, one washer, one washcloth. These are recorded CUDA physics states, not a generated animation."),
        dict(start=7, end=19, text="Opening is underway when its acknowledgement is withheld. Retries keep the same command identity, so the receiver does not open twice. Old observations are rejected."),
        dict(start=19, end=31, text="Fresh observations confirm the lid is open. The next skill can queue, but the local executor finishes the current movement before accepting the next one."),
        dict(start=31, end=43, text="The left hand lifts a deformable washcloth and lowers it into the cavity. Grasping uses an idealized contact constraint, not dexterous fingers."),
        dict(start=43, end=55, text="Release requires the cloth to be inside. The hand retracts, and the cloth stays in the washer. Completion comes from physical observations, not a receipt."),
        dict(start=55, end=66, text="Both the stress episode and a timely observation reference completed on the GPU. This is one calibrated setup with hand authored skills, not hardware validation or a general safety guarantee."),
    ]


def annotate_scene(scene, pos, color, radius=.012):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius]*3), pos,
                       np.eye(3).ravel(), np.array(color, dtype=np.float32))
    scene.ngeom += 1


def picture(renderer, m, d, row, camera):
    renderer.update_scene(d, camera=camera)
    for active, side in ((row["lid_grasp_active"], "right"), (row["cloth_grasp_active"], "left")):
        if active:
            annotate_scene(renderer.scene, d.site_xpos[m.site(side+"_hand_tip").id], [.05,.85,.63,1])
    return Image.fromarray(renderer.render())


def task_frames(name="laundry_cuda_delayed"):
    report = json.loads((OUT/f"{name}.json").read_text())
    reference = json.loads((OUT/"laundry_cuda_immediate.json").read_text())
    assert report["summary"]["success"] and reference["summary"]["success"]
    assert report["summary"]["gpu_physics"] and reference["summary"]["gpu_physics"]
    for file, digest in report["summary"]["source_sha256"].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, file
    states = np.load(OUT/f"{name}.npz")
    rows = report["frames"]
    times = [r["time_s"] for r in rows]
    model, data = build(), None
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.27, .08, .70]
    camera.distance, camera.elevation = 2.23, -23
    close_camera = mujoco.MjvCamera()
    close_camera.lookat[:] = [.43, .08, .76]
    close_camera.distance, close_camera.azimuth, close_camera.elevation = 1.02, -55, -52
    segments = narration()
    verified = [e for e in report["supervisor_events"] if e["event"] == "VERIFIED_POSTCONDITION"]
    starts = [e for e in report["physical_events"] if e["event"] == "PHYSICAL_SKILL_START"]
    network = report["network_events"]
    withheld_at = next(e["time_s"] for e in network if e["event"] == "ACK_WITHHELD")
    opened_at = next(e["time_s"] for e in network if e["event"] == "EXECUTOR_FINISHED" and e["skill"] == "open_door")
    ledger = []
    renderer = mujoco.Renderer(model, width=1280, height=820)
    close_renderer = mujoco.Renderer(model, width=566, height=270)
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
    renderer.update_scene(data)
    provenance = {"physics": report["summary"]["device"], "renderer": glGetString(GL_RENDERER).decode(),
                  "graphics_vendor": glGetString(GL_VENDOR).decode(), "fps": FPS,
                  "duration_s": DURATION, "trace": f"{name}.npz",
                  "trace_sha256": hashlib.sha256((OUT/f"{name}.npz").read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "interpolation": "none: each frame selects one retained 25 Hz physics observation",
                  "coverage": f"all {len(rows)} retained observations of the delayed episode; reference summarized only"}
    previous_index, previous_azimuth = -1, None
    scene, close = None, None
    try:
        for frame in range(FPS*DURATION):
            movie_t = frame/FPS
            requested_t = float(np.clip((movie_t-5)/50, 0, 1)*times[-1])
            index = max(0, bisect.bisect_right(times, requested_t)-1)
            row = rows[index]
            t = row["time_s"]
            # A deliberate camera transition reveals the cloth side after opening.
            f = float(np.clip((t-8.4)/1.2, 0, 1))
            camera.azimuth = 70-120*f*f*(3-2*f)
            if previous_index != index or previous_azimuth != camera.azimuth:
                data.qpos[:] = states["qpos"][index]
                data.qvel[:] = states["qvel"][index]
                data.eq_active[:] = states["eq_active"][index]
                mujoco.mj_forward(model, data)
                scene = picture(renderer, model, data, row, camera)
                close = picture(close_renderer, model, data, row, close_camera)
                previous_index, previous_azimuth = index, camera.azimuth
            im = Image.new("RGB", (1920, 1080), WHITE)
            im.paste(scene, (0, 115))
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,1920,110), fill=INK)
            draw.text((34,14), "Open. Load. Verify.", font=font(40,True), fill=WHITE)
            draw.text((35,65), "QUEUE-AWARE HUMANOID TASK COORDINATION", font=font(22), fill="#aed8d2")
            draw.text((1285,22), "CUDA physics | RTX 5070 Ti", font=font(24,True), fill=WHITE)
            draw.text((1285,61), "Synthetic, calibrated demonstration", font=font(21), fill="#d3dce0")
            draw.rectangle((14,125,610,169), fill=WHITE)
            draw.text((28,132), f"Simulation {t:05.2f} s  |  slow replay  |  frame {index:03d}", font=font(21), fill=INK)
            draw.text((30,888), "Transparent washer walls: visual cutaway, not missing collision walls", font=font(20), fill=MUTED)
            draw.text((30,917), "Green marker = idealized grasp active  |  33.34 kg free-base humanoid", font=font(20), fill=MUTED)
            draw.line((1280,115,1280,944), fill="#d8e0e4", width=2)
            x = 1310
            draw.text((x,126), "PHYSICAL TASK / LOCAL EXECUTOR", font=font(19,True), fill=MUTED)
            draw.text((x,155), NAMES[row["skill"]], font=font(31,True), fill=TEAL)
            for k, skill in enumerate(SKILLS):
                y = 205+k*35
                done = any(e["skill"] == skill and e["time_s"] <= t for e in verified)
                active = row["skill"] == skill
                color = TEAL if done else (ORANGE if active else MUTED)
                draw.ellipse((x,y+4,x+14,y+18), fill=color)
                draw.text((x+26,y-2), NAMES[skill], font=font(21,active), fill=INK)
                draw.text((1720,y-2), "verified" if done else ("executing" if active else "waiting"), font=font(20), fill=color)
            age = row["observation_age_s"]
            stale = age is not None and age > .25
            draw.line((x,389,1876,389), fill="#d8e0e4", width=2)
            draw.text((x,402), "MESSAGE COORDINATION", font=font(19,True), fill=MUTED)
            age_text = "--" if age is None else f"{age:.2f} s"
            draw.text((x,435), f"Observation age: {age_text} / 0.25 s limit", font=font(22,True), fill=RED if stale else INK)
            pending = row["pending_id"]
            draw.text((x,470), "Pending: "+(f"skill #{int(pending.rsplit(':',1)[1])+1}" if pending else "none"), font=font(21), fill=INK)
            opening = withheld_at <= t < opened_at
            hint = ("Reject stale observation; keep current skill" if stale else
                    "Open receipt missing; retry same identity" if opening else
                    "Physical checks authorize task progression")
            draw.text((x,504), hint, font=font(20,True), fill=ORANGE if opening or stale else TEAL)
            executed = sum(e["time_s"] <= t for e in starts)
            draw.text((x,538), f"Duplicate attempts blocked: {row['duplicates_blocked']}  |  starts: {executed}/5", font=font(20), fill=INK)
            # Every displayed measurement comes from this row of the retained trace.
            draw.text((x,576), f"Lid {np.degrees(row['lid_angle_rad']):.1f} deg  |  body tilt {row['base_tilt_deg']:.2f} deg", font=font(21), fill=INK)
            draw.text((x,610), "Cloth inside: "+("YES" if row["cloth_inside"] else "NO")+"  |  hand clear: "+("YES" if row["hand_clear"] else "NO"), font=font(21,True), fill=TEAL if row["cloth_inside"] else MUTED)
            im.paste(close, (1310,650))
            draw = ImageDraw.Draw(im)
            draw.rectangle((1310,650,1876,683), fill=WHITE)
            draw.text((1320,654), "Synchronized manipulation close-up", font=font(19), fill=INK)
            # Progress bar is tied to simulation time, not narrative wall time.
            draw.rectangle((28,953,1878,959), fill="#d9e2e5")
            draw.rectangle((28,953,28+1850*t/times[-1],959), fill=TEAL)
            if movie_t >= 55:
                draw.rectangle((30,185,1170,307), fill=WHITE, outline="#c1d9d3", width=2)
                s, ref = report["summary"], reference["summary"]
                draw.text((48,196), f"5 skills completed | {s['retention_seconds']:.0f} s retained after retraction", font=font(30,True), fill=TEAL)
                draw.text((48,242), f"Stress: {s['completion_time_s']:.2f} s  |  reference: {ref['completion_time_s']:.2f} s", font=font(24), fill=INK)
                draw.text((48,276), "Two episodes, one calibrated setup; not a reliability estimate", font=font(19), fill=MUTED)
            caption = next(s["text"] for s in segments if s["start"] <= movie_t < s["end"])
            lines = textwrap.wrap(caption, width=123)
            for j, line in enumerate(lines):
                draw.text((35,974+j*31), line, font=font(25), fill=INK)
            draw.text((35,1047), "Hand-authored IK + ankle feedback; idealized gripper; no hardware or learned-policy validation.", font=font(18), fill=MUTED)
            ledger.append({"video_frame":frame, "video_time_s":movie_t, "trace_index":index,
                           "simulation_time_s":t, "trace":name})
            yield frame, im
    finally:
        renderer.close()
        close_renderer.close()
    (OUT/"task_movie_frames.json").write_text(json.dumps(ledger, indent=2)+"\n")
    (OUT/"task_render_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (OUT/"task_narration.json").write_text(json.dumps(segments, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg must be on PATH")
    samples = {0, 300, 680, 900, 1100, 1450}
    process = None
    if not args.preview:
        process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(OUT/"laundry_task_silent.mp4")], stdin=subprocess.PIPE)
    try:
        for n, im in task_frames():
            if n in samples:
                im.save(OUT/f"task_preview_{n:04d}.jpg", quality=93)
            if process:
                process.stdin.write(im.tobytes())
            if n % 250 == 0:
                print(f"rendered {n}/{FPS*DURATION}", flush=True)
    finally:
        if process:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Video encoder failed")


if __name__ == "__main__":
    main()
