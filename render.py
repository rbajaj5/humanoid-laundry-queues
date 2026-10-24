"""Render the retained physics states, with synchronized telemetry and captions."""
import argparse
from functools import lru_cache
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
from simulate import ROOT, build_model

OUT = ROOT / "output"
WIDTH, HEIGHT, FPS, LENGTH = 1920, 1080, 25, 48
INK = "#20252c"
MUTED = "#5c6872"
COLORS = ["#bd3464", "#008978"]
CAPTIONS = [
    (0, 6, "A humanoid is not a moving point. Every link contributes mass and inertia."),
    (6, 15, "Both robots receive the same reach commands, disturbances, and internal delays."),
    (15, 24, "Left: joint groups apply arrivals independently. The arms can move before the legs."),
    (24, 33, "Right: wait for a complete command version, then update the whole body together."),
    (33, 40, "Synchronization reduces the tested posture disturbance. Both robots stay upright."),
    (40, 48, "This is a standing-reach prerequisite, not a trained or safety-certified laundry robot.")
]


@lru_cache(maxsize=32)
def font(size, bold=False):
    choices = [Path("C:/Windows/Fonts")/("segoeuib.ttf" if bold else "segoeui.ttf"),
               Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for p in choices:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def load():
    config = json.loads((ROOT/"config.json").read_text())
    data, rows = [], []
    for arm in config["arms"]:
        data.append(np.load(OUT/f"replay_{arm}.npz"))
        with (OUT/f"replay_{arm}.csv").open() as handle:
            records = list(csv.DictReader(handle))
        for r in records:
            for key in r:
                if key not in ("decision", "fallen"):
                    r[key] = float(r[key])
        rows.append(records)
    return config, data, rows


def marker(scene, pos, size, color):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size]*3), np.array(pos),
                        np.eye(3).ravel(), np.array(color, dtype=np.float32))
    scene.ngeom += 1


def line(scene, a, b, color):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array(color, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .003, np.array(a), np.array(b))
    scene.ngeom += 1


def picture(renderer, model, state, qpos, target, camera):
    state.qpos[:] = qpos
    mujoco.mj_forward(model, state)
    renderer.update_scene(state, camera=camera)
    scene = renderer.scene
    for g in scene.geoms[:scene.ngeom]:
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid >= 0 and model.geom_bodyid[g.objid] > 0:
            g.rgba[3] = .38
    for body in range(1, model.nbody):
        if model.body_mass[body] > .55:
            marker(scene, state.xipos[body], .011+.005*np.cbrt(model.body_mass[body]), [1, .62, .12, 1])
    com = state.subtree_com[1].copy()
    marker(scene, com, .027, [0, .8, .7, 1])
    floorpoint = com.copy()
    floorpoint[2] = .012
    line(scene, com, floorpoint, [0, .65, .55, 1])
    marker(scene, floorpoint, .018, [0, .8, .7, 1])
    marker(scene, target, .027, [.85, .15, .7, .5])
    return Image.fromarray(renderer.render())


def chart(draw, bounds, rows, key, maximum, current, title, unit):
    x, y, w, h = bounds
    draw.text((x, y), title, fill=INK, font=font(22, True))
    left, top, right, bottom = x+52, y+35, x+w-18, y+h-24
    draw.line((left, top, left, bottom, right, bottom), fill="#b7bec5", width=1)
    for frac in (0, .5, 1):
        yy = bottom-frac*(bottom-top)
        draw.line((left, yy, right, yy), fill="#e2e6e9", width=1)
        draw.text((x, yy-10), f"{maximum*frac:g}", fill=MUTED, font=font(16))
    for values, color in zip(rows, COLORS):
        points = [(left+r["time_s"]/12*(right-left), bottom-min(r[key]/maximum, 1)*(bottom-top))
                  for r in values[:current+1]]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    cursor = left+rows[0][current]["time_s"]/12*(right-left)
    draw.line((cursor, top, cursor, bottom), fill=INK, width=2)
    draw.text((left, bottom+3), "0", fill=MUTED, font=font(15))
    draw.text((right-80, bottom+3), "12 seconds", fill=MUTED, font=font(15))
    draw.text((right-55, top-27), unit, fill=MUTED, font=font(16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    config, replays, rows = load()
    model = build_model(config)
    state = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=510, width=940)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.40, 0, .64]
    camera.distance = 2.55
    camera.azimuth = 65
    camera.elevation = -13
    hardware = {"renderer": glGetString(GL_RENDERER).decode(), "vendor": glGetString(GL_VENDOR).decode(),
                "physics": "MuJoCo CPU", "encoding": "libx264 CPU"}
    frames = [470] if args.preview else range(FPS*LENGTH)
    process = None
    if not args.preview:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required")
        process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
            "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(OUT/"humanoid_joint_queue_silent.mp4")], stdin=subprocess.PIPE)
    frame_index = []
    last_i, panels = None, None
    for frame in frames:
        movie_time = frame/FPS
        sim_time = np.clip((movie_time-4)/2.5, .02, 12)
        i = min(599, max(0, round(sim_time/.02)-1))
        if last_i != i:
            panels = [picture(renderer, model, state, p["qpos"][i], p["target"], camera) for p in replays]
            last_i = i
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f9fafb")
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 18), "ONE BODY. DIFFERENT COMMAND AGES.", fill=INK, font=font(34, True))
        draw.text((30, 63), "Unitree G1 mass-bearing reach test  |  33.34 kg  |  29 joints + free base  |  illustrative simulation",
                  fill=MUTED, font=font(23))
        for arm in range(2):
            x = 10+960*arm
            canvas.paste(panels[arm], (x, 153))
            draw.text((x+20, 109), ["Independent joint-group arrivals", "Queue-aware whole-body bundles"][arm],
                      fill=COLORS[arm], font=font(28, True))
            r = rows[arm][i]
            draw.rectangle((x+17, 614, x+923, 657), fill="#ffffff")
            draw.text((x+27, 620), f"Tilt {r['base_tilt_deg']:.2f} deg     command skew {r['queue_skew_ms']:.0f} ms     height {r['base_height_m']:.3f} m",
                      fill=INK, font=font(22, True))
            draw.text((x+20, 670), r["decision"], fill=COLORS[arm], font=font(21, True))
            draw.text((x+20, 701), f"Command age at dispatch {r['oldest_applied_age_ms']:.0f} ms  |  packets in flight {r['inflight_packets']:.0f}",
                      fill=MUTED, font=font(20))
        draw.line((960, 100, 960, 738), fill="#d4dadf", width=2)
        draw.text((30, 743), "Amber: link centers of mass   |   Teal: whole-body COM + ground projection   |   Magenta: nominal reach target",
                  fill=MUTED, font=font(19))
        chart(draw, (30, 784, 600, 185), rows, "base_tilt_deg", 5, i, "BODY TILT", "deg")
        chart(draw, (660, 784, 600, 185), rows, "queue_skew_ms", 700, i, "COMMAND VERSION SKEW", "ms")
        chart(draw, (1290, 784, 600, 185), rows, "right_hand_error_m", .6, i, "DISTANCE TO REACH TARGET", "m")
        caption = next(text for start, end, text in CAPTIONS if start <= movie_time < end)
        draw.rectangle((0, 978, 1920, 1035), fill="#e6f1ee")
        draw.text((30, 987), caption, fill=INK, font=font(25, True))
        draw.text((30, 1046), f"t = {rows[0][i]['time_s']:05.2f} s  |  matched seed 24100  |  legs delayed up to 640 ms  |  MuJoCo CPU physics; GPU graphics",
                  fill=MUTED, font=font(19))
        if movie_time >= 36:
            summary = json.loads((OUT/"episode_results.json").read_text())
            group = [r for r in summary["aggregate"] if r["profile"] == config["movie_profile"]]
            draw.rectangle((340, 221, 1580, 570), fill="#ffffff", outline="#b7bec5", width=2)
            draw.text((385, 245), f"MATCHED 600 ms SKEW TEST: {group[0]['n']} SEEDS PER ARM", fill=INK, font=font(29, True))
            for j, g in enumerate(group):
                yy = 310+72*j
                draw.text((385, yy), ["Independent", "Coherent bundles"][j], fill=COLORS[j], font=font(27, True))
                draw.text((790, yy), f"peak tilt {g['mean_peak_tilt_deg']:.2f} +/- {g['sd_peak_tilt_deg']:.2f} deg",
                          fill=INK, font=font(27))
                draw.text((1300, yy), f"falls {g['falls']}/{g['n']}", fill=INK, font=font(27))
            draw.text((385, 465), f"Means +/- sample SD. Reach successes: {group[0]['reach_successes']}/{group[0]['n']} and {group[1]['reach_successes']}/{group[1]['n']}.", fill=MUTED, font=font(24))
            draw.text((385, 512), "No learned policy, cloth manipulation, washing cycle, or hardware safety claim.", fill=MUTED, font=font(23))
        if args.preview or frame in (150, 470, 760, 1050):
            canvas.save(OUT/f"preview_{frame:04d}.jpg", quality=93)
        if process:
            process.stdin.write(canvas.tobytes())
        frame_index.append({"frame": frame, "movie_time_s": movie_time, "replay_row": i,
                            "simulation_time_s": rows[0][i]["time_s"]})
        if frame % 200 == 0:
            print("rendered", frame, "of", FPS*LENGTH, flush=True)
    if process:
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        (OUT/"movie_frame_index.json").write_text(json.dumps(frame_index, separators=(",", ":")), encoding="utf-8")
    (OUT/"render_backend.json").write_text(json.dumps(hardware, indent=2)+"\n", encoding="utf-8")
    renderer.close()
    print(hardware)


if __name__ == "__main__":
    main()
