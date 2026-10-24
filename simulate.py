"""Free-base, mass-bearing humanoid experiment using MuJoCo rigid-body physics.

The queue supervisor changes packet application semantics, NOT the robot state.
No learned policy, cloth, washing cycle, or hardware validation is claimed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent
GROUPS = [np.arange(0, 12), np.arange(12, 15), np.arange(15, 22), np.arange(22, 29)]
GROUP_NAMES = ["legs", "waist", "left_arm", "right_arm"]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_model(config):
    source = ROOT / "models/unitree_g1"
    tree = ET.parse(source / "g1.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(source / "assets"))
    root.find("option").set("timestep", str(config["physics_dt_s"]))
    ET.SubElement(root.find("asset"), "texture", type="skybox", builtin="gradient", rgb1=".88 .91 .94",
                  rgb2=".88 .91 .94", width="64", height="384")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.35 0.35 0.35")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -3 4", dir="0 0 -1", diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .1", rgba=".87 .90 .91 1",
                  friction="0.8 0.01 0.001")
    # Fixed appliance proxy: an actual collision box, but no operable door or cloth.
    ET.SubElement(world, "geom", name="washer", type="box", pos=".94 0 .53", size=".3 .34 .53",
                  rgba=".94 .95 .96 1")
    ET.SubElement(world, "geom", name="washer_window", type="cylinder", pos=".632 0 .54", size=".22 .006",
                  euler="0 1.57079632679 0", rgba=".12 .20 .24 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="washer_panel", type="box", pos=".632 0 .96", size=".006 .29 .065",
                  rgba=".16 .21 .25 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="washer_button", type="sphere", pos=".62 -.16 .96", size=".016",
                  rgba=".0 .65 .52 1", contype="0", conaffinity="0")
    for side, y in (("left", .003), ("right", -.003)):
        body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(body, "site", name=f"{side}_hand_tip", pos=f".085 {y} 0", size=".008", rgba="0 .6 .5 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


def reset(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    mujoco.mj_forward(model, data)
    return data


def make_path(model):
    """Offline mass-aware, symmetric reach poses. Root is free during simulation.

    IK constrains foot x/z, foot pitch and whole-body COM x, using model inertias.
    This is a nominal static posture construction, NOT a dynamic safety proof.
    """
    data = reset(model)
    home = data.qpos.copy()
    foot = model.body("left_ankle_roll_link").id
    comx = data.subtree_com[1, 0]
    footxz = data.xpos[foot, [0, 2]].copy()
    qs, diagnostics = [], []
    previous = np.array([0., home[2], 0., 0.])
    for fraction in np.linspace(0, 1, 101):
        def pose(x):
            data.qpos[:] = home
            data.qpos[0], data.qpos[2] = x[:2]
            for side in ("left", "right"):
                for joint, value in (("hip_pitch", x[2]), ("knee", .60*fraction), ("ankle_pitch", x[3]),
                                     ("shoulder_pitch", .2-1.20*fraction), ("elbow", 1.28-.8*fraction)):
                    data.qpos[model.joint(f"{side}_{joint}_joint").qposadr[0]] = value
            data.qpos[model.joint("waist_pitch_joint").qposadr[0]] = .35*fraction
            mujoco.mj_forward(model, data)
        def residual(x):
            pose(x)
            rot = data.xmat[foot].reshape(3, 3)
            return np.r_[data.xpos[foot, [0, 2]]-footxz, np.arctan2(rot[0, 2], rot[2, 2]),
                         data.subtree_com[1, 0]-comx]
        opt = least_squares(residual, previous, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        previous = opt.x
        pose(previous)
        assert np.max(np.abs(residual(previous))) < 1e-7
        qs.append(data.qpos.copy())
        diagnostics.append(float(np.linalg.norm(residual(previous))))
    poses = np.asarray(qs)
    data.qpos[:] = poses[-1]
    mujoco.mj_forward(model, data)
    target = data.site_xpos[model.site("right_hand_tip").id].copy()
    return poses, target, max(diagnostics)


def phase(t):
    if t < 2:
        return 0.
    if t < 4:
        x = (t-2)/2
        return x*x*(3-2*x)
    if t < 7:
        return 1.
    if t < 10:
        x = (t-7)/3
        return 1-x*x*(3-2*x)
    return 0.


def inputs(model, poses, config, seed, profile):
    n = round(config["duration_s"]/config["control_dt_s"])
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, config["command_noise_sd_rad"], (n, model.nu))
    force = rng.normal(0, config["torso_force_sd_N"], (n, 3))
    force[:, 2] = 0
    jitter = rng.integers(0, 3, (n, 4))
    command = np.empty((n, model.nu))
    for i in range(n):
        f = phase(i*config["control_dt_s"])*100
        lo = min(int(f), 99)
        command[i] = (1-f+lo)*poses[lo, 7:] + (f-lo)*poses[lo+1, 7:] + noise[i]
    delay = np.zeros((n, 4), dtype=int)
    if profile == "uniform_200ms":
        delay[:] = round(.2/config["control_dt_s"])
    elif profile.startswith("skew_"):
        duration = float(profile.split("_")[1].removesuffix("ms"))/1000
        delay[:] = jitter
        for i in range(n):
            t = i*config["control_dt_s"]
            if 2 <= t < 4 or 7 <= t < 9:
                delay[i, 0] += round(duration/config["control_dt_s"])
                delay[i, 1] += round(.25*duration/config["control_dt_s"])
    return command, force, delay


class PacketQueue:
    def __init__(self, home, delay, arm):
        self.applied = home.copy()
        self.version = np.full(4, -1, dtype=int)
        self.arm = arm
        self.arrivals = {}
        self.received = {}
        self.last = -1
        self.stale = 0
        self.inflight = 0
        self.delay = delay

    def tick(self, i, commands):
        for g in range(4):
            self.arrivals.setdefault(i+int(self.delay[i, g]), []).append((i, g))
            self.inflight += 1
        packets = self.arrivals.pop(i, [])
        for seq, g in packets:
            self.inflight -= 1
            if self.arm == "independent":
                if seq > self.version[g]:
                    self.version[g] = seq
                    self.applied[GROUPS[g]] = commands[seq, GROUPS[g]]
                else:
                    self.stale += 1
            elif seq > self.last:
                self.received[seq] = self.received.get(seq, 0) | (1 << g)
            else:
                self.stale += 1
        decision = "APPLY INDEPENDENTLY" if self.arm == "independent" else "WAIT FOR COMPLETE BUNDLE"
        if self.arm == "coherent":
            complete = [seq for seq, mask in self.received.items() if mask == 15]
            if complete:
                seq = max(complete)
                self.last = seq
                self.version[:] = seq
                self.applied[:] = commands[seq]
                self.received = {s: m for s, m in self.received.items() if s > seq}
                decision = "APPLY COMPLETE BUNDLE"
        return self.applied.copy(), decision


def support_margin(model, data):
    # Static projected support diagnostic; dynamic balance requires more than this.
    points = [data.geom_xpos[g, :2] for g in range(model.ngeom)
              if model.geom_bodyid[g] in (model.body("left_ankle_roll_link").id,
                                         model.body("right_ankle_roll_link").id)
              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    polygon = ConvexHull(np.asarray(points))
    return float(-np.max(polygon.equations[:, :2] @ data.subtree_com[1, :2] + polygon.equations[:, 2]))


def simulate(model, poses, target, config, seed, profile, arm, keep=False):
    data = reset(model)
    substeps = round(config["control_dt_s"]/model.opt.timestep)
    for _ in range(round(config["settle_s"]/model.opt.timestep)):
        mujoco.mj_step(model, data)
    commands, forces, delay = inputs(model, poses, config, seed, profile)
    queue = PacketQueue(data.ctrl.copy(), delay, arm)
    filtered = data.ctrl.copy()
    torso = model.body("torso_link").id
    hand = model.site("right_hand_tip").id
    rows, snapshots = [], []
    first_fall = None
    peak_torque = 0.
    for i in range(len(commands)):
        applied, decision = queue.tick(i, commands)
        data.xfrc_applied[torso, :3] = forces[i]
        for _ in range(substeps):
            alpha = -np.expm1(-model.opt.timestep/config["setpoint_lag_s"])
            filtered += alpha*(applied-filtered)
            data.ctrl[:] = filtered
            mujoco.mj_step(model, data)
            peak_torque = max(peak_torque, float(np.max(np.abs(data.qfrc_actuator))))
        mujoco.mj_forward(model, data)
        t = (i+1)*config["control_dt_s"]
        tilt = np.degrees(np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1, 1)))
        fallen = data.qpos[2] < config["fall_height_m"] or tilt > config["fall_tilt_deg"]
        if fallen and first_fall is None:
            first_fall = t
        ground_force = 0.
        for c in range(data.ncon):
            contact = data.contact[c]
            if model.geom("floor").id in contact.geom:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, wrench)
                ground_force += max(0., wrench[0])
        row = {"time_s": t, "base_height_m": float(data.qpos[2]), "base_tilt_deg": float(tilt),
               "com_x_m": float(data.subtree_com[1, 0]), "com_y_m": float(data.subtree_com[1, 1]),
               "com_z_m": float(data.subtree_com[1, 2]), "static_support_margin_m": support_margin(model, data),
               "right_hand_error_m": float(np.linalg.norm(data.site_xpos[hand]-target)),
               "joint_tracking_rmse_rad": float(np.sqrt(np.mean((data.qpos[7:]-applied)**2))),
               "queue_skew_ms": int(np.ptp(queue.version))*config["control_dt_s"]*1000,
               "oldest_applied_age_ms": int(i-min(queue.version))*config["control_dt_s"]*1000,
               "inflight_packets": queue.inflight, "buffered_versions": len(queue.received),
               "ground_normal_N": float(ground_force), "fallen": bool(fallen), "decision": decision}
        for g, name in enumerate(GROUP_NAMES):
            row[f"{name}_command_version"] = int(queue.version[g])
        row["issued_version"] = i
        rows.append(row)
        if keep:
            snapshots.append(data.qpos.copy())
    window = [r for r in rows if config["reach_window_s"][0] < r["time_s"] <= config["reach_window_s"][1]]
    dwell = sum(r["right_hand_error_m"] <= config["reach_tolerance_m"] and not r["fallen"] for r in window)*config["control_dt_s"]
    reached = [r["time_s"] for r in rows if 2 < r["time_s"] <= config["reach_window_s"][1]
               and r["right_hand_error_m"] <= config["reach_tolerance_m"] and not r["fallen"]]
    result = {"seed": seed, "profile": profile, "arm": arm, "fell": first_fall is not None,
              "first_fall_s": first_fall, "reach_dwell_s": round(dwell, 4),
              "first_reach_s": reached[0] if reached else None,
              "reach_success": dwell >= config["reach_duration_required_s"] and first_fall is None,
              "min_base_height_m": min(r["base_height_m"] for r in rows),
              "max_base_tilt_deg": max(r["base_tilt_deg"] for r in rows),
              "min_static_support_margin_m": min(r["static_support_margin_m"] for r in rows),
              "mean_joint_tracking_rmse_rad": float(np.mean([r["joint_tracking_rmse_rad"] for r in rows])),
              "max_queue_skew_ms": max(r["queue_skew_ms"] for r in rows),
              "max_command_age_ms": max(r["oldest_applied_age_ms"] for r in rows),
              "discarded_stale_packets": queue.stale, "peak_joint_actuator_torque_Nm": peak_torque,
              "command_sha256": digest(commands), "force_sha256": digest(forces), "delay_sha256": digest(delay),
              "qpos_sha256": digest(np.array(snapshots)) if keep else None}
    return result, rows, np.asarray(snapshots)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    model = build_model(config)
    poses, target, residual = make_path(model)
    inventory = [{"body": model.body(i).name, "mass_kg": float(model.body_mass[i]),
                  "com_local_m": model.body_ipos[i].tolist(), "inertia_principal_kg_m2": model.body_inertia[i].tolist()}
                 for i in range(1, model.nbody)]
    write_csv(out / "link_inertias.csv", inventory)
    results = []
    started = time.perf_counter()
    for profile in (["zero", "skew_600ms"] if args.pilot else config["profiles"]):
        for seed in (config["seeds"][:1] if args.pilot else config["seeds"]):
            for arm in config["arms"]:
                keep = seed == config["movie_seed"] and profile == config["movie_profile"]
                result, rows, snapshots = simulate(model, poses, target, config, seed, profile, arm, keep)
                results.append(result)
                print(profile, seed, arm, "fall", result["fell"], "reach", result["reach_success"],
                      "tilt", round(result["max_base_tilt_deg"], 2), "skew", result["max_queue_skew_ms"], flush=True)
                if keep:
                    np.savez_compressed(out / f"replay_{arm}.npz", qpos=snapshots, poses=poses, target=target)
                    write_csv(out / f"replay_{arm}.csv", rows)
    filename = "pilot_results" if args.pilot else "episode_results"
    write_csv(out / f"{filename}.csv", results)
    summary = {"config": config, "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
               "python": platform.python_version(), "compute": "CPU MuJoCo, not GPU physics or neural-policy training",
               "mass_kg": float(model.body_mass.sum()), "actuated_joints": model.nu, "free_base": model.nq == model.nu+7,
               "reach_target_m": target.tolist(), "ik_residual": residual, "seconds": time.perf_counter()-started,
               "config_sha256": hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest(), "results": results}
    summary["simulation_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["replay_coverage"] = {"retained_full_poses": 2, "total_episodes": len(results),
                                  "seed": config["movie_seed"], "profile": config["movie_profile"]}
    if not args.pilot:
        summary["aggregate"] = []
        for profile in config["profiles"]:
            for arm in config["arms"]:
                group = [r for r in results if r["profile"] == profile and r["arm"] == arm]
                summary["aggregate"].append({"profile": profile, "arm": arm, "n": len(group),
                    "falls": sum(r["fell"] for r in group), "reach_successes": sum(r["reach_success"] for r in group),
                    "mean_peak_tilt_deg": float(np.mean([r["max_base_tilt_deg"] for r in group])),
                    "sd_peak_tilt_deg": float(np.std([r["max_base_tilt_deg"] for r in group], ddof=1)),
                    "min_support_margin_m": min(r["min_static_support_margin_m"] for r in group),
                    "mean_reach_dwell_s": float(np.mean([r["reach_dwell_s"] for r in group])),
                    "mean_first_reach_s": float(np.mean([r["first_reach_s"] for r in group if r["first_reach_s"] is not None])),
                    "max_command_age_ms": max(r["max_command_age_ms"] for r in group)})
    (out / f"{filename}.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Saved", out / f"{filename}.json")


if __name__ == "__main__":
    main()
