"""Run the real rigid/deformable simulation; retain physical state and protocol log."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from laundry_task.physical import (
    ROOT, Planner, build, reset, smooth, lid_handle, INSIDE_LOW, INSIDE_HIGH,
)
from laundry_task.supervisor import Config, Coordinator, Observation, Skill, SkillReceiver

OUT = ROOT / "laundry_task/output"
DT = .04


class Physics:
    def __init__(self, backend):
        self.model = build()
        self.data = reset(self.model)
        self.backend = backend
        self.gpu = None
        if backend == "cuda":
            import warp as wp
            import mujoco_warp as mjw
            wp.config.kernel_cache_dir = str(ROOT / ".warp-cache")
            wp.init()
            if not wp.get_cuda_devices():
                raise RuntimeError("CUDA unavailable; no CPU fallback")
            self.wp, self.mjw = wp, mjw
            with wp.ScopedDevice("cuda:0"):
                self.gm = mjw.put_model(self.model)
                self.gpu = mjw.put_data(self.model, self.data, nconmax=384, njmax=1536)
                print("Compile/warm CUDA task kernels", flush=True)
                mjw.step(self.gm, self.gpu)
                wp.synchronize()
                self.gpu = mjw.put_data(self.model, self.data, nconmax=384, njmax=1536)
                with wp.ScopedCapture() as cap:
                    for _ in range(round(DT / self.model.opt.timestep)):
                        mjw.step(self.gm, self.gpu)
                self.graph = cap.graph
            self.device_name = wp.get_device("cuda:0").name
        else:
            self.device_name = "MuJoCo CPU (development pilot only)"

    def step(self, ctrl, eq):
        self.data.ctrl[:] = ctrl
        self.data.eq_active[:] = eq
        if self.gpu is None:
            for _ in range(round(DT / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)
        else:
            self.gpu.ctrl.assign(np.asarray([ctrl], dtype=np.float32))
            self.gpu.eq_active.assign(np.asarray([eq], dtype=bool))
            self.wp.capture_launch(self.graph)
            self.data.qpos[:] = self.gpu.qpos.numpy()[0]
            self.data.qvel[:] = self.gpu.qvel.numpy()[0]
            self.data.time = float(self.gpu.time.numpy()[0])
        # CPU kinematics for sensing/rendering only; no CPU integration on CUDA path.
        mujoco.mj_forward(self.model, self.data)


class Executor:
    def __init__(self, model, data):
        self.model = model
        self.planner = Planner(model)
        self.home = self.planner.home_hands.copy()
        self.ids = self.planner.hand_ids
        self.hand_targets = self.home.copy()
        self.current = None
        self.done = False
        self.stage_time = 0.
        self.grab_time = None
        self.release_time = None
        self.eq = data.eq_active.copy()
        self.lid_eq = model.equality("lid_grasp").id
        self.cloth_eq = model.equality("cloth_grasp").id
        self.events = []
        self.max_ik_error = 0.
        self.cloth_pick = None
        self.last_control = data.ctrl.copy()

    def log(self, t, event, **kw):
        self.events.append({"time_s": t, "event": event, **kw})

    def start(self, command, t, data):
        self.current, self.stage_time, self.done = command, t, False
        self.start_hands = data.site_xpos[self.ids].copy()
        self.hand_targets = self.start_hands.copy()
        self.grab_time = self.release_time = None
        self.cloth_pick = data.flexvert_xpos[0].copy()
        self.log(t, "PHYSICAL_SKILL_START", command_id=command.command_id, skill=command.skill.value)

    def control(self, t, data):
        if self.current is None or self.done:
            return self.last_control if hasattr(self, "last_control") else data.ctrl.copy()
        skill, elapsed = self.current.skill, t-self.stage_time
        left, right = self.hand_targets.copy()
        hands = data.site_xpos[self.ids]
        angle = float(data.qpos[self.model.joint("lid_hinge").qposadr[0]])
        if skill == Skill.OPEN:
            if self.grab_time is None:
                if elapsed < 1.6:
                    right = smooth(self.start_hands[1], lid_handle(0)+[0,0,.06], elapsed/1.6)
                else:
                    right = smooth(lid_handle(0)+[0,0,.06], data.site_xpos[self.model.site("handle_tip").id],
                                   (elapsed-1.6)/.8)
                distance = np.linalg.norm(hands[1]-data.site_xpos[self.model.site("handle_tip").id])
                if elapsed >= 2.4 and distance < .025:
                    self.eq[self.lid_eq] = True
                    self.grab_time = t
                    self.log(t, "LID_GRASP", distance_m=float(distance))
            elif self.release_time is None:
                right = lid_handle(float(smooth(0, 1.24, (t-self.grab_time)/3.)))
                if t-self.grab_time > 3.2 and angle > 1.10:
                    self.eq[self.lid_eq] = False
                    self.release_time = t
                    self.release_hand = hands[1].copy()
                    self.log(t, "LID_RELEASE", angle_rad=angle)
            else:
                right = smooth(self.release_hand, self.home[1], (t-self.release_time)/1.6)
                self.done = t-self.release_time > 2.
        elif skill == Skill.GRASP:
            if self.grab_time is None:
                if elapsed < 1.4:
                    left = smooth(self.start_hands[0], self.cloth_pick+[0,0,.09], elapsed/1.4)
                else:
                    left = smooth(self.cloth_pick+[0,0,.09], data.flexvert_xpos[0], (elapsed-1.4)/.9)
                distance = np.linalg.norm(hands[0]-data.flexvert_xpos[0])
                if elapsed >= 2.3 and distance < .025:
                    self.eq[self.cloth_eq] = True
                    self.grab_time = t
                    self.cloth_pick = data.flexvert_xpos[0].copy()
                    self.log(t, "CLOTH_GRASP", distance_m=float(distance))
            else:
                left = smooth(self.cloth_pick, self.cloth_pick+[0,0,.28], (t-self.grab_time)/1.3)
                self.done = t-self.grab_time > 1.7
        elif skill == Skill.INSERT:
            above = np.array([.43, .015, 1.09])
            inside = np.array([.43, .015, .80])
            left = (smooth(self.start_hands[0], above, elapsed/2.) if elapsed < 2.
                    else smooth(above, inside, (elapsed-2.)/2.))
            self.done = elapsed > 4.8
        elif skill == Skill.RELEASE:
            if self.release_time is None:
                self.eq[self.cloth_eq] = False
                self.release_time = t
                self.log(t, "CLOTH_RELEASE")
            left = self.start_hands[0]
            self.done = elapsed > 1.6
        elif skill == Skill.RETRACT:
            above = np.array([.43, .015, 1.08])
            outside = np.array([.12, .32, 1.08])
            if elapsed < 1.2:
                left = smooth(self.start_hands[0], above, elapsed/1.2)
            elif elapsed < 2.4:
                left = smooth(above, outside, (elapsed-1.2)/1.2)
            else:
                left = smooth(outside, self.home[0], (elapsed-2.4)/1.3)
            self.done = elapsed > 4.0
        self.hand_targets = np.asarray([left, right])
        self.last_control, error = self.planner.solve(left, right)
        self.max_ik_error = max(error, self.max_ik_error)
        return self.last_control

    def balance(self, control, data):
        """Small ankle feedback through the real torque-limited joint actuators."""
        result = control.copy()
        rotation = data.xmat[self.model.body("pelvis").id].reshape(3, 3)
        pitch = np.arctan2(rotation[0, 2], rotation[2, 2])
        roll = np.arctan2(-rotation[1, 2], rotation[2, 2])
        error = data.subtree_com[1, :2]-self.planner.com_xy
        correction = np.clip([.6*pitch+.10*data.qvel[4]+1.2*error[0],
                              .6*roll+.10*data.qvel[3]-1.2*error[1]], -.15, .15)
        for side in ("left", "right"):
            result[self.model.actuator(side+"_ankle_pitch_joint").id] += correction[0]
            result[self.model.actuator(side+"_ankle_roll_joint").id] += correction[1]
        return np.clip(result, self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1])


def sense(model, data, executor, t, sequence):
    quat = data.qpos[3:7]
    tilt = float(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1))))
    verts = data.flexvert_xpos.copy()
    inside = bool(np.all(verts > INSIDE_LOW-[0,0,.012]) and np.all(verts < INSIDE_HIGH))
    hands = data.site_xpos[executor.ids]
    clear = bool(all((h[0] < INSIDE_LOW[0]-.025 or abs(h[1]) > .245 or h[2] > .925) for h in hands))
    angle = float(data.qpos[model.joint("lid_hinge").qposadr[0]])
    # A grasp is measured by both constraint status and actual point proximity.
    gap = float(np.linalg.norm(hands[0]-verts[0]))
    grasped = bool(executor.eq[executor.cloth_eq] and gap < .035)
    o = Observation(sequence, t, bool(.65 < data.qpos[2] < .90 and tilt < 15),
                    angle, grasped, inside, clear)
    return o, {"base_height_m": float(data.qpos[2]), "base_tilt_deg": tilt,
               "cloth_inside": inside, "cloth_grasped": grasped, "hand_clear": clear,
               "cloth_grasp_gap_m": gap, "lid_angle_rad": angle,
               "cloth_min_z_m": float(verts[:,2].min()), "cloth_max_z_m": float(verts[:,2].max())}


def run(backend, scenario, name, duration=40.):
    physics = Physics(backend)
    model, data = physics.model, physics.data
    ex = Executor(model, data)
    for _ in range(25):
        physics.step(data.ctrl.copy(), ex.eq)
    config = Config(command_ttl_s=10., door_open_threshold_rad=1.10)
    co, receiver = Coordinator(name, config), SkillReceiver(name, config)
    commands, receipts, observations = [], [], []
    seq, last_obs, complete_at = 0, None, None
    rows, states, events, velocities, active_eq, controls = [], [], [], [], [], []
    failures, duplicates = [], 0
    start = time.perf_counter()
    for frame in range(round(duration/DT)):
        t = round(frame*DT, 9)
        o, metrics = sense(model, data, ex, t, seq)
        seq += 1
        if not o.body_ready:
            failures.append({"time_s": t, "reason": "body_envelope"})
            break
        if not np.isfinite(data.qpos).all():
            failures.append({"time_s": t, "reason": "nonfinite"})
            break
        if ex.current is not None and ex.done:
            if receiver.finish(ex.current.command_id, t, o):
                events.append({"time_s": t, "event": "EXECUTOR_FINISHED", "skill": ex.current.skill.value})
                ex.current = None
        observations.append((t+(0.12 if scenario == "delayed" else 0.), o))
        due = [item for item in observations if item[0] <= t+1e-8]
        observations = [item for item in observations if item[0] > t+1e-8]
        blackout = scenario == "delayed" and 4.0 <= t < 5.2
        if due and not blackout:
            last_obs = due[-1][1]
        for at, cid, status in list(receipts):
            if at <= t:
                co.receive_receipt(cid, t, status)
                events.append({"time_s": t, "event": "RECEIPT_ARRIVED", "command_id": cid})
                receipts.remove((at, cid, status))
        if t >= 1. and last_obs is not None:
            proposed = co.update(t, last_obs)
            if proposed:
                commands.append((t+.12, proposed))
        for at, command in list(commands):
            if at <= t+1e-8:
                verdict = receiver.dequeue(command, t, o)
                events.append({"time_s": t, "event": verdict, "command_id": command.command_id})
                if verdict == "EXECUTE":
                    ex.start(command, t, data)
                    if scenario == "delayed" and command.skill == Skill.OPEN:
                        events.append({"time_s": t, "event": "ACK_WITHHELD", "command_id": command.command_id})
                    else:
                        delay = 2.5 if scenario == "delayed" else .04
                        receipts.append((t+delay, command.command_id, "received"))
                if verdict == "DUPLICATE_NO_EXECUTION":
                    duplicates += 1
                commands.remove((at, command))
        ctrl = ex.balance(ex.control(t, data), data)
        row = {"time_s": t, "frame": frame, "skill": ex.current.skill.value if ex.current else "idle",
               "decision": co.events[-1]["event"] if co.events else "INITIALIZE",
               "completed_skills": len(co.completed), "duplicates_blocked": duplicates,
               "observation_age_s": t-last_obs.observed_at if last_obs else None,
               "pending_id": co.pending.command_id if co.pending else None,
               "lid_grasp_active": bool(ex.eq[ex.lid_eq]),
               "cloth_grasp_active": bool(ex.eq[ex.cloth_eq]), **metrics}
        rows.append(row)
        states.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        active_eq.append(ex.eq.copy())
        controls.append(ctrl.copy())
        if len(co.completed) == 5 and ex.current is None:
            if complete_at is None:
                complete_at = t
            if not o.cloth_inside:
                failures.append({"time_s": t, "reason": "cloth_escaped_after_release"})
            if t-complete_at >= 3.:
                break
        if ex.current is not None and t-ex.stage_time > 12:
            failures.append({"time_s": t, "reason": "skill_timeout", "skill": ex.current.skill.value})
            break
        physics.step(ctrl, ex.eq)
        if frame % 100 == 0:
            print(f"{backend} {scenario} t={t:.2f} skill={row['skill']} lid={o.door_angle_rad:.2f} "
                  f"inside={o.cloth_inside} tilt={metrics['base_tilt_deg']:.2f} ik={ex.max_ik_error:.4f}", flush=True)
    success = bool(complete_at is not None and rows[-1]["time_s"]-complete_at >= 3. and not failures)
    summary = {"name": name, "backend": backend, "device": physics.device_name, "gpu_physics": backend == "cuda",
               "packages": {p: importlib.metadata.version(p) for p in
                            (["mujoco", "mujoco-warp", "warp-lang", "numpy", "scipy"] if backend == "cuda"
                             else ["mujoco", "numpy", "scipy"])},
               "scenario": scenario, "control_dt_s": DT, "physics_dt_s": model.opt.timestep,
               "humanoid_mass_kg": float(model.body_subtreemass[1]), "cloth_mass_kg": .055,
               "grasp_model": "proximity-gated compliant point constraints; no dexterous finger model",
               "command_delay_s": .12, "observation_delay_s": .12 if scenario == "delayed" else 0.,
               "observation_blackout_s": [4.,5.2] if scenario == "delayed" else None,
               "receipt_delay_s": 2.5 if scenario == "delayed" else .04,
               "withheld_receipt": "open_door" if scenario == "delayed" else None,
               "settle_s": 1., "body_height_envelope_m": [.65,.90], "body_tilt_envelope_deg": 15.,
               "grasp_activation_distance_m": .025, "grasp_verification_distance_m": .035,
               "cloth_inside_low_m": (INSIDE_LOW-[0,0,.012]).tolist(),
               "cloth_inside_high_m": INSIDE_HIGH.tolist(), "retention_check_required_s": 3.,
               "queue_config": asdict(config), "frames": len(rows), "completed_skills": co.completed,
               "completion_time_s": complete_at, "retention_seconds": None if complete_at is None else rows[-1]["time_s"]-complete_at,
               "success": success, "failures": failures, "duplicates_blocked": duplicates,
               "max_ik_error_m": ex.max_ik_error, "wall_seconds": time.perf_counter()-start,
               "max_base_tilt_deg": max(r["base_tilt_deg"] for r in rows),
               "maximum_lid_angle_rad": max(r["lid_angle_rad"] for r in rows),
               "source_sha256": {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                    ["laundry_task/physical.py", "laundry_task/run_physical.py", "laundry_task/supervisor.py"]},
               "limitations": ["Synthetic single-cloth task", "Hand-authored IK skills, not a trained policy",
                               "No hardware validation or general safety proof", "One robot and one washer session"]}
    OUT.mkdir(exist_ok=True)
    (OUT/f"{name}.json").write_text(json.dumps({"summary": summary, "frames": rows,
        "network_events": events, "supervisor_events": co.events, "physical_events": ex.events}, indent=2)+"\n")
    np.savez_compressed(OUT/f"{name}.npz", qpos=states, qvel=velocities, eq_active=active_eq, ctrl=controls)
    print(json.dumps(summary, indent=2), flush=True)
    return success


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--scenario", choices=["delayed", "immediate"], default="delayed")
    p.add_argument("--name", default="laundry_cuda_delayed")
    p.add_argument("--duration", type=float, default=40.)
    args = p.parse_args()
    if not run(args.backend, args.scenario, args.name, args.duration):
        raise SystemExit(2)
