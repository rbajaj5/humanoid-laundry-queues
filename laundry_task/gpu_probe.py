"""Fail-closed CUDA forward-dynamics probe; NOT a laundry task demonstration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from simulate import build_model, reset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1:
        parser.error("worlds and steps must be positive")
    out = ROOT / "laundry_task/output"
    out.mkdir(exist_ok=True)
    wp.config.kernel_cache_dir = str(ROOT / ".warp-cache")
    wp.init()
    if not wp.get_cuda_devices():
        raise RuntimeError("No CUDA physics device; refusing CPU fallback")
    config = json.loads((ROOT / "config.json").read_text())
    model = build_model(config)
    data = reset(model)
    initial = data.qpos.copy()
    device = wp.get_device("cuda:0")
    with wp.ScopedDevice(device):
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=args.worlds, nconmax=128, njmax=768)
        assert d.qpos.device.is_cuda
        # Warm up kernels, then capture 10 real physics steps per graph launch.
        print("Warming up CUDA dynamics kernels", flush=True)
        mjw.step(m, d)
        wp.synchronize()
        d = mjw.put_data(model, data, nworld=args.worlds, nconmax=128, njmax=768)
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mjw.step(m, d)
        # Capture records GPU work but does not execute it; this starts at reset.
        samples = []
        start = time.perf_counter()
        for i in range(args.steps // 10):
            wp.capture_launch(capture.graph)
            if i % 10 == 9:
                samples.append(d.qpos.numpy())
        for _ in range(args.steps % 10):
            mjw.step(m, d)
        wp.synchronize()
        elapsed = time.perf_counter() - start
        final = d.qpos.numpy()
        samples.append(final)
        simulation_times = d.time.numpy()
    poses = np.asarray(samples)
    height = poses[:, :, 2]
    # Rotation of body z axis relative to world z from free-joint quaternion.
    quat = poses[:, :, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2*(quat[..., 1]**2 + quat[..., 2]**2), -1, 1)))
    passed = bool(np.isfinite(poses).all() and np.min(height) > .55 and
                  np.max(tilt) < 45 and
                  np.allclose(simulation_times, args.steps * model.opt.timestep, atol=1e-4) and
                  np.max(np.abs(final - initial)) > 1e-6)
    report = {
        "purpose": "CUDA forward-dynamics compatibility and standing probe, not laundry success",
        "device": device.name, "array_device": str(d.qpos.device), "gpu_physics": True,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["mujoco", "mujoco-warp", "warp-lang", "numpy"]},
        "worlds": args.worlds, "worlds_identical": True,
        "steps_per_world": args.steps, "physics_dt_s": model.opt.timestep,
        "simulated_seconds_per_world": simulation_times.tolist(),
        "physics_wall_seconds_excluding_compilation": elapsed,
        "initial_qpos_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "final_qpos_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
        "minimum_sampled_base_height_m": float(height.min()),
        "maximum_sampled_base_tilt_deg": float(tilt.max()),
        "sampling_note": "diagnostics sampled every 100 steps plus final; not continuous guarantees",
        "passed": passed,
        "limits": ["Fixed standing target only", "Not independent reliability trials",
                   "No cloth, grasp, door operation or pretrained policy",
                   "Old published v1 experiment remains CPU physics"],
    }
    (out / "gpu_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("CUDA probe ran but its numerical/posture checks failed")


if __name__ == "__main__":
    main()
