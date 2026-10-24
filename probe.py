"""Development-only baseline checks, not an outcome-selected comparison."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    results = []
    for scene, key in (("scene.xml", "stand"), ("scene_mjx.xml", "home")):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "models/unitree_g1" / scene))
        data = mujoco.MjData(model)
        keyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
        mujoco.mj_resetDataKeyframe(model, data, keyid)
        mujoco.mj_forward(model, data)
        initial = data.qpos.copy()
        low = initial[2]
        for _ in range(round(6 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            low = min(low, float(data.qpos[2]))
        record = {"scene": scene, "mass_kg": float(model.body_mass.sum()),
                  "nu": model.nu, "nq": model.nq, "nv": model.nv,
                  "initial_base_xyz": initial[:3].tolist(), "final_base_xyz": data.qpos[:3].tolist(),
                  "min_base_height": low, "final_base_quaternion": data.qpos[3:7].tolist(),
                  "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}
        results.append(record)
        print(json.dumps(record, indent=2))
    (ROOT / "output/baseline_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
