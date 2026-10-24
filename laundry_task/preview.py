import argparse
import mujoco
import numpy as np
from PIL import Image
from laundry_task.physical import build, ROOT


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pilot_cpu")
    p.add_argument("--frame", type=int, default=200)
    p.add_argument("--azimuth", type=float, default=140)
    args = p.parse_args()
    model = build()
    data = mujoco.MjData(model)
    states = np.load(ROOT / f"laundry_task/output/{args.name}.npz")
    data.qpos[:] = states["qpos"][min(args.frame,len(states["qpos"])-1)]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.27,.08,.7]
    cam.distance = 2.3
    cam.azimuth = args.azimuth
    cam.elevation = -23
    with mujoco.Renderer(model, height=900, width=1200) as r:
        r.update_scene(data, camera=cam)
        Image.fromarray(r.render()).save(ROOT / "laundry_task/output/preview.jpg")
