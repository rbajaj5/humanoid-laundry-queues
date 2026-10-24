"""Mass-bearing G1, passive washer lid and deformable small washcloth."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
HAND_LOCAL = {"left": np.array([.085, .003, 0]), "right": np.array([.085, -.003, 0])}
HINGE = np.array([.64, 0., .84])
HANDLE_LOCAL = np.array([-.34, -.10, .025])
TABLE_CENTER = np.array([.23, .40, .74])
INSIDE_LOW = np.array([.28, -.19, .50])
INSIDE_HIGH = np.array([.62, .19, .835])


def model_xml():
    source = ROOT / "models/unitree_g1"
    root = ET.parse(source / "g1.xml").getroot()
    root.find("compiler").set("meshdir", str(source / "assets"))
    root.find("option").set("timestep", ".002")
    root.find("option").set("iterations", "50")
    collision = root.find(".//default[@class='collision']/geom")
    collision.set("contype", "4")
    collision.set("conaffinity", "15")
    root.remove(root.find("keyframe"))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "headlight", diffuse=".65 .65 .65", ambient=".35 .35 .35")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=".88 .92 .94",
                  rgb2=".88 .92 .94", width="64", height="384")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="-1 2 3", dir="0 0 -1", diffuse=".7 .7 .7")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .1",
                  rgba=".81 .85 .87 1", friction=".8 .01 .001")

    def box(name, pos, size, rgba, **extra):
        extra.setdefault("contype", "1")
        extra.setdefault("conaffinity", "12")
        return ET.SubElement(world, "geom", name=name, type="box",
                             pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                             rgba=rgba, **extra)

    # Separate solid walls leave a real cavity. Transparent front is a visual cutaway.
    box("washer_front", [.26, 0, .43], [.02, .23, .41], ".83 .88 .93 .28")
    box("washer_back", [.66, 0, .43], [.02, .23, .41], ".83 .88 .93 1")
    box("washer_left", [.46, .21, .43], [.18, .02, .41], ".83 .88 .93 1")
    box("washer_right", [.46, -.21, .43], [.18, .02, .41], ".83 .88 .93 .48")
    box("drum_floor", [.46, 0, .485], [.18, .19, .015], ".20 .30 .36 1")
    box("washer_base", [.46, 0, .08], [.20, .23, .06], ".12 .20 .25 1")
    for y in (-.22, .22):
        box(f"rim_{y}", [.46, y, .845], [.20, .015, .015], ".10 .24 .28 1")
    box("control_panel", [.66, 0, .91], [.025, .23, .05], ".12 .25 .30 1")
    box("display", [.631, -.06, .92], [.003, .085, .02], ".15 .78 .64 1", contype="0", conaffinity="0")
    box("table_top", TABLE_CENTER, [.13, .13, .018], ".60 .68 .72 1")
    for x in (.13, .33):
        for y in (.30, .50):
            box(f"table_leg_{x}_{y}", [x, y, .36], [.012, .012, .36], ".28 .34 .40 1")

    lid = ET.SubElement(world, "body", name="washer_lid", pos=" ".join(map(str, HINGE)))
    ET.SubElement(lid, "joint", name="lid_hinge", type="hinge", axis="0 1 0",
                  range="0 1.65", damping=".15", frictionloss=".80", armature=".002",
                  solreffriction=".004 1", solimpfriction=".99 .999 .0001")
    ET.SubElement(lid, "geom", name="lid", type="box", pos="-.19 0 0",
                  size=".19 .205 .012", mass=".25", rgba=".17 .36 .40 .75", contype="2", conaffinity="12")
    ET.SubElement(lid, "geom", name="handle", type="capsule", fromto="-.34 -.14 .025 -.34 -.06 .025",
                  size=".012", mass=".02", rgba=".75 .80 .83 1", contype="2", conaffinity="12")
    ET.SubElement(lid, "site", name="handle_tip", pos=" ".join(map(str, HANDLE_LOCAL)), size=".008")

    for side in HAND_LOCAL:
        body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        for geom in body.findall("geom"):
            if geom.get("class") == "collision":
                geom.set("contype", "16")
                geom.set("conaffinity", "7")
        ET.SubElement(body, "site", name=f"{side}_hand_tip",
                      pos=" ".join(map(str, HAND_LOCAL[side])), size=".009", rgba=".1 .7 .55 1")

    cloth = ET.SubElement(world, "flexcomp", name="washcloth", type="grid", count="4 4 1",
                          spacing=".05 .05 .05", pos=".23 .40 .774", dim="2", mass=".055",
                          radius=".006", rgba=".90 .28 .17 1")
    ET.SubElement(cloth, "edge", equality="true", solref=".008 1", damping=".001")
    ET.SubElement(cloth, "contact", selfcollide="none", condim="3",
                  friction=".65 .005 .001", solref=".008 1", contype="8", conaffinity="7")
    eq = ET.SubElement(root, "equality")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    # Idealized hand-object grasp replaces finger contact for these two pairs.
    ET.SubElement(contact, "exclude", body1="right_wrist_yaw_link", body2="washer_lid")
    for name, hand, obj in (("lid_grasp", "right", "washer_lid"),
                            ("cloth_grasp", "left", "washcloth_0")):
        ET.SubElement(eq, "connect", name=name, body1=f"{hand}_wrist_yaw_link", body2=obj,
                      anchor="0 0 0", active="false", solref=".008 1", solimp=".95 .99 .001")
    return ET.tostring(root, encoding="unicode")


def build():
    model = mujoco.MjModel.from_xml_string(model_xml())
    for name, hand, target in (("lid_grasp", "right", HANDLE_LOCAL),
                               ("cloth_grasp", "left", np.zeros(3))):
        model.eq_data[model.equality(name).id, :6] = np.r_[HAND_LOCAL[hand], target]
    return model


def reset(model):
    data = mujoco.MjData(model)
    original = ET.parse(ROOT / "models/unitree_g1/g1.xml").getroot()
    key = original.find("keyframe/key")
    data.qpos[:36] = np.fromstring(key.get("qpos"), sep=" ")
    data.ctrl[:] = np.fromstring(key.get("ctrl"), sep=" ")
    mujoco.mj_forward(model, data)
    return data


def lid_handle(angle):
    c, s = np.cos(angle), np.sin(angle)
    return HINGE + np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ HANDLE_LOCAL


class Planner:
    """CPU geometric IK supplies joint targets; never moves the simulated root."""
    def __init__(self, model):
        self.model = model
        self.data = reset(model)
        self.home = self.data.qpos.copy()
        self.vars = np.r_[np.arange(3), np.arange(7, 36)]
        self.previous = self.home[self.vars].copy()
        self.feet = [model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
        self.foot_positions = self.data.xpos[self.feet].copy()
        self.foot_rotations = self.data.xmat[self.feet].copy().reshape(2, 3, 3)
        self.com_xy = self.data.subtree_com[1, :2].copy()
        self.hand_ids = [model.site(s + "_hand_tip").id for s in ("left", "right")]
        self.home_hands = self.data.site_xpos[self.hand_ids].copy()
        joints = [model.actuator_trnid[a, 0] for a in range(29)]
        self.lower = np.r_[[-.15, -.12, .62], model.jnt_range[joints, 0] + .001]
        self.upper = np.r_[[.20, .12, .82], model.jnt_range[joints, 1] - .001]

    def solve(self, left, right):
        targets = np.asarray([left, right])
        prior = self.previous.copy()

        def residual(x):
            self.data.qpos[self.vars] = x
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            rotations = self.data.xmat[self.feet].reshape(2, 3, 3)
            errors = [30*(self.data.site_xpos[self.hand_ids] - targets).ravel(),
                      40*(self.data.xpos[self.feet] - self.foot_positions).ravel(),
                      8*(rotations - self.foot_rotations).ravel(),
                      30*(self.data.subtree_com[1, :2] - self.com_xy),
                      .02*(x-prior), .015*(x-self.home[self.vars])]
            return np.concatenate(errors)

        fit = least_squares(residual, self.previous, bounds=(self.lower, self.upper),
                            max_nfev=35, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        self.previous = fit.x
        residual(fit.x)
        return self.data.qpos[7:36].copy(), float(np.max(np.linalg.norm(
            self.data.site_xpos[self.hand_ids]-targets, axis=1)))


def smooth(a, b, fraction):
    f = float(np.clip(fraction, 0, 1))
    f = f*f*(3-2*f)
    return (1-f)*np.asarray(a) + f*np.asarray(b)
