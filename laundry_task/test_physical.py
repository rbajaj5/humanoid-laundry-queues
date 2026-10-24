"""Validate the actual model and retained GPU trajectories, not a cartoon path."""
import hashlib
import json
import unittest

import mujoco
import numpy as np

from laundry_task.physical import ROOT, build, reset, INSIDE_LOW, INSIDE_HIGH
from laundry_task.run_physical import Executor, sense
from laundry_task.supervisor import ORDER


class PhysicalModelTests(unittest.TestCase):
    def test_free_base_and_passive_lid(self):
        m = build()
        self.assertEqual(m.nu, 29)
        self.assertEqual(m.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertNotIn(m.joint("lid_hinge").id, m.actuator_trnid[:, 0])
        self.assertAlmostEqual(m.body_subtreemass[1], 33.341142, places=5)
        self.assertEqual(m.nflexvert, 16)
        self.assertFalse(np.any(m.eq_type == mujoco.mjtEq.mjEQ_WELD))

    def test_initial_task_is_not_precompleted(self):
        m = build()
        d = reset(m)
        ex = Executor(m, d)
        o, _ = sense(m, d, ex, 0., 0)
        self.assertAlmostEqual(o.door_angle_rad, 0.)
        self.assertFalse(o.cloth_inside)
        self.assertFalse(o.cloth_grasped)
        self.assertFalse(ex.eq[ex.lid_eq])
        self.assertFalse(ex.eq[ex.cloth_eq])


class RetainedCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {}
        for scenario in ("delayed", "immediate"):
            p = ROOT / f"laundry_task/output/laundry_cuda_{scenario}"
            if not p.with_suffix(".json").exists():
                raise RuntimeError(f"Missing required CUDA run: {p}")
            cls.records[scenario] = (json.loads(p.with_suffix(".json").read_text()),
                                     np.load(p.with_suffix(".npz")))

    def test_compute_and_source_provenance(self):
        for report, trace in self.records.values():
            s = report["summary"]
            self.assertTrue(s["gpu_physics"])
            self.assertEqual(s["backend"], "cuda")
            self.assertIn("RTX", s["device"])
            for file, sha in s["source_sha256"].items():
                self.assertEqual(hashlib.sha256((ROOT/file).read_bytes()).hexdigest(), sha)
            self.assertEqual(s["frames"], len(trace["qpos"]))
            for a in trace.values():
                self.assertTrue(np.isfinite(a).all())

    def test_unique_execution_and_verified_completion(self):
        for report, _ in self.records.values():
            starts = [e for e in report["physical_events"] if e["event"] == "PHYSICAL_SKILL_START"]
            self.assertEqual([e["skill"] for e in starts], [s.value for s in ORDER])
            self.assertEqual(len({e["command_id"] for e in starts}), 5)
            verified = [e for e in report["supervisor_events"] if e["event"] == "VERIFIED_POSTCONDITION"]
            self.assertEqual([e["skill"] for e in verified], [s.value for s in ORDER])
            self.assertTrue(report["summary"]["success"])
            self.assertEqual(report["summary"]["failures"], [])

    def test_grasp_proximity_and_receipt_stress(self):
        report, _ = self.records["delayed"]
        grasps = [e for e in report["physical_events"] if e["event"] in ("LID_GRASP", "CLOTH_GRASP")]
        self.assertEqual(len(grasps), 2)
        self.assertTrue(all(e["distance_m"] < .025 for e in grasps))
        events = report["network_events"]
        withheld = [e for e in events if e["event"] == "ACK_WITHHELD"]
        self.assertEqual(len(withheld), 1)
        self.assertFalse(any(e["event"] == "RECEIPT_ARRIVED" and e["command_id"] == withheld[0]["command_id"] for e in events))
        duplicates = sum(e["event"] == "DUPLICATE_NO_EXECUTION" for e in events)
        self.assertGreater(duplicates, 0)
        self.assertEqual(duplicates, report["summary"]["duplicates_blocked"])
        self.assertTrue(any(e["event"] == "REJECT_OBSERVATION" for e in report["supervisor_events"]))

    def test_metrics_recomputed_from_every_retained_pose(self):
        for report, trace in self.records.values():
            m = build()
            d = reset(m)
            ex = Executor(m, d)
            for row, q, v, eq in zip(report["frames"], trace["qpos"], trace["qvel"], trace["eq_active"], strict=True):
                d.qpos[:], d.qvel[:], ex.eq[:] = q, v, eq
                mujoco.mj_forward(m, d)
                obs, metrics = sense(m, d, ex, row["time_s"], row["frame"])
                self.assertTrue(obs.body_ready)
                # The runner samples physical state before a new latch is enabled;
                # logged active bits describe the command applied in the next step.
                for key in ("base_height_m", "base_tilt_deg", "lid_angle_rad", "cloth_min_z_m", "cloth_max_z_m"):
                    self.assertAlmostEqual(metrics[key], row[key], places=8)
                self.assertEqual(obs.cloth_inside, row["cloth_inside"])
                self.assertEqual(obs.hand_clear, row["hand_clear"])
                if row["time_s"] >= report["summary"]["completion_time_s"]:
                    self.assertTrue(np.all(d.flexvert_xpos > INSIDE_LOW-[0,0,.012]))
                    self.assertTrue(np.all(d.flexvert_xpos < INSIDE_HIGH))
                    self.assertFalse(eq[ex.cloth_eq])
                    self.assertTrue(obs.hand_clear)
            self.assertGreaterEqual(report["summary"]["retention_seconds"], 3.)


if __name__ == "__main__":
    unittest.main()
