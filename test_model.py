"""Focused mechanism, model-integrity and deterministic replay tests."""
import csv
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import mujoco
from simulate import ROOT, GROUPS, PacketQueue, build_model, reset, make_path, inputs, simulate


class MechanismTests(unittest.TestCase):
    def test_waits_for_all_groups(self):
        delay = np.array([[2, 0, 0, 0]]*4)
        commands = np.array([np.ones(29)*(i+1) for i in range(4)])
        q = PacketQueue(np.zeros(29), delay, "coherent")
        self.assertTrue(np.all(q.tick(0, commands)[0] == 0))
        self.assertTrue(np.all(q.tick(1, commands)[0] == 0))
        self.assertTrue(np.all(q.tick(2, commands)[0] == 1))
        self.assertTrue(np.all(q.version == 0))

    def test_obsolete_packets_do_not_reverse_time(self):
        delay = np.array([[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        commands = np.array([np.ones(29)*i for i in range(4)])
        for arm in ("independent", "coherent"):
            q = PacketQueue(np.zeros(29), delay, arm)
            for i in range(4):
                q.tick(i, commands)
            self.assertTrue(np.all(q.applied == 3))
            self.assertGreaterEqual(q.stale, 1)

    def test_independent_exposes_skew(self):
        q = PacketQueue(np.zeros(29), np.array([[5, 0, 0, 0]]), "independent")
        applied, _ = q.tick(0, np.ones((1, 29)))
        self.assertTrue(np.all(applied[:12] == 0))
        self.assertTrue(np.all(applied[12:] == 1))

    def test_no_delay_equivalence(self):
        commands = np.random.default_rng(1).normal(size=(20, 29))
        queues = [PacketQueue(np.zeros(29), np.zeros((20, 4), int), arm) for arm in ("independent", "coherent")]
        for i in range(20):
            np.testing.assert_array_equal(queues[0].tick(i, commands)[0], queues[1].tick(i, commands)[0])


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/"config.json").read_text())
        cls.model = build_model(cls.config)
        cls.poses, cls.target, cls.residual = make_path(cls.model)

    def test_real_mass_inertia_and_free_base(self):
        m = self.model
        self.assertEqual(m.nu, 29)
        self.assertEqual(m.nv, 35)
        self.assertEqual(m.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertAlmostEqual(m.body_mass.sum(), 33.341142, places=6)
        inertial = m.body_mass > 0
        self.assertTrue(np.all(m.body_inertia[inertial] > 0))
        d = reset(m)
        center = (m.body_mass[:, None]*d.xipos).sum(axis=0)/m.body_mass.sum()
        np.testing.assert_allclose(center, d.subtree_com[1], atol=1e-12)
        self.assertEqual(m.neq, 0, "No hidden welds or supports may hold up the robot")

    def test_model_hashes(self):
        manifest = json.loads((ROOT/"output/model_manifest.json").read_text())
        for item in manifest["files"]:
            path = ROOT/"models/unitree_g1"/item["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])

    def test_saved_pairing_and_aggregates(self):
        source = ROOT / "output/episode_results.json"
        if not source.exists():
            self.skipTest("Run the complete experiment first")
        summary = json.loads(source.read_text())
        results = summary["results"]
        self.assertEqual(len(results), len(self.config["profiles"])*len(self.config["seeds"])*2)
        for profile in self.config["profiles"]:
            for seed in self.config["seeds"]:
                pair = [r for r in results if r["profile"] == profile and r["seed"] == seed]
                for key in ("command_sha256", "force_sha256", "delay_sha256"):
                    self.assertEqual(pair[0][key], pair[1][key])
                if profile in ("zero", "uniform_200ms"):
                    for key in ("max_base_tilt_deg", "min_base_height_m", "mean_joint_tracking_rmse_rad"):
                        self.assertEqual(pair[0][key], pair[1][key])
        for g in summary["aggregate"]:
            sample = [r for r in results if r["profile"] == g["profile"] and r["arm"] == g["arm"]]
            self.assertEqual(g["falls"], sum(r["fell"] for r in sample))
            self.assertEqual(g["reach_successes"], sum(r["reach_success"] for r in sample))
            self.assertAlmostEqual(g["mean_peak_tilt_deg"], np.mean([r["max_base_tilt_deg"] for r in sample]))

    def test_nominal_path_within_joint_limits(self):
        m = self.model
        for i in range(1, m.njnt):
            value = self.poses[:, m.jnt_qposadr[i]]
            self.assertGreaterEqual(value.min(), m.jnt_range[i, 0])
            self.assertLessEqual(value.max(), m.jnt_range[i, 1])
        self.assertLess(self.residual, 1e-7)

    def test_matched_inputs_and_repeatable_noise(self):
        for profile in self.config["profiles"]:
            a = inputs(self.model, self.poses, self.config, 24100, profile)
            b = inputs(self.model, self.poses, self.config, 24100, profile)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        a = inputs(self.model, self.poses, self.config, 24100, "zero")
        b = inputs(self.model, self.poses, self.config, 24100, "skew_600ms")
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_retained_replay(self):
        source = ROOT / "output/episode_results.json"
        if not source.exists():
            self.skipTest("Run the complete experiment first")
        results = json.loads(source.read_text())["results"]
        for arm in self.config["arms"]:
            found = next(r for r in results if r["seed"] == self.config["movie_seed"] and
                         r["profile"] == self.config["movie_profile"] and r["arm"] == arm)
            regenerated, rows, qpos = simulate(self.model, self.poses, self.target, self.config,
                                             self.config["movie_seed"], self.config["movie_profile"], arm, True)
            self.assertEqual(found, regenerated)
            old = np.load(ROOT/f"output/replay_{arm}.npz")["qpos"]
            np.testing.assert_array_equal(old, qpos)
            for row in rows:
                for name in ("legs", "waist", "left_arm", "right_arm"):
                    self.assertLessEqual(row[f"{name}_command_version"], row["issued_version"])
                if arm == "coherent":
                    self.assertEqual(row["queue_skew_ms"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
