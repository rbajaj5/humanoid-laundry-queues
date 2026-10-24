import dataclasses
import unittest

from laundry_task.supervisor import (
    Command, Config, Coordinator, Observation, Skill, SkillReceiver,
)


def obs(t=0., sequence=0, **changes):
    values = dict(sequence=sequence, observed_at=t, body_ready=True,
                  door_angle_rad=0., cloth_grasped=False, cloth_inside=False,
                  hand_clear=True)
    values.update(changes)
    return Observation(**values)


class ProtocolTests(unittest.TestCase):
    def test_missing_receipt_retries_same_identity(self):
        c = Coordinator("test")
        p = c.update(0, obs())
        retry = c.update(.8, obs(.8, 1))
        self.assertEqual(p, retry)
        self.assertEqual(c.completed, [])

    def test_receipt_is_not_task_completion(self):
        c = Coordinator("test")
        p = c.update(0, obs())
        c.receive_receipt(p.command_id, .2, "done")
        self.assertEqual(c.pending, p)
        self.assertEqual(c.completed, [])

    def test_expiry_does_not_cancel_execution(self):
        c = Coordinator("test")
        p = c.update(0, obs())
        self.assertIsNone(c.update(3., obs(3., 1)))
        self.assertEqual(c.pending, p)
        c.update(4., obs(4., 2, door_angle_rad=1.1))
        self.assertEqual(c.completed, ["open_door"])

    def test_reject_old_future_and_nan_observations(self):
        for o in (obs(-1), obs(1), obs(float("nan")), obs(door_angle_rad=float("nan"))):
            self.assertIsNone(Coordinator("test").update(0, o))

    def test_observation_order(self):
        c = Coordinator("test")
        p = c.update(1, obs(1, 10))
        self.assertIsNone(c.update(1.1, obs(1.1, 9, door_angle_rad=1.1)))
        self.assertIsNone(c.update(1.1, obs(.95, 11, door_angle_rad=1.1)))
        self.assertEqual(c.pending, p)

    def test_receiver_deduplicates(self):
        c, r = Coordinator("test"), SkillReceiver("test")
        p = c.update(0, obs())
        self.assertEqual(r.dequeue(p, .1, obs(.1, 1)), "EXECUTE")
        self.assertEqual(r.dequeue(p, .2, obs(.2, 2)), "DUPLICATE_NO_EXECUTION")
        self.assertEqual(r.dequeue(dataclasses.replace(p, skill=Skill.GRASP),
                                   .3, obs(.3, 3)), "REJECT_ID_COLLISION")

    def test_dequeue_checks_current_state(self):
        p = Command("test:2", "test", Skill.INSERT, 0, 2, 0)
        r = SkillReceiver("test")
        self.assertEqual(r.dequeue(p, .1, obs(.1, 1, cloth_grasped=True)),
                         "REJECT_PRECONDITION")
        self.assertEqual(r.dequeue(p, 3, obs(3, 2)), "REJECT_EXPIRED_OR_FUTURE")

    def test_old_session_rejected(self):
        p = Coordinator("old").update(0, obs())
        self.assertEqual(SkillReceiver("new").dequeue(p, .1, obs(.1, 1)),
                         "REJECT_SESSION")

    def test_active_skill_blocks_conflicting_command(self):
        r = SkillReceiver("test")
        p = Coordinator("test").update(0, obs())
        r.dequeue(p, .1, obs(.1, 1))
        conflict = dataclasses.replace(p, command_id="test:other")
        self.assertEqual(r.dequeue(conflict, .2, obs(.2, 2)), "HOLD_ACTIVE_SKILL")

    def test_unreliable_body_holds(self):
        self.assertIsNone(Coordinator("test").update(0, obs(body_ready=False)))

    def test_release_requires_cloth_inside(self):
        p = Command("test:3", "test", Skill.RELEASE, 0, 2, 0)
        r = SkillReceiver("test")
        self.assertEqual(r.dequeue(p, .1, obs(.1, 1, door_angle_rad=1.1,
                         cloth_grasped=True)), "REJECT_PRECONDITION")

    def test_complete_ordered_task_without_any_receipts(self):
        c, r = Coordinator("test"), SkillReceiver("test")
        states = [{}, {"door_angle_rad": 1.1},
                  {"door_angle_rad": 1.1, "cloth_grasped": True, "hand_clear": False},
                  {"door_angle_rad": 1.1, "cloth_grasped": True,
                   "cloth_inside": True, "hand_clear": False},
                  {"door_angle_rad": 1.1, "cloth_inside": True, "hand_clear": False},
                  {"door_angle_rad": 1.1, "cloth_inside": True}]
        for i in range(5):
            t = float(i)
            p = c.update(t, obs(t, i * 3, **states[i]))
            self.assertEqual(r.dequeue(p, t + .1, obs(t + .1, i * 3 + 1,
                                                      **states[i])), "EXECUTE")
            final = obs(t + .5, i * 3 + 2, **states[i + 1])
            self.assertTrue(r.finish(p.command_id, t + .5, final))
            c.update(t + .5, final)
        self.assertEqual(c.completed, [s.value for s in Skill])
        self.assertEqual(len(r.seen), 5)

    def test_invalid_thresholds(self):
        for value in (-1, 0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Config(max_observation_age_s=value)

    def test_completion_evidence_must_follow_execution_start(self):
        p = Coordinator("test").update(0, obs())
        r = SkillReceiver("test")
        r.dequeue(p, 1., obs(1., 2))
        self.assertFalse(r.finish(p.command_id, 1.1,
                                  obs(.95, 3, door_angle_rad=1.1)))

    def test_old_receipt_cannot_clear_next_command(self):
        c = Coordinator("test")
        old = c.update(0, obs())
        c.update(.2, obs(.2, 1, door_angle_rad=1.1))
        new = c.update(.3, obs(.3, 2, door_angle_rad=1.1))
        c.receive_receipt(old.command_id, .4, "done")
        self.assertEqual(c.pending, new)

    def test_no_retransmission_before_timeout(self):
        c = Coordinator("test")
        p = c.update(0, obs())
        self.assertIsNone(c.update(.1, obs(.1, 1)))
        self.assertEqual(c.pending, p)


if __name__ == "__main__":
    unittest.main()
