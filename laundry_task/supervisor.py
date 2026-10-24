"""A bounded laundry skill protocol, not a robot controller or safety proof.

An acknowledgement is transport evidence. Advancement requires a fresh physical
observation satisfying the current skill's postcondition. The receiver must also
recheck preconditions when dequeuing, not just trust the planner's earlier check.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import math


class Skill(str, Enum):
    OPEN = "open_door"
    GRASP = "grasp_cloth"
    INSERT = "insert_cloth"
    RELEASE = "release_cloth"
    RETRACT = "retract_hand"


ORDER = tuple(Skill)


@dataclass(frozen=True)
class Observation:
    sequence: int
    observed_at: float
    body_ready: bool
    door_angle_rad: float
    cloth_grasped: bool
    cloth_inside: bool
    hand_clear: bool


@dataclass(frozen=True)
class Command:
    command_id: str
    session: str
    skill: Skill
    issued_at: float
    valid_until: float
    observation_sequence: int


@dataclass(frozen=True)
class Config:
    max_observation_age_s: float = .25
    command_ttl_s: float = 2.
    acknowledgement_timeout_s: float = .5
    door_open_threshold_rad: float = .8

    def __post_init__(self):
        for value in asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Thresholds must be finite and positive")


def observation_valid(o: Observation, now: float, c: Config) -> bool:
    return (o.sequence >= 0 and math.isfinite(now) and
            math.isfinite(o.observed_at) and math.isfinite(o.door_angle_rad) and
            0 <= now - o.observed_at <= c.max_observation_age_s)


def precondition(skill: Skill, o: Observation, c: Config) -> bool:
    if not o.body_ready:
        return False
    opened = o.door_angle_rad >= c.door_open_threshold_rad
    if skill == Skill.OPEN:
        return not opened and not o.cloth_grasped and o.hand_clear
    if skill == Skill.GRASP:
        return opened and not o.cloth_grasped and not o.cloth_inside
    if skill == Skill.INSERT:
        return opened and o.cloth_grasped and not o.cloth_inside
    if skill == Skill.RELEASE:
        return opened and o.cloth_grasped and o.cloth_inside
    if skill == Skill.RETRACT:
        return opened and not o.cloth_grasped and o.cloth_inside and not o.hand_clear
    return False


def postcondition(skill: Skill, o: Observation, c: Config) -> bool:
    opened = o.door_angle_rad >= c.door_open_threshold_rad
    if not o.body_ready or not opened:
        return False
    if skill == Skill.OPEN:
        return True
    if skill == Skill.GRASP:
        return o.cloth_grasped
    if skill == Skill.INSERT:
        return o.cloth_grasped and o.cloth_inside
    if skill == Skill.RELEASE:
        return not o.cloth_grasped and o.cloth_inside
    return not o.cloth_grasped and o.cloth_inside and o.hand_clear


class Coordinator:
    """One robot/washer session; no lease reassignment or distributed consensus.

    Retransmission preserves command identity. A timeout never means failure or
    cancellation: an uncertain pending action blocks a conflicting new action.
    """
    def __init__(self, session: str, config: Config | None = None):
        if not session:
            raise ValueError("A unique session identifier is required")
        self.session = session
        self.config = config or Config()
        self.pending: Command | None = None
        self.completed: list[str] = []
        self.last_observation_sequence = -1
        self.last_observed_at = -math.inf
        self.events: list[dict] = []

    def log(self, now: float, event: str, **details):
        self.events.append({"time_s": now, "event": event, **details})

    def observe(self, o: Observation, now: float) -> bool:
        valid = (observation_valid(o, now, self.config) and
                 o.sequence > self.last_observation_sequence and
                 o.observed_at >= self.last_observed_at)
        if not valid:
            self.log(now, "REJECT_OBSERVATION", sequence=o.sequence)
            return False
        self.last_observation_sequence = o.sequence
        self.last_observed_at = o.observed_at
        return True

    def update(self, now: float, o: Observation) -> Command | None:
        if not self.observe(o, now):
            return None
        if not o.body_ready:
            self.log(now, "HOLD_BODY_NOT_READY")
            return None
        if self.pending is not None:
            p = self.pending
            if (o.observed_at > p.issued_at and
                    postcondition(p.skill, o, self.config)):
                self.completed.append(p.skill.value)
                self.pending = None
                self.log(now, "VERIFIED_POSTCONDITION", command_id=p.command_id,
                         skill=p.skill.value, observation_sequence=o.sequence)
                # A separate observation must authorize the following operation.
                return None
            if now > p.valid_until:
                self.log(now, "HOLD_UNCERTAIN_PENDING", command_id=p.command_id)
                return None
            if now - p.issued_at < self.config.acknowledgement_timeout_s:
                self.log(now, "WAIT_PENDING", command_id=p.command_id)
                return None
            self.log(now, "RETRANSMIT_SAME_ID", command_id=p.command_id)
            return p
        if len(self.completed) == len(ORDER):
            self.log(now, "TASK_COMPLETE")
            return None
        skill = ORDER[len(self.completed)]
        if not precondition(skill, o, self.config):
            self.log(now, "HOLD_PRECONDITION", skill=skill.value)
            return None
        p = Command(f"{self.session}:{len(self.completed)}", self.session, skill,
                    now, now + self.config.command_ttl_s, o.sequence)
        self.pending = p
        self.log(now, "ISSUE", **asdict(p))
        return p

    def receive_receipt(self, command_id: str, now: float, status: str):
        matches = self.pending is not None and self.pending.command_id == command_id
        self.log(now, "RECEIPT_ONLY" if matches else "IGNORED_OLD_RECEIPT",
                 command_id=command_id, status=status)
        # Even a remote 'done' message does not substitute for physical evidence.


class SkillReceiver:
    """In-memory idempotency ledger with execution-time precondition checking.

    A hardware implementation must persist this ledger across crashes. Here a
    receiver restart is outside the contract and cannot provide exactly-once
    physical execution. An EXECUTE result authorizes a skill, not its completion.
    """
    def __init__(self, session: str, config: Config | None = None):
        self.session = session
        self.config = config or Config()
        self.seen: dict[str, Command] = {}
        self.active: str | None = None
        self.last_observation_sequence = -1
        self.started_at: dict[str, float] = {}

    def dequeue(self, command: Command, now: float, o: Observation) -> str:
        if command.session != self.session:
            return "REJECT_SESSION"
        if command.command_id in self.seen:
            return ("DUPLICATE_NO_EXECUTION" if self.seen[command.command_id] == command
                    else "REJECT_ID_COLLISION")
        if (not math.isfinite(command.issued_at) or not math.isfinite(command.valid_until)
                or not command.issued_at <= now <= command.valid_until):
            return "REJECT_EXPIRED_OR_FUTURE"
        if (not observation_valid(o, now, self.config) or
                o.observed_at < command.issued_at or
                o.sequence < command.observation_sequence or
                o.sequence < self.last_observation_sequence):
            return "REJECT_OBSERVATION"
        if self.active is not None:
            return "HOLD_ACTIVE_SKILL"
        if not precondition(command.skill, o, self.config):
            return "REJECT_PRECONDITION"
        self.seen[command.command_id] = command
        self.started_at[command.command_id] = now
        self.active = command.command_id
        self.last_observation_sequence = o.sequence
        return "EXECUTE"

    def finish(self, command_id: str, now: float, o: Observation) -> bool:
        command = self.seen.get(command_id)
        if command is None or self.active != command_id:
            return False
        if (not observation_valid(o, now, self.config) or
                o.sequence <= self.last_observation_sequence or
                o.observed_at <= self.started_at[command_id] or
                not postcondition(command.skill, o, self.config)):
            return False
        self.last_observation_sequence = o.sequence
        self.active = None
        return True
