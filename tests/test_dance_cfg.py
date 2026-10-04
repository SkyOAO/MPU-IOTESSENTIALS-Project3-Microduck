"""Cfg + MDP invariants for the dance routine: one swaying walk, then a stand.

These lock the phase conventions, the reward wiring, and the structural
decisions that are easy to undo by accident: the sway MUST stay FUSED into the
walk (one lean per step — the split "walk level, then sway on the spot" version
never produced a lean in any run); the stock `feet_air_time` MUST stay out (its
command gate can never close on a unit-circle phase, so it keeps paying for a
lifted foot through the closing stand — measured, 89 % of the stand was spent on
one foot); and the squats MUST stay out of the term list (a 25 mm dip on a 0.45 s
cycle gave a saturated height Gaussian with no gradient, and the trunk never left
its standing height).
"""

import math
import types

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    DANCE_FORWARD_SPEED,
    DANCE_LEAN_FRAC,
    DANCE_PERIOD_S,
    DANCE_START_PHASE_PROB,
    DANCE_SWAY_AMPLITUDE,
    DANCE_SWAY_DEG,
    EPISODE_LENGTH_S,
    N_STEPS,
    REP_END,
    STAND_S,
    WALK_S,
    WALK_END,
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)



def _cmd(phases):
    phases = torch.as_tensor(phases, dtype=torch.float32)
    return torch.stack(
        [torch.cos(2 * math.pi * phases), torch.sin(2 * math.pi * phases), torch.zeros_like(phases)],
        dim=1,
    )


def _local(t):
    """Command slot for a walk-window phase (inverse of dance_local_phase).

    The walk occupies [0, REP_END) of the period, so a walk-window phase t is at
    period phase t * REP_END.
    """
    return _cmd([t * REP_END])


def _tail(frac):
    """Command slot for `frac` of the way through the closing stand."""
    return _cmd([REP_END + (1.0 - REP_END) * frac])


# --------------------------------------------------------------------------- #
# Phase conventions                                                            #
# --------------------------------------------------------------------------- #
def test_phase_is_recovered_from_the_runtime_command_slot():
    phases = torch.linspace(0.0, 1.0, 101)[:-1]
    assert torch.allclose(microduck_mdp.dance_phase(_cmd(phases)), phases, atol=1e-6)


def test_local_phase_stretches_the_walk_window_over_unit_range():
    uint = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    local = microduck_mdp.dance_local_phase(_cmd(uint * REP_END), REP_END)
    # the round trip through cos/sin/atan2 is not exact at the wrap, so compare
    # on the circle rather than on the raw value
    diff = (local - uint).abs()
    assert (torch.minimum(diff, 1.0 - diff) < 1e-4).all()


def test_the_closing_stand_reads_as_local_phase_one():
    """The tail must close every window at once, or the robot would keep walking
    through what is supposed to be a stand: local phase 1.0 is outside the walk
    window and outside the step schedule."""
    tail = microduck_mdp.dance_local_phase(_tail(0.5), REP_END)
    assert tail.item() == pytest.approx(1.0)
    assert tail.item() >= WALK_END                     # walk window closed
    assert _sway_ref(tail.item()) == 0.0               # sway reference level
    sched = microduck_mdp.dance_step_schedule(tail, WALK_END, N_STEPS)
    assert sched.tolist() == [[0.0, 0.0]]              # both feet planted


def test_the_walk_fills_the_period_and_the_stand_is_its_tail():
    """No split: the walk occupies the whole repeatable part of the period and
    the only thing after it is the closing stand."""
    assert WALK_END == pytest.approx(1.0)
    assert REP_END * DANCE_PERIOD_S == pytest.approx(WALK_S)
    assert (1.0 - REP_END) * DANCE_PERIOD_S == pytest.approx(STAND_S)  # closing stand
    assert EPISODE_LENGTH_S == pytest.approx(DANCE_PERIOD_S)
    assert WALK_S == pytest.approx(N_STEPS * 0.5)          # steps at 0.5 s


def test_episode_stays_short():
    """The routine must not grow into a 9 s sit: the previous long version spent
    most of its budget on the walk and failed at the repetition handover."""
    assert EPISODE_LENGTH_S <= 7.0


def test_the_walk_is_specified_by_speed_not_by_step_count():
    """Speed is the spec; N_STEPS only sets the cadence it is covered at."""
    assert DANCE_FORWARD_SPEED == pytest.approx(0.10)


def test_sway_is_exaggerated_but_physical():
    # ~8° of lean is the minimum that unloads the other foot (the stance foot's
    # inner edge is 21 mm off the centreline with the CoM 148 mm up).
    assert 12.0 <= DANCE_SWAY_DEG <= 20.0
    assert DANCE_SWAY_AMPLITUDE == pytest.approx(math.radians(DANCE_SWAY_DEG))


# --------------------------------------------------------------------------- #
# Phase references                                                             #
# --------------------------------------------------------------------------- #
def _sway_local(frac):
    """Local phase `frac` of the way through ONE step's left-right swing."""
    return WALK_END * frac / N_STEPS


def _sway_ref(local):
    return microduck_mdp.dance_sway_reference(
        torch.tensor([local]), WALK_END, N_STEPS, DANCE_SWAY_AMPLITUDE
    ).item()


def test_sway_reference_completes_whole_swings_over_the_walk():
    """The swing starts and ends level at every step boundary, so the walk hands
    over to the level closing stand without a step in the target."""
    for step in range(N_STEPS + 1):
        assert _sway_ref(WALK_END * step / N_STEPS) == pytest.approx(0.0, abs=1e-6)
    assert _sway_ref(1.0) == pytest.approx(0.0, abs=1e-9)   # the stand tail


def test_sway_reference_swings_once_per_step():
    """One left-right swing per step: the swing and the step rhythm are the same
    clock, which is what makes the lean a step rather than a second motion."""
    left = _sway_local(0.25)
    right = _sway_local(0.75)
    assert _sway_ref(left) == pytest.approx(DANCE_SWAY_AMPLITUDE, abs=1e-6)
    assert _sway_ref(right) == pytest.approx(-DANCE_SWAY_AMPLITUDE, abs=1e-6)
    # the swing starts and ends level, so consecutive steps hand over cleanly
    assert _sway_ref(0.0) == pytest.approx(0.0, abs=1e-6)
    assert _sway_ref(_sway_local(1.0)) == pytest.approx(0.0, abs=1e-6)
    # ... and it completes exactly N_STEPS cycles over the walk, so the walk
    # hands over to the level stand without a step in the target
    assert _sway_ref(WALK_END) == pytest.approx(0.0, abs=1e-6)


def test_step_schedule_alternates_through_the_walk_and_stops_for_the_stand():
    """One lift per half-cycle for the walk, both feet planted in the stand."""
    # sampled mid-half to stay clear of the boundaries
    walk = torch.tensor(
        [0.0, WALK_END / (4 * N_STEPS), 3 * WALK_END / (4 * N_STEPS)]
    )
    s = microduck_mdp.dance_step_schedule(walk, WALK_END, N_STEPS)
    # columns are (LEFT should be up, RIGHT should be up)
    assert s[0].tolist() == [0.0, 1.0]
    assert s[1].tolist() == [0.0, 1.0]   # still the right foot's half-cycle
    assert s[2].tolist() == [1.0, 0.0]   # hand-over to the left foot
    # one more step, then the walk ends and the stand keeps both feet planted
    s2 = microduck_mdp.dance_step_schedule(
        torch.tensor([_sway_local(1.25), 1.0]),
        WALK_END, N_STEPS,
    )
    assert s2[0].tolist() == [0.0, 1.0]
    assert s2[1].tolist() == [0.0, 0.0]


def test_the_sway_lift_is_paired_with_the_lean():
    """Leaning left must coincide with the right foot being up — that pairing is
    what gives the weight somewhere to go, and without it the robot just stands
    (measured: 0.4 deg of lean against a 15 deg reference)."""
    left_peak = _sway_local(0.25)
    right_peak = _sway_local(0.75)
    assert _sway_ref(left_peak) > 0  # leaning left ...
    assert _sway_ref(right_peak) < 0
    s = microduck_mdp.dance_step_schedule(
        torch.tensor([left_peak, right_peak]), WALK_END, N_STEPS
    )
    assert s[0].tolist() == [0.0, 1.0]  # ... right foot up
    assert s[1].tolist() == [1.0, 0.0]  # ... left foot up


# --------------------------------------------------------------------------- #
# Task terms                                                                   #
# --------------------------------------------------------------------------- #
def _dance_env(
    local, gravity=None, pos=None, sites=None, air_time=None, heights=None, command=None
):
    """Minimal stand-in for a ManagerBasedRlEnv carrying one environment."""
    command = _local(local) if command is None else command
    data = types.SimpleNamespace(
        # Default to the upright reading (0, 0, -1): an all-zero gravity vector
        # makes atan2(-0, -0) return -pi and quietly fakes a fallen trunk.
        projected_gravity_b=(
            gravity if gravity is not None else torch.tensor([[0.0, 0.0, -1.0]])
        ),
        root_link_pos_w=pos if pos is not None else torch.zeros(1, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        site_pos_w=sites if sites is not None else torch.zeros(1, 2, 3),
    )
    scene = _Scene(robot=types.SimpleNamespace(data=data))
    if air_time is not None:
        scene["feet_ground_contact"] = types.SimpleNamespace(
            data=types.SimpleNamespace(current_air_time=air_time)
        )
    if heights is not None:
        scene["foot_height_scan"] = types.SimpleNamespace(
            data=types.SimpleNamespace(heights=heights)
        )
    return types.SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        commands={"twist": command},
        command_manager=types.SimpleNamespace(get_command=lambda _n: command),
        scene=scene,
    )


class _Scene(dict):
    """dict for `scene[name]` lookups plus the terrain the metrics read."""

    def __init__(self, **items):
        super().__init__(**items)
        self.terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))


def _lean(a_rad):
    """Gravity vector for a trunk rolled left by `a_rad`."""
    return torch.tensor([[0.0, -math.sin(a_rad), -math.cos(a_rad)]])


def test_sway_tracking_is_maximal_when_the_lean_matches_the_reference():
    local = _sway_local(0.25)  # a swing peak
    env = _dance_env(local, gravity=_lean(DANCE_SWAY_AMPLITUDE))
    score = microduck_mdp.dance_sway_tracking(
        env,
        command_name="twist",
        rep_end=REP_END,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
        std=0.10,
    )
    assert score.item() == pytest.approx(1.0, abs=1e-6)


def test_sway_tracking_does_not_pay_full_marks_for_standing_level():
    """The Gaussian is the look, not the teacher: at a peak a level trunk must
    score well below full marks, which is why `dance_sway_l1` exists."""
    local = _sway_local(0.25)
    env = _dance_env(local)
    score = microduck_mdp.dance_sway_tracking(
        env,
        command_name="twist",
        rep_end=REP_END,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
        std=0.10,
    )
    assert score.item() < 0.05


def test_sway_tracking_pays_nothing_in_the_closing_stand():
    """The look is paid inside the walk only, so a level trunk cannot farm the
    closing stand — that out-of-window subsidy is what a 400-iteration run took
    instead of leaning (sway metric stuck at 0.2°)."""

    def _score(env):
        return microduck_mdp.dance_sway_tracking(
            env,
            command_name="twist",
                rep_end=REP_END,
            walk_end=WALK_END,
            n_steps=N_STEPS,
            amplitude=DANCE_SWAY_AMPLITUDE,
            std=0.10,
        ).item()

    # A level trunk at a swing peak scores the Gaussian minimum, never a subsidy.
    assert _score(_dance_env(_sway_local(0.25))) < 0.05
    # The stand is scored by nothing at all.
    assert _score(_dance_env(0.0, command=_tail(0.5))) == pytest.approx(0.0)


def test_sway_l1_charges_the_lean_error_in_radians():
    local = _sway_local(0.25)
    env = _dance_env(local)
    score = microduck_mdp.dance_sway_l1(
        env,
        command_name="twist",
        rep_end=REP_END,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
    )
    reference = _sway_ref(local)
    assert score.item() == pytest.approx(-reference)

    # Matching the reference costs nothing, and the term is bounded by twice the
    # amplitude (a full opposite lean), so it cannot pay the robot to end early.
    matching = _dance_env(local, gravity=_lean(reference))
    assert microduck_mdp.dance_sway_l1(
        matching,
        command_name="twist",
        rep_end=REP_END,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
    ).item() == pytest.approx(0.0, abs=1e-6)
    assert -reference >= -2.0 * DANCE_SWAY_AMPLITUDE


def _step(env):
    return microduck_mdp.dance_step_tracking(
        env,
        command_name="twist",
        sensor_name="feet_ground_contact",
        rep_end=REP_END,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
        lean_frac=DANCE_LEAN_FRAC,
    ).item()


def test_step_tracking_needs_the_schedule_and_the_lift():
    # At a swing zero-crossing the reference is 0, so the lean test is vacuous
    # there and a scheduled airborne foot is simply a step.
    env = _dance_env(0.0, air_time=torch.tensor([[0.0, 0.2]]))
    assert _step(env) == pytest.approx(1.0)
    # ... and the WRONG foot being up is not a step either
    wrong = _dance_env(0.0, air_time=torch.tensor([[0.2, 0.0]]))
    assert _step(wrong) == 0.0


def test_the_lift_requires_the_lean_sign_not_a_formed_lean():
    """The lift is the ladder the gait is learned on, so it must NOT require a
    formed lean: a foot is only unloaded by leaning over the other one, and a gate
    that demands both locks them together. Measured at lean_frac 0.3, the trunk
    never rolled at all for 550 iterations. What it does require is the correct
    SIGN, so a level shuffle leaning the wrong way is still not paid."""
    left_peak = _sway_local(0.25)  # the reference leans left here, right foot up
    air = torch.tensor([[0.0, 0.2]])
    # leaning the full reference: paid
    leaning = _dance_env(left_peak, gravity=_lean(DANCE_SWAY_AMPLITUDE), air_time=air)
    assert _step(leaning) == pytest.approx(1.0)
    # leaning only a token amount: still paid — this is the ladder
    token = _dance_env(left_peak, gravity=_lean(0.2 * DANCE_SWAY_AMPLITUDE), air_time=air)
    assert _step(token) == pytest.approx(1.0)
    # level trunk: paid, because `roll * ref >= 0` holds at roll = 0
    level = _dance_env(left_peak, air_time=air)
    assert _step(level) == pytest.approx(1.0)
    # leaning the WRONG way: not paid
    wrong_way = _dance_env(left_peak, gravity=_lean(-DANCE_SWAY_AMPLITUDE), air_time=air)
    assert _step(wrong_way) == 0.0
    assert DANCE_LEAN_FRAC == pytest.approx(0.0)


def _progress_env(local):
    env = _dance_env(local)
    env._dance_origin_pos = torch.zeros(1, 3)
    env._dance_origin_yaw = torch.zeros(1)
    env._dance_fwd_max = torch.zeros(1)
    env._dance_fwd_paid = torch.zeros(1)
    return env


def _progress(env):
    return microduck_mdp.dance_forward_progress(
        env, command_name="twist", rep_end=REP_END,
        walk_end=WALK_END, setpoint_speed=DANCE_FORWARD_SPEED,
    )


def test_forward_progress_pays_only_for_new_ground():
    env = _progress_env(0.0)
    asset = env.scene["robot"]
    assert _progress(env).item() == 0.0  # marching in place pays nothing

    asset.data.root_link_pos_w[0, 0] = 0.01
    stepped = _progress(env)
    # Saturating rate: 1 cm in one control step is 0.5 m/s, well past the
    # setpoint, so the step pays the ceiling of 1.0 and the excess is forfeited.
    assert stepped.item() == pytest.approx(1.0)

    # Walking back does not pay again, and the frontier is not un-earned.
    asset.data.root_link_pos_w[0, 0] = 0.0
    assert _progress(env).item() == 0.0


def test_forward_progress_saturates_at_the_setpoint_speed():
    """Racing past the setpoint forfeits the excess, so the policy has nothing to
    gain from running: a step at 0.5 m/s pays exactly what a step at 0.1 m/s does,
    and a step at half the setpoint pays half."""
    env = _progress_env(0.0)
    asset = env.scene["robot"]
    at_setpoint = DANCE_FORWARD_SPEED * env.step_dt

    asset.data.root_link_pos_w[0, 0] = at_setpoint
    assert _progress(env).item() == pytest.approx(1.0)

    # 5x the setpoint in one step still pays 1.0, no more.
    asset.data.root_link_pos_w[0, 0] = at_setpoint + 5.0 * at_setpoint
    assert _progress(env).item() == pytest.approx(1.0)

    # Half the setpoint pays half.
    env = _progress_env(0.0)
    env.scene["robot"].data.root_link_pos_w[0, 0] = 0.5 * at_setpoint
    assert _progress(env).item() == pytest.approx(0.5)


def test_forward_progress_is_inactive_during_the_closing_stand():
    env = _progress_env(1.0)  # local phase in the stand tail
    env.scene["robot"].data.root_link_pos_w[0, 0] = 0.05
    assert _progress(env).item() == 0.0


# --------------------------------------------------------------------------- #
# Cfg invariants                                                               #
# --------------------------------------------------------------------------- #
def test_cfg_reuses_the_shared_phase_command():
    cfg = make_microduck_dance_env_cfg()
    command = cfg.commands["twist"]
    # No new command class: GroundPickPhaseCommand already emits [cos, sin, 0].
    assert isinstance(command, microduck_mdp.GroundPickPhaseCommandCfg)
    assert command.period == DANCE_PERIOD_S
    # Reverse curriculum: a slice keeps the real deployment start (phase 0), the
    # rest are scattered so the last steps get on-policy data of their own.
    assert command.randomize_phase is True
    assert command.zero_phase_prob == pytest.approx(DANCE_START_PHASE_PROB)
    # This is the fraction that starts at phase 0, the deployment hand-over, so
    # it must stay HIGH: lowering it means MORE mid-walk spawns, which turn an
    # episode into recovery practice and pollute the yaw metric.
    assert 0.5 <= DANCE_START_PHASE_PROB < 1.0


def test_the_routine_terms_are_registered():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "dance_sway_tracking": 5.0,
        "dance_sway_l1": 2.0,
        "dance_step_tracking": 3.0,
        "dance_forward_progress": 0.0,  # ramped by the forward_progress_weight curriculum
        "dance_heading_l1": 3.0,
        "dance_pitch_balance": 4.0,
        "dance_head_hold": 0.4,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, name
        assert cfg.rewards[name].weight == pytest.approx(weight), name
    assert len([n for n in cfg.rewards if n.startswith("dance_")]) == len(expected)


def test_the_squat_is_gone():
    """The squat was dropped on 2026-10-04: a 25 mm dip on a 0.45 s cycle put the
    height Gaussian ~20 mm away from the target where it is already saturated, so
    the trunk never left its standing height in any run. Neither the reward nor
    the reference may come back by accident."""
    cfg = make_microduck_dance_env_cfg()
    for name in ("dance_squat_tracking", "dance_squat_l1", "dance_squat_dip_progress"):
        assert name not in cfg.rewards, name
        assert name not in cfg.metrics, name
    assert not any("squat" in name for name in cfg.metrics)
    for name in (
        "dance_squat_tracking",
        "dance_squat_l1",
        "dance_squat_dip_progress",
        "dance_squat_height_reference",
        "dance_metric_height_err_mm",
    ):
        assert not hasattr(microduck_mdp, name), name


def test_the_bounce_terms_are_gone():
    """The in-place bounce was replaced, not patched: none of its task terms may
    come back, and neither may the stock `air_time`, whose always-open command
    gate would pay for lifting a foot straight through the closing stand."""
    cfg = make_microduck_dance_env_cfg()
    for name in (
        "bounce_roll_tracking",
        "bounce_roll_l1",
        "bounce_lift_tracking",
        "bounce_stay_in_place",
        "bounce_stay_in_place_l1",
        "bounce_pitch_balance",
        "bounce_head_hold",
    ):
        assert name not in cfg.rewards, name
    assert "air_time" not in cfg.rewards
    for name in (
        "bounce_phase",
        "bounce_roll_reference",
        "bounce_roll_tracking",
        "bounce_lift_tracking",
        "bounce_lift_schedule",
        "bounce_stay_in_place",
        "bounce_pitch_balance",
        "bounce_head_hold",
        "bounce_reset_origin",
        "bounce_metric_sway_amp_deg",
        "bounce_metric_drift_m",
    ):
        assert not hasattr(microduck_mdp, name), name


def test_upright_is_replaced_by_a_pitch_only_balance_term():
    # `upright` penalises roll, which the sway is; it must not come back.
    cfg = make_microduck_dance_env_cfg()
    assert "upright" not in cfg.rewards
    assert cfg.rewards["dance_pitch_balance"].weight > 0.0


def test_velocity_tracking_terms_are_removed():
    # The twist slot carries [cos, sin, 0] over the unit circle; tracking it as a
    # velocity would reward moving at ~1 m/s.
    cfg = make_microduck_dance_env_cfg()
    assert "track_linear_velocity" not in cfg.rewards
    assert "track_angular_velocity" not in cfg.rewards


def test_self_negating_terms_keep_a_positive_weight():
    cfg = make_microduck_dance_env_cfg()
    for name in ("dance_sway_l1", "dance_heading_l1"):
        assert cfg.rewards[name].weight > 0.0, name


def test_forward_is_the_only_staged_term():
    """Forward translation is what pushes the robot past what it can balance
    before it has learned the steps, so it starts at 0 and ramps in."""
    cfg = make_microduck_dance_env_cfg()
    staged = [n for n, t in cfg.curriculum.items() if getattr(t, "params", {}).get("reward_name")]
    # action_rate_weight is the shared ramp every microduck env carries.
    assert staged == ["action_rate_weight", "forward_progress_weight"]
    stages = cfg.curriculum["forward_progress_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert stages[0]["step"] == 0 and weights[0] == 0.0
    assert weights == sorted(weights) and weights[-1] > weights[0]


def test_diagnostics_are_registered_as_episode_metrics():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "dance_sway_amp_deg": "mean",
        "dance_forward_m": "last",
        "dance_yaw_drift_deg": "last",
    }
    for name, reduce in expected.items():
        assert name in cfg.metrics, name
        assert cfg.metrics[name].reduce == reduce, name
        assert callable(cfg.metrics[name].func), name


def test_every_registered_term_takes_env_as_its_first_parameter():
    """mjlab calls reward/metric functions as ``func(env, **params)``.

    Registering a pure helper by mistake binds the env to whatever its first
    argument is and only explodes at run time. Signature binding alone cannot
    catch that; the parameter NAME can.
    """
    import inspect

    cfg = make_microduck_dance_env_cfg()
    for kind, terms in (("reward", cfg.rewards), ("metric", cfg.metrics)):
        for name, term in terms.items():
            if not inspect.isfunction(term.func):
                continue  # class-based stock terms are instantiated with (cfg, env)
            first = next(iter(inspect.signature(term.func).parameters))
            assert first == "env", f"{kind} '{name}' must take env first, got {first!r}"
            # A param the function does not take is a TypeError on the first step.
            inspect.signature(term.func).bind(None, **term.params)


def test_actor_observation_block_is_still_the_shared_61d_layout():
    cfg = make_microduck_dance_env_cfg()
    terms = cfg.observations["actor"].terms
    command_terms = [n for n in terms if n in ("command", "head_command", "body_command")]
    assert command_terms == ["command", "head_command", "body_command"]
    assert terms["command"].params["command_name"] == "twist"
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


def test_runner_cfg_is_unchanged():
    assert MicroduckDanceRlCfg.wandb_project == "mjlab_microduck"
    assert MicroduckDanceRlCfg.experiment_name == "dance"
