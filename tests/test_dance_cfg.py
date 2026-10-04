"""Cfg + MDP invariants for the dance routine: three swaying steps, stand, x2.

These lock the phase conventions, the reward wiring, and the structural
decisions that are easy to undo by accident: the stock `feet_air_time` MUST stay
out (its command gate can never close on a unit-circle phase, so it would keep
paying for a lifted foot during the stand window), and the squats MUST stay out
of the term list (a 25 mm dip on a 0.45 s cycle gave a saturated height Gaussian
with no gradient, and the trunk never left its standing height).
"""

import math
import types

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    DANCE_FEET_LEVEL_SCALE,
    DANCE_FORWARD_M,
    DANCE_PERIOD_S,
    DANCE_START_PHASE_PROB,
    DANCE_SWAY_AMPLITUDE,
    DANCE_SWAY_DEG,
    DANCE_STEP_LIFT_M,
    EPISODE_LENGTH_S,
    N_REPS,
    N_STEPS,
    N_SWAYS,
    WALK_END,
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)

FEET_CFG = SceneEntityCfg("robot", site_ids=[0, 1])


def _cmd(phases):
    phases = torch.as_tensor(phases, dtype=torch.float32)
    return torch.stack(
        [torch.cos(2 * math.pi * phases), torch.sin(2 * math.pi * phases), torch.zeros_like(phases)],
        dim=1,
    )


def _local(t):
    """Command slot for a repetition-local phase (inverse of dance_local_phase)."""
    return _cmd([t / N_REPS])


# --------------------------------------------------------------------------- #
# Phase conventions                                                            #
# --------------------------------------------------------------------------- #
def test_phase_is_recovered_from_the_runtime_command_slot():
    phases = torch.linspace(0.0, 1.0, 101)[:-1]
    assert torch.allclose(microduck_mdp.dance_phase(_cmd(phases)), phases, atol=1e-6)


def test_local_phase_wraps_once_per_repetition():
    phases = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999])
    local = microduck_mdp.dance_local_phase(_cmd(phases), N_REPS)
    assert torch.allclose(local, (phases * N_REPS) % 1.0, atol=1e-6)


def test_repetition_is_walk_then_sway():
    assert 0.0 < WALK_END < 1.0
    rep = DANCE_PERIOD_S / N_REPS
    assert WALK_END * rep == pytest.approx(1.5, abs=1e-9)          # 3 steps at 0.5 s
    assert (1.0 - WALK_END) * rep == pytest.approx(1.5, abs=1e-9)  # 2 sways at 0.75 s
    assert EPISODE_LENGTH_S == pytest.approx(N_REPS * rep)


def test_episode_stays_short():
    """The routine must not grow into a 9 s sit: the previous long version spent
    most of its budget on the walk and failed at the repetition handover."""
    assert EPISODE_LENGTH_S <= 6.5


def test_walk_half_is_specified_by_distance_not_by_step_count():
    """15 cm is the spec; N_STEPS only sets the cadence it is covered at."""
    assert DANCE_FORWARD_M == pytest.approx(0.15)


def test_sway_is_exaggerated_but_physical():
    # ~8° of lean is the minimum that unloads the other foot (the stance foot's
    # inner edge is 21 mm off the centreline with the CoM 148 mm up).
    assert 12.0 <= DANCE_SWAY_DEG <= 20.0
    assert DANCE_SWAY_AMPLITUDE == pytest.approx(math.radians(DANCE_SWAY_DEG))


# --------------------------------------------------------------------------- #
# Phase references                                                             #
# --------------------------------------------------------------------------- #
def _sway_local(frac):
    """Local phase `frac` of the way through the sway half."""
    return WALK_END + (1.0 - WALK_END) * frac


def _sway_ref(local):
    return microduck_mdp.dance_sway_reference(
        torch.tensor([local]), WALK_END, N_SWAYS, DANCE_SWAY_AMPLITUDE
    ).item()


def test_sway_reference_is_level_while_walking():
    """The walk half must not be leaning: running the sway through it is what
    made the whole routine read as a shuffle instead of a walk plus a sway."""
    for local in (0.0, WALK_END * 0.5, WALK_END):
        assert _sway_ref(local) == pytest.approx(0.0, abs=1e-9)


def test_sway_reference_swings_left_and_right_in_the_sway_half():
    # peaks at window fraction (2k + 1) / (4 * N_SWAYS)
    left = _sway_local(1.0 / (4 * N_SWAYS))
    right = _sway_local(3.0 / (4 * N_SWAYS))
    assert _sway_ref(left) == pytest.approx(DANCE_SWAY_AMPLITUDE, abs=1e-6)
    assert _sway_ref(right) == pytest.approx(-DANCE_SWAY_AMPLITUDE, abs=1e-6)
    # starts and ends level, so both hand-overs are clean
    assert _sway_ref(WALK_END) == pytest.approx(0.0, abs=1e-6)
    assert _sway_ref(1.0 - 1e-9) == pytest.approx(0.0, abs=1e-6)


def test_step_schedule_alternates_in_both_halves():
    """The lifts run over the WHOLE routine: they are the steps in the walk half
    and the counterweight to the lean in the sway half."""
    # walk half, one lift per half-cycle at the walk cadence (sampled mid-half to
    # stay clear of the boundaries)
    walk = torch.tensor(
        [0.0, WALK_END / (4 * N_STEPS), 3 * WALK_END / (4 * N_STEPS)]
    )
    s = microduck_mdp.dance_step_schedule(walk, WALK_END, N_STEPS, N_SWAYS)
    # columns are (LEFT should be up, RIGHT should be up)
    assert s[0].tolist() == [0.0, 1.0]
    assert s[1].tolist() == [0.0, 1.0]   # still the right foot's half-cycle
    assert s[2].tolist() == [1.0, 0.0]   # hand-over to the left foot
    # sway half, same alternation at the sway cadence
    def sway(frac):
        return WALK_END + (1.0 - WALK_END) * frac
    s2 = microduck_mdp.dance_step_schedule(
        torch.tensor([sway(0.0), sway(3.0 / (4 * N_SWAYS)), sway(1.0 / N_SWAYS)]),
        WALK_END, N_STEPS, N_SWAYS,
    )
    assert s2[0].tolist() == [0.0, 1.0]
    assert s2[1].tolist() == [1.0, 0.0]
    assert s2[2].tolist() == [0.0, 1.0]


def test_the_sway_lift_is_paired_with_the_lean():
    """Leaning left must coincide with the right foot being up — that pairing is
    what gives the weight somewhere to go, and without it the robot just stands
    (measured: 0.4 deg of lean against a 15 deg reference)."""
    left_peak = _sway_local(1.0 / (4 * N_SWAYS))
    right_peak = _sway_local(3.0 / (4 * N_SWAYS))
    assert _sway_ref(left_peak) > 0  # leaning left ...
    assert _sway_ref(right_peak) < 0
    s = microduck_mdp.dance_step_schedule(
        torch.tensor([left_peak, right_peak]), WALK_END, N_STEPS, N_SWAYS
    )
    assert s[0].tolist() == [0.0, 1.0]  # ... right foot up
    assert s[1].tolist() == [1.0, 0.0]  # ... left foot up


# --------------------------------------------------------------------------- #
# Task terms                                                                   #
# --------------------------------------------------------------------------- #
def _dance_env(local, gravity=None, pos=None, sites=None, air_time=None, heights=None):
    """Minimal stand-in for a ManagerBasedRlEnv carrying one environment."""
    command = _local(local)
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
    local = _sway_local(1.0 / (4 * N_SWAYS))  # a sway peak
    env = _dance_env(local, gravity=_lean(DANCE_SWAY_AMPLITUDE))
    score = microduck_mdp.dance_sway_tracking(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_sways=N_SWAYS,
        amplitude=DANCE_SWAY_AMPLITUDE,
        std=0.10,
    )
    assert score.item() == pytest.approx(1.0, abs=1e-6)


def test_sway_tracking_does_not_pay_full_marks_for_standing_level():
    """The Gaussian is the look, not the teacher: at a peak a level trunk must
    score well below full marks, which is why `dance_sway_l1` exists."""
    local = _sway_local(1.0 / (4 * N_SWAYS))
    env = _dance_env(local)
    score = microduck_mdp.dance_sway_tracking(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_sways=N_SWAYS,
        amplitude=DANCE_SWAY_AMPLITUDE,
        std=0.10,
    )
    assert score.item() < 0.05


def test_sway_l1_charges_the_lean_error_in_radians():
    local = _sway_local(1.0 / (4 * N_SWAYS))
    env = _dance_env(local)
    score = microduck_mdp.dance_sway_l1(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_sways=N_SWAYS,
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
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_sways=N_SWAYS,
        amplitude=DANCE_SWAY_AMPLITUDE,
    ).item() == pytest.approx(0.0, abs=1e-6)
    assert -reference >= -2.0 * DANCE_SWAY_AMPLITUDE


def _step(env):
    return microduck_mdp.dance_step_tracking(
        env,
        command_name="twist",
        sensor_name="feet_ground_contact",
        height_sensor_name="foot_height_scan",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        n_sways=N_SWAYS,
        min_lift=DANCE_STEP_LIFT_M,
    ).item()


def test_step_tracking_needs_the_schedule_the_lift_and_the_clearance():
    # RIGHT scheduled, RIGHT airborne and clear -> paid
    env = _dance_env(
        0.0, air_time=torch.tensor([[0.0, 0.2]]), heights=torch.tensor([[0.0, 0.02]])
    )
    assert _step(env) == pytest.approx(1.0)
    # ... but a scuff under DANCE_STEP_LIFT_M pays nothing
    scuff = _dance_env(
        0.0, air_time=torch.tensor([[0.0, 0.2]]), heights=torch.tensor([[0.0, 0.005]])
    )
    assert _step(scuff) == 0.0
    # ... and the WRONG foot being up is not a step either
    wrong = _dance_env(
        0.0, air_time=torch.tensor([[0.2, 0.0]]), heights=torch.tensor([[0.02, 0.0]])
    )
    assert _step(wrong) == 0.0


def test_step_tracking_also_runs_in_the_sway_half():
    """The sway's lean needs its paired lift, so the step term must not switch
    off after the walk."""
    left_peak = _sway_local(1.0 / (4 * N_SWAYS))  # leaning left, right foot up
    env = _dance_env(
        left_peak, air_time=torch.tensor([[0.0, 0.2]]), heights=torch.tensor([[0.0, 0.02]])
    )
    assert _step(env) == pytest.approx(1.0)


def _progress_env(local):
    env = _dance_env(local)
    env._dance_origin_pos = torch.zeros(1, 3)
    env._dance_origin_yaw = torch.zeros(1)
    env._dance_fwd_max = torch.zeros(1)
    env._dance_fwd_paid = torch.zeros(1)
    return env


def _progress(env):
    return microduck_mdp.dance_forward_progress(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END,
        target_distance=DANCE_FORWARD_M,
    )


def test_forward_progress_pays_only_for_new_ground():
    env = _progress_env(0.0)
    asset = env.scene["robot"]
    assert _progress(env).item() == 0.0  # marching in place pays nothing

    asset.data.root_link_pos_w[0, 0] = 0.01
    stepped = _progress(env)
    # Rate-normalised: 1 cm of new ground at a 0.22 m scale, per control step.
    assert stepped.item() == pytest.approx(0.01 / (env.step_dt * DANCE_FORWARD_M), rel=1e-5)

    # Walking back does not pay again, and the frontier is not un-earned.
    asset.data.root_link_pos_w[0, 0] = 0.0
    assert _progress(env).item() == 0.0


def test_forward_progress_is_capped_at_the_target_distance():
    env = _progress_env(0.0)
    env.scene["robot"].data.root_link_pos_w[0, 0] = 3.0 * DANCE_FORWARD_M
    assert _progress(env).item() > 0.0
    env.scene["robot"].data.root_link_pos_w[0, 0] = 6.0 * DANCE_FORWARD_M
    assert _progress(env).item() == pytest.approx(0.0, abs=1e-9)


def test_forward_progress_is_inactive_during_the_stand_window():
    env = _progress_env(WALK_END + 0.1)
    env.scene["robot"].data.root_link_pos_w[0, 0] = 0.05
    assert _progress(env).item() == 0.0


def test_feet_level_penalty_grows_with_the_fore_aft_stagger():
    sites = torch.zeros(1, 2, 3)
    sites[0, 0, 0] = 0.02  # left foot 2 cm ahead of the right one
    env = _dance_env(WALK_END + 0.1, sites=sites)
    score = microduck_mdp.dance_feet_level_l1(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        feet_cfg=FEET_CFG,
        scale=DANCE_FEET_LEVEL_SCALE,
    )
    # Normalised: one DANCE_FEET_LEVEL_SCALE of stagger is worth a full unit.
    assert score.item() == pytest.approx(-0.02 / DANCE_FEET_LEVEL_SCALE, abs=1e-6)

    sites[0, 0, 0] = 0.0
    assert microduck_mdp.dance_feet_level_l1(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        feet_cfg=FEET_CFG,
        scale=DANCE_FEET_LEVEL_SCALE,
    ).item() == pytest.approx(0.0, abs=1e-6)


def test_feet_are_not_judged_while_walking():
    sites = torch.zeros(1, 2, 3)
    sites[0, 0, 0] = 0.05  # a long stride, which is not a stagger
    env = _dance_env(0.0, sites=sites)
    score = microduck_mdp.dance_feet_level_l1(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        feet_cfg=FEET_CFG,
        scale=DANCE_FEET_LEVEL_SCALE,
    )
    assert score.item() == 0.0


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
    # rest are scattered so the second repetition is practised on its own.
    assert command.randomize_phase is True
    assert command.zero_phase_prob == pytest.approx(DANCE_START_PHASE_PROB)
    assert 0.0 < DANCE_START_PHASE_PROB < 1.0


def test_the_routine_terms_are_registered():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "dance_sway_tracking": 5.0,
        "dance_sway_l1": 0.6,
        "dance_step_tracking": 3.0,
        "dance_forward_progress": 0.0,  # ramped by the forward_progress_weight curriculum
        "dance_heading_l1": 1.0,
        "dance_feet_level_l1": 1.0,
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
    """The in-place bounce was replaced, not patched: none of its terms may come
    back, and neither may the stock air-time term it superseded."""
    cfg = make_microduck_dance_env_cfg()
    for name in (
        "bounce_roll_tracking",
        "bounce_roll_l1",
        "bounce_lift_tracking",
        "bounce_stay_in_place",
        "bounce_stay_in_place_l1",
        "bounce_pitch_balance",
        "bounce_head_hold",
        "air_time",
    ):
        assert name not in cfg.rewards, name
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
    for name in ("dance_sway_l1", "dance_heading_l1", "dance_feet_level_l1"):
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
        "dance_feet_fore_aft_mm": "last",
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
