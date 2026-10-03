"""Cfg + MDP invariants for the dance routine (walk 3 steps → stand → 3 squats, ×2).

These tests lock the phase conventions, the reward wiring the routine depends
on, and the one structural decision that is easy to undo by accident: the stock
`feet_air_time` MUST stay out. Its command gate can never close on a unit-circle
phase, so it would keep paying the policy for lifting a foot during the stand
and squat windows.
"""

import math
import types

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    DANCE_FORWARD_M,
    DANCE_PERIOD_S,
    DANCE_SQUAT_DEPTH,
    DANCE_STAND_Z,
    DANCE_SWAY_AMPLITUDE,
    DANCE_SWAY_DEG,
    EPISODE_LENGTH_S,
    N_REPS,
    N_SQUATS,
    N_STEPS,
    SQUAT_START,
    WALK_END,
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)

FEET_CFG = SceneEntityCfg("robot", site_ids=[0, 1])


# --------------------------------------------------------------------------- #
# Phase conventions                                                            #
# --------------------------------------------------------------------------- #
def _cmd(phases):
    phases = torch.as_tensor(phases, dtype=torch.float32)
    return torch.stack(
        [torch.cos(2 * math.pi * phases), torch.sin(2 * math.pi * phases), torch.zeros_like(phases)],
        dim=1,
    )


def _local(t):
    """Command slot for a repetition-local phase ℓ (inverse of dance_local_phase)."""
    return _cmd([t / N_REPS])


def test_phase_is_recovered_from_the_runtime_command_slot():
    phases = torch.linspace(0.0, 1.0, 101)[:-1]
    assert torch.allclose(microduck_mdp.dance_phase(_cmd(phases)), phases, atol=1e-6)


def test_local_phase_wraps_once_per_repetition():
    phases = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999])
    local = microduck_mdp.dance_local_phase(_cmd(phases), N_REPS)
    assert torch.allclose(local, (phases * N_REPS) % 1.0, atol=1e-6)


def test_repetition_split_is_walk_then_stand_then_squat():
    assert 0.0 < WALK_END < SQUAT_START < 1.0
    rep = DANCE_PERIOD_S / N_REPS
    assert WALK_END * rep == pytest.approx(1.5, abs=1e-9)          # 3 steps @ 0.5 s
    assert (SQUAT_START - WALK_END) * rep == pytest.approx(0.5, abs=1e-9)
    assert (1.0 - SQUAT_START) * rep == pytest.approx(1.5, abs=1e-9)  # 3 squats @ 0.5 s


def test_sway_is_exaggerated_but_physical():
    # ~8° of lean is the minimum that unloads the other foot (the stance foot's
    # inner edge is 21 mm off the centreline with the CoM 148 mm up); the walk is
    # asked for more than that, but not for a fall.
    assert 12.0 <= DANCE_SWAY_DEG <= 20.0
    assert DANCE_SWAY_AMPLITUDE == pytest.approx(math.radians(DANCE_SWAY_DEG))


# --------------------------------------------------------------------------- #
# Phase references                                                             #
# --------------------------------------------------------------------------- #
def test_sway_reference_is_level_outside_the_walk():
    for local in (0.0, WALK_END, 0.5, 0.9, 1.0 - 1e-6):
        ref = microduck_mdp.dance_sway_reference(
            torch.tensor([local]), WALK_END, N_STEPS, DANCE_SWAY_AMPLITUDE
        )
        if local >= WALK_END:
            assert ref.item() == 0.0
    # The sway is also zero at both ends of the walk window, so the hand-over to
    # the stand is level.
    for local in (0.0, WALK_END):
        ref = microduck_mdp.dance_sway_reference(
            torch.tensor([local]), WALK_END, N_STEPS, DANCE_SWAY_AMPLITUDE
        )
        assert ref.item() == pytest.approx(0.0, abs=1e-6)


def test_sway_reference_alternates_left_and_right_each_step():
    # Peaks sit mid-lift: window fraction (k + 0.5) / (2 · N_STEPS).
    left = torch.tensor([WALK_END * 0.5 / (2 * N_STEPS)])
    right = torch.tensor([WALK_END * 1.5 / (2 * N_STEPS)])
    assert microduck_mdp.dance_sway_reference(
        left, WALK_END, N_STEPS, DANCE_SWAY_AMPLITUDE
    ).item() == pytest.approx(DANCE_SWAY_AMPLITUDE, abs=1e-6)
    assert microduck_mdp.dance_sway_reference(
        right, WALK_END, N_STEPS, DANCE_SWAY_AMPLITUDE
    ).item() == pytest.approx(-DANCE_SWAY_AMPLITUDE, abs=1e-6)


def test_step_schedule_alternates_and_stops_after_the_walk():
    local = torch.tensor([0.0, WALK_END / 12, WALK_END * 0.25, WALK_END, 0.9])
    schedule = microduck_mdp.dance_step_schedule(local, WALK_END, N_STEPS)
    # columns are (LEFT should be up, RIGHT should be up)
    assert schedule[0].tolist() == [0.0, 1.0]
    assert schedule[1].tolist() == [0.0, 1.0]
    assert schedule[2].tolist() == [1.0, 0.0]
    # Standing and squatting keep both feet planted.
    assert schedule[3].tolist() == [0.0, 0.0]
    assert schedule[4].tolist() == [0.0, 0.0]


def test_squat_reference_is_level_outside_the_squat_window():
    level = torch.tensor([0.0, WALK_END, SQUAT_START - 1e-6])
    ref = microduck_mdp.dance_squat_height_reference(
        level, SQUAT_START, N_SQUATS, DANCE_STAND_Z, DANCE_SQUAT_DEPTH
    )
    assert torch.allclose(ref, torch.full_like(ref, DANCE_STAND_Z), atol=1e-5)


def test_squat_reference_makes_exactly_N_SQUATS_dips_of_the_requested_depth():
    span = 1.0 - SQUAT_START
    bottoms = SQUAT_START + span * (torch.arange(N_SQUATS) + 0.5) / N_SQUATS
    ref = microduck_mdp.dance_squat_height_reference(
        bottoms, SQUAT_START, N_SQUATS, DANCE_STAND_Z, DANCE_SQUAT_DEPTH
    )
    assert torch.allclose(
        ref, torch.full_like(ref, DANCE_STAND_Z - DANCE_SQUAT_DEPTH), atol=1e-6
    )
    # ... and comes back level at the end of the repetition.
    end = microduck_mdp.dance_squat_height_reference(
        torch.tensor([1.0]), SQUAT_START, N_SQUATS, DANCE_STAND_Z, DANCE_SQUAT_DEPTH
    )
    assert end.item() == pytest.approx(DANCE_STAND_Z, abs=1e-6)


# --------------------------------------------------------------------------- #
# Task terms                                                                   #
# --------------------------------------------------------------------------- #
class _Scene(dict):
    """dict for `scene[name]` lookups plus the terrain the height terms read."""

    def __init__(self, **items):
        super().__init__(**items)
        self.terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))


def _dance_env(local, gravity=None, pos=None, sites=None, air_time=None):
    """Minimal stand-in for a ManagerBasedRlEnv carrying one environment."""
    command = _local(local)
    data = types.SimpleNamespace(
        projected_gravity_b=gravity if gravity is not None else torch.zeros(1, 3),
        root_link_pos_w=pos if pos is not None else torch.zeros(1, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        site_pos_w=sites if sites is not None else torch.zeros(1, 2, 3),
    )
    scene = _Scene(robot=types.SimpleNamespace(data=data))
    if air_time is not None:
        scene["feet_ground_contact"] = types.SimpleNamespace(
            data=types.SimpleNamespace(current_air_time=air_time)
        )
    return types.SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        commands={"twist": command},
        command_manager=types.SimpleNamespace(get_command=lambda _n: command),
        scene=scene,
    )


def test_sway_tracking_is_maximal_when_the_lean_matches_the_reference():
    gravity = torch.tensor(
        [[0.0, -math.sin(DANCE_SWAY_AMPLITUDE), -math.cos(DANCE_SWAY_AMPLITUDE)]]
    )
    env = _dance_env(WALK_END * 0.5 / (2 * N_STEPS), gravity=gravity)
    score = microduck_mdp.dance_sway_tracking(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
        std=0.10,
    )
    assert score.item() == pytest.approx(1.0, abs=1e-6)


def test_step_tracking_needs_both_the_schedule_and_the_lift():
    env = _dance_env(0.0, air_time=torch.tensor([[0.0, 0.2]]))  # RIGHT airborne
    score = microduck_mdp.dance_step_tracking(
        env,
        command_name="twist",
        sensor_name="feet_ground_contact",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_steps=N_STEPS,
    )
    assert score.item() == pytest.approx(1.0)


def test_step_tracking_pays_nothing_during_the_stand_window():
    env = _dance_env(SQUAT_START * 0.9, air_time=torch.tensor([[0.2, 0.2]]))
    score = microduck_mdp.dance_step_tracking(
        env,
        command_name="twist",
        sensor_name="feet_ground_contact",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_steps=N_STEPS,
    )
    assert score.item() == 0.0


def _progress_env(local):
    env = _dance_env(local)
    env._dance_origin_pos = torch.zeros(1, 3)
    env._dance_origin_yaw = torch.zeros(1)
    env._dance_fwd_max = torch.zeros(1)
    env._dance_fwd_paid = torch.zeros(1)
    return env


def test_forward_progress_pays_only_for_new_ground():
    env = _progress_env(0.0)
    asset = env.scene["robot"]

    stand = microduck_mdp.dance_forward_progress(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END,
        target_distance=DANCE_FORWARD_M,
    )
    assert stand.item() == 0.0  # marching in place pays nothing

    asset.data.root_link_pos_w[0, 0] = 0.01
    stepped = microduck_mdp.dance_forward_progress(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END,
        target_distance=DANCE_FORWARD_M,
    )
    assert stepped.item() > 0.0

    # Walking back does not pay again, and the frontier is not un-earned.
    asset.data.root_link_pos_w[0, 0] = 0.0
    back = microduck_mdp.dance_forward_progress(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END,
        target_distance=DANCE_FORWARD_M,
    )
    assert back.item() == 0.0


def test_forward_progress_is_inactive_during_the_stand_window():
    env = _progress_env(SQUAT_START * 0.9)
    env.scene["robot"].data.root_link_pos_w[0, 0] = 0.05
    reward = microduck_mdp.dance_forward_progress(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END,
        target_distance=DANCE_FORWARD_M,
    )
    assert reward.item() == 0.0


def test_squat_tracking_is_inactive_during_the_walk():
    env = _dance_env(0.0, pos=torch.tensor([[0.0, 0.0, DANCE_STAND_Z - 0.05]]))
    score = microduck_mdp.dance_squat_tracking(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        squat_start=SQUAT_START,
        n_squats=N_SQUATS,
        stand_z=DANCE_STAND_Z,
        depth=DANCE_SQUAT_DEPTH,
        std=0.015,
    )
    assert score.item() == 0.0


def test_squat_tracking_is_maximal_at_the_dip_bottom():
    span = 1.0 - SQUAT_START
    local = SQUAT_START + span * 0.5 / N_SQUATS
    env = _dance_env(local, pos=torch.tensor([[0.0, 0.0, DANCE_STAND_Z - DANCE_SQUAT_DEPTH]]))
    score = microduck_mdp.dance_squat_tracking(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        squat_start=SQUAT_START,
        n_squats=N_SQUATS,
        stand_z=DANCE_STAND_Z,
        depth=DANCE_SQUAT_DEPTH,
        std=0.015,
    )
    assert score.item() == pytest.approx(1.0, abs=1e-6)


def test_feet_level_penalty_grows_with_the_fore_aft_stagger():
    sites = torch.zeros(1, 2, 3)
    sites[0, 0, 0] = 0.02  # left foot 2 cm ahead of the right one
    env = _dance_env(SQUAT_START * 0.9, sites=sites)
    score = microduck_mdp.dance_feet_level_l1(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END, feet_cfg=FEET_CFG
    )
    assert score.item() == pytest.approx(-0.02, abs=1e-6)

    sites[0, 0, 0] = 0.0
    score = microduck_mdp.dance_feet_level_l1(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END, feet_cfg=FEET_CFG
    )
    assert score.item() == pytest.approx(0.0, abs=1e-6)


def test_feet_are_not_judged_while_walking():
    sites = torch.zeros(1, 2, 3)
    sites[0, 0, 0] = 0.05  # a long stride, which is not a stagger
    env = _dance_env(0.0, sites=sites)
    score = microduck_mdp.dance_feet_level_l1(
        env, command_name="twist", n_reps=N_REPS, walk_end=WALK_END, feet_cfg=FEET_CFG
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
    # Every episode starts at φ=0, like the runtime button press.
    assert command.randomize_phase is False


def test_the_bounce_terms_are_gone():
    """The in-place bounce was replaced, not patched: none of its terms may
    come back, and neither may the stock air-time term it superseded."""
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


def test_the_routine_terms_are_registered():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "dance_sway_tracking": 5.0,
        "dance_sway_l1": 0.6,
        "dance_step_tracking": 3.0,
        "dance_forward_progress": 4.0,
        "dance_heading_l1": 1.0,
        "dance_squat_tracking": 4.0,
        "dance_squat_l1": 0.5,
        "dance_feet_level_l1": 1.0,
        "dance_pitch_balance": 2.0,
        "dance_head_hold": 0.4,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, name
        assert cfg.rewards[name].weight == pytest.approx(weight), name


def test_upright_is_replaced_by_a_pitch_only_balance_term():
    # `upright` penalises roll, which the sway is; it must not come back.
    cfg = make_microduck_dance_env_cfg()
    assert "upright" not in cfg.rewards
    assert cfg.rewards["dance_pitch_balance"].weight > 0.0


def test_velocity_tracking_terms_are_removed():
    # The twist slot carries [cos(2πφ), sin(2πφ), 0]; tracking it as a velocity
    # would reward moving at ~1 m/s.
    cfg = make_microduck_dance_env_cfg()
    assert "track_linear_velocity" not in cfg.rewards
    assert "track_angular_velocity" not in cfg.rewards


def test_self_negating_l1_terms_keep_a_positive_weight():
    cfg = make_microduck_dance_env_cfg()
    for name in (
        "dance_sway_l1",
        "dance_squat_l1",
        "dance_heading_l1",
        "dance_feet_level_l1",
    ):
        assert cfg.rewards[name].weight > 0.0, name


def test_the_unused_command_slots_are_zero_padded():
    """This skill drives only the twist slot: the head/body command slots stay
    in the 61D layout but carry zeros (the velocity template's head/body
    commands are not part of this env, so there is nothing to sample)."""
    cfg = make_microduck_dance_env_cfg()
    assert set(cfg.commands) == {"twist"}
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].func is microduck_mdp.zero_command_padding
        assert cfg.observations[group].terms["body_command"].func is microduck_mdp.zero_command_padding


def test_diagnostics_are_registered_as_episode_metrics():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "dance_sway_amp_deg": "mean",
        "dance_height_err_mm": "mean",
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


def test_every_registered_term_accepts_its_configured_params():
    """A param the function does not take is a TypeError on the first step —
    catch it here instead of in a 5-iteration smoke test."""
    import inspect

    cfg = make_microduck_dance_env_cfg()
    for kind, terms in (("reward", cfg.rewards), ("metric", cfg.metrics)):
        for name, term in terms.items():
            if not inspect.isfunction(term.func):
                continue
            inspect.signature(term.func).bind(None, **term.params)  # env positional


def test_sway_l1_stays_O_of_one():
    """L1 shaping terms must stay O(1): a term that grows with absolute units
    pays the robot to terminate early. Worst case here is a full opposite lean."""
    gravity = torch.tensor(
        [[0.0, math.sin(DANCE_SWAY_AMPLITUDE), -math.cos(DANCE_SWAY_AMPLITUDE)]]
    )
    env = _dance_env(0.0, gravity=gravity)
    worst = microduck_mdp.dance_sway_l1(
        env,
        command_name="twist",
        n_reps=N_REPS,
        walk_end=WALK_END,
        n_steps=N_STEPS,
        amplitude=DANCE_SWAY_AMPLITUDE,
    )
    assert worst.item() > -1.0


def test_episode_is_exactly_one_routine():
    cfg = make_microduck_dance_env_cfg()
    assert EPISODE_LENGTH_S == pytest.approx(DANCE_PERIOD_S)
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S)


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
