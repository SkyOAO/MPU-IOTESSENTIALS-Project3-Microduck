"""Cfg + MDP invariants for the in-place turn dance task.

Two things here are easy to break silently and are therefore locked down:

  • the turn envelope — exactly one revolution, clockwise, with slew-limited
    ramps and a settle segment that HOLDS the finish instead of allowing the
    robot to creep past 2π; and
  • the reward sign convention — the L1 companions are self-negating (they
    return ≤ 0) and must keep a POSITIVE weight, or they double-negate into a
    reward for the violation.
"""

import math
import types

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    DANCE_PERIOD_S,
    DANCE_RAMP_S,
    DANCE_SETTLE_S,
    DANCE_TURN_ANGLE,
    DANCE_TURN_S,
    DANCE_TURN_SIGN,
    DANCE_TURN_SPEED,
    DANCE_TURN_SECONDS_AT_FULL_SPEED,
    EPISODE_LENGTH_S,
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)

ENVELOPE_KWARGS = dict(
    turn_speed=DANCE_TURN_SPEED,
    ramp_s=DANCE_RAMP_S,
    turn_angle=DANCE_TURN_ANGLE,
    turn_sign=DANCE_TURN_SIGN,
)


# --------------------------------------------------------------------------- #
# Turn envelope                                                                #
# --------------------------------------------------------------------------- #
def test_turn_geometry_is_derived_from_the_angle():
    # One revolution takes DANCE_TURN_SECONDS_AT_FULL_SPEED of full-speed
    # rotation, plus ramps. Slowed 8.0 -> 11.0 s after the 2026-10-01 run
    # (the robot fell in every episode at ~250°): completion over speed.
    assert DANCE_TURN_SECONDS_AT_FULL_SPEED >= 10.0
    assert DANCE_TURN_SPEED == pytest.approx(
        2.0 * math.pi / DANCE_TURN_SECONDS_AT_FULL_SPEED
    )
    assert DANCE_TURN_S == pytest.approx(
        DANCE_TURN_ANGLE / DANCE_TURN_SPEED + DANCE_RAMP_S
    )
    assert DANCE_PERIOD_S == pytest.approx(DANCE_TURN_S + DANCE_SETTLE_S)


def test_envelope_turns_exactly_one_revolution_then_holds():
    t = torch.linspace(0.0, EPISODE_LENGTH_S, 2501)
    omega, target = microduck_mdp.dance_turn_envelope(t, **ENVELOPE_KWARGS)

    assert target[0].item() == pytest.approx(0.0, abs=1e-9)
    at_finish = target[t >= DANCE_TURN_S]
    assert torch.allclose(
        at_finish,
        torch.full_like(at_finish, -DANCE_TURN_ANGLE),
        atol=1e-6,
    )
    # The settle segment commands no rotation at all, and the target does not
    # drift past one revolution no matter how long the episode tail runs.
    assert torch.all(omega[t >= DANCE_TURN_S] == 0.0)
    _, tail_target = microduck_mdp.dance_turn_envelope(
        torch.tensor([1.0e4]), **ENVELOPE_KWARGS
    )
    assert tail_target.item() == pytest.approx(-DANCE_TURN_ANGLE, abs=1e-6)


def test_turn_is_clockwise_and_capped_at_the_commanded_speed():
    t = torch.linspace(0.0, DANCE_TURN_S, 2000)
    omega, _ = microduck_mdp.dance_turn_envelope(t, **ENVELOPE_KWARGS)
    # ω_z convention: clockwise is NEGATIVE (x forward, y left, z up).
    assert omega.max().item() == pytest.approx(0.0, abs=1e-9)
    assert omega.min().item() == pytest.approx(-DANCE_TURN_SPEED, abs=1e-3)


def test_turn_is_slew_limited_rather_than_a_step():
    # The old dance commanded an instantaneous ω step; the robot could not
    # follow it. Sample the ramp: consecutive samples must not jump.
    t = torch.linspace(0.0, DANCE_TURN_S, 2000)
    omega, _ = microduck_mdp.dance_turn_envelope(t, **ENVELOPE_KWARGS)
    assert omega.diff().abs().max().item() < 0.02


def test_counter_clockwise_sign_flips_the_whole_envelope():
    t = torch.linspace(0.0, DANCE_TURN_S, 500)
    omega, target = microduck_mdp.dance_turn_envelope(
        t, **{**ENVELOPE_KWARGS, "turn_sign": +1.0}
    )
    assert omega.min().item() == pytest.approx(0.0, abs=1e-9)
    assert target[-1].item() == pytest.approx(DANCE_TURN_ANGLE, abs=1e-3)


# --------------------------------------------------------------------------- #
# Rotation accumulator helpers (no simulator needed)                           #
# --------------------------------------------------------------------------- #
def test_accumulator_state_is_lazily_created_and_zeroed():
    env = types.SimpleNamespace(num_envs=4, device=torch.device("cpu"))
    accum = microduck_mdp._dance_turn_accum_state(env)
    assert accum.shape == (4,)
    assert torch.all(accum == 0.0)
    assert env._dance_turn_last_step == -1
    assert torch.equal(env._dance_turn_last_yaw, accum)


def test_target_angle_falls_back_to_the_episode_clock():
    class _BareTerm:
        """Command term without `turn_target_angle` → exercise the fallback."""

    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda _n: _BareTerm()),
        episode_length_buf=torch.tensor([0, 100, 10_000]),
        step_dt=0.02,
    )
    target = microduck_mdp.dance_turn_target_angle(env)
    assert target[0].item() == pytest.approx(0.0, abs=1e-6)
    assert target[1].item() < 0.0  # clockwise
    assert target[2].item() == pytest.approx(-DANCE_TURN_ANGLE, abs=1e-3)


def test_fallback_defaults_mirror_the_cfg():
    defaults = microduck_mdp._DANCE_TURN_DEFAULTS
    assert defaults["turn_speed"] == pytest.approx(DANCE_TURN_SPEED)
    assert defaults["ramp_s"] == pytest.approx(DANCE_RAMP_S)
    assert defaults["turn_angle"] == pytest.approx(DANCE_TURN_ANGLE)
    assert defaults["settle_s"] == pytest.approx(DANCE_SETTLE_S)
    assert defaults["turn_sign"] == DANCE_TURN_SIGN
    assert DANCE_TURN_SIGN == microduck_mdp.DANCE_TURN_SIGN_CW


# --------------------------------------------------------------------------- #
# Cfg invariants                                                               #
# --------------------------------------------------------------------------- #
def test_cfg_command_matches_the_envelope_constants():
    cfg = make_microduck_dance_env_cfg()
    command = cfg.commands["twist"]
    assert isinstance(command, microduck_mdp.DanceTurnVelocityCommandCfg)
    assert command.turn_angle == pytest.approx(DANCE_TURN_ANGLE)
    assert command.turn_speed == pytest.approx(DANCE_TURN_SPEED)
    assert command.settle_s == pytest.approx(DANCE_SETTLE_S)
    assert command.turn_sign == DANCE_TURN_SIGN
    # Every episode starts at phase 0 (standing), like a button press.
    assert command.randomize_phase is False


def test_rotation_closed_loop_terms_replace_forward_tracking():
    cfg = make_microduck_dance_env_cfg()
    for name in (
        "dance_turn_angle_tracking",
        "dance_turn_angle_l1",
        "dance_stay_in_place",
        "dance_stay_in_place_l1",
    ):
        assert name in cfg.rewards, name
    assert "dance_forward_tracking" not in cfg.rewards

    # Absolute-rotation tracking is the main task term.
    assert cfg.rewards["dance_turn_angle_tracking"].weight == pytest.approx(8.0)
    assert cfg.rewards["dance_turn_angle_tracking"].params["std"] < math.radians(15.0)


def test_task_term_dominates_every_style_term():
    """The 2026-10-01 run failed on reward MASS, not on physics.

    `dance_turn_angle_tracking` was only 5.5 % of the positive reward, so PPO
    could reach mean_reward 86 by shuffling on the spot. Lock the ratio: using
    the weights as the proxy for mass (all these terms live in [0, 1] raw),
    the task term must stay at least twice any single style/regulariser term.
    """
    cfg = make_microduck_dance_env_cfg()
    task = cfg.rewards["dance_turn_angle_tracking"].weight
    for style in (
        "upright",
        "air_time",
        "track_linear_velocity",
        "pose",
        "dance_head_yaw_tracking",
        "dance_head_pitch_tracking",
        "dance_neck_pitch_tracking",
        "dance_head_roll_tracking",
    ):
        assert task >= 2.0 * cfg.rewards[style].weight, style


def test_no_rate_loop_annuity_during_the_settle():
    """`track_angular_velocity` is removed, not re-weighted.

    The settle tail commands ω = 0, so the stock rate term paid a full
    weight×1.0 (5.0) per step for standing perfectly still — a do-nothing
    annuity that out-earned the turn and pulled the policy into the
    "splay the legs and freeze" basin.
    """
    cfg = make_microduck_dance_env_cfg()
    assert "track_angular_velocity" not in cfg.rewards


def test_stability_terms_are_priced_for_the_fall_regime():
    """2026-10-01 second run: every episode ended in a fall.

    `time_out` was 0 for the whole run (mean length 315/575 steps) while the
    logged action rate was ~29.5 summed over 14 joints and the upright raw
    value had fallen 0.80 → 0.68. These three are the levers that were raised —
    all of them re-pricing EXISTING terms, no new shaping.
    """
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["upright"].weight >= 3.0
    assert cfg.rewards["upright"].params["std"] <= 0.20
    # body_ang_vel penalises trunk pitch/roll rate only (yaw is the task).
    assert cfg.rewards["body_ang_vel"].weight <= -0.15

    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] <= -0.08  # smooth from step 0, not -0.05
    assert stages[-1]["weight"] <= -0.15
    assert stages[-1]["step"] <= 1500 * 24  # and ramped in by 1500 iterations


def test_stay_in_place_gaussian_is_wide_enough_to_have_a_gradient():
    # std = 3 cm was numerically dead once the robot had drifted 0.6 m
    # (logged raw value 0.002): a Gaussian only shapes what it can still see.
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["dance_stay_in_place"].params["std"] >= 0.05
    # ... and the L1 companion is what covers the far tail.
    assert cfg.rewards["dance_stay_in_place_l1"].weight >= 1.0


def test_self_negating_l1_terms_keep_positive_weights():
    # mdp.py has two penalty styles: self-negating functions (returning ≤ 0)
    # take a POSITIVE weight. A negative weight here would pay for the
    # violation (AGENTS.md: every Episode_Reward/<penalty> must be ≤ 0).
    cfg = make_microduck_dance_env_cfg()
    for name in ("dance_turn_angle_l1", "dance_stay_in_place_l1"):
        assert cfg.rewards[name].weight > 0.0, name


def test_gait_shaping_is_off_during_the_settle_segment():
    cfg = make_microduck_dance_env_cfg()
    for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip"):
        params = cfg.rewards[name].params
        assert params["command_name"] == "twist"
        # |ω| = turn_speed while turning and exactly 0 while settling, so a
        # threshold inside (0, turn_speed) switches the gait terms off with the
        # turn — no reward for marching in place after the revolution.
        assert 0.0 < params["command_threshold"] < DANCE_TURN_SPEED


def test_settle_segment_selects_the_standing_pose_branch():
    # The pose reward picks std_standing when |command| ≤ walking_threshold;
    # the settle segment commands [0, 0, 0], so the robot is asked to stand.
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["pose"].params["walking_threshold"] > 0.0
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S)
    assert EPISODE_LENGTH_S > DANCE_PERIOD_S
    # ... but the tail must stay short: every extra second of stand-still is
    # free regulariser reward for a collapsed policy.
    assert EPISODE_LENGTH_S - DANCE_PERIOD_S <= 1.5


def test_diagnostics_are_registered_as_episode_metrics():
    """The numbers the next decision depends on must be logged per episode.

    `reduce="last"` = value at the final step of the episode (how far it
    actually rotated, where it ended up); `reduce="mean"` = per-step average
    (tilt and rate curves).
    """
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "dance_rotation_deg": "last",
        "dance_target_deg": "last",
        "dance_rotation_error_deg": "last",
        "dance_drift_x_m": "last",
        "dance_drift_y_m": "last",
        "dance_drift_m": "last",
        "dance_pitch_deg": "mean",
        "dance_roll_deg": "mean",
        "dance_yaw_rate": "mean",
        # Measurement only (no reward weight): is the head moving at all?
        "dance_head_excursion_deg": "mean",
    }
    for name, reduce in expected.items():
        assert name in cfg.metrics, name
        assert cfg.metrics[name].reduce == reduce, name
        assert callable(cfg.metrics[name].func), name


def test_scene_overlay_is_enabled_on_the_turn_command():
    cfg = make_microduck_dance_env_cfg()
    assert cfg.commands["twist"].debug_vis is True
    # The dance command must draw its own overlay (start / drift / heading vs
    # target heading); inheriting the stock velocity arrows would be useless.
    assert (
        microduck_mdp.DanceTurnVelocityCommand._debug_vis_impl
        is not microduck_mdp.UniformVelocityCommand._debug_vis_impl
    )


def test_play_mode_prints_per_episode_diagnostics():
    """Console summary in play (one line per finished episode), silent in train."""
    train_cfg = make_microduck_dance_env_cfg(play=False)
    play_cfg = make_microduck_dance_env_cfg(play=True)
    assert "dance_print_diagnostics" not in train_cfg.events
    assert "dance_print_diagnostics" in play_cfg.events
    assert play_cfg.events["dance_print_diagnostics"].mode == "reset"
    assert callable(play_cfg.events["dance_print_diagnostics"].func)


def test_actor_observation_block_is_still_the_shared_61d_layout():
    cfg = make_microduck_dance_env_cfg()
    terms = cfg.observations["actor"].terms
    command_terms = [
        n for n in terms if n in ("command", "head_command", "body_command")
    ]
    assert command_terms == ["command", "head_command", "body_command"]
    assert terms["command"].params["command_name"] == "twist"
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


def test_runner_cfg_is_unchanged():
    assert MicroduckDanceRlCfg.wandb_project == "mjlab_microduck"
    assert MicroduckDanceRlCfg.experiment_name == "dance"
