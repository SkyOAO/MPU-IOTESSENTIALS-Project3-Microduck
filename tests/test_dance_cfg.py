"""Cfg + MDP invariants for the in-place bounce (dance) task.

The move is small on purpose: one phase-referenced sway term, one
phase-referenced step term, and "stay on the spot / do not tip over". These
tests lock the phase conventions (which foot, which way the body leans) and the
two structural decisions that are easy to undo by accident:

  • the stock `upright` term MUST stay out (it penalises roll, and roll is the
    move), and
  • the velocity-tracking terms MUST stay out (the twist slot carries a phase
    pair, not m/s, so they would be tracking a unit circle as a velocity).
"""

import math

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    EPISODE_LENGTH_S,
    BOUNCE_PERIOD_S,
    BOUNCE_SWAY_AMPLITUDE,
    BOUNCE_SWAY_DEG,
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)


# --------------------------------------------------------------------------- #
# Phase conventions                                                            #
# --------------------------------------------------------------------------- #
def test_phase_is_recovered_from_the_runtime_command_slot():
    phases = torch.linspace(0.0, 1.0, 101)[:-1]
    command = torch.stack(
        [
            torch.cos(2.0 * math.pi * phases),
            torch.sin(2.0 * math.pi * phases),
            torch.zeros_like(phases),
        ],
        dim=1,
    )
    assert torch.allclose(microduck_mdp.bounce_phase(command), phases, atol=1e-6)


def test_sway_is_left_first_then_right():
    # +roll = leaning LEFT. The reference peaks in the MIDDLE of each stance
    # half (φ=0.25 while the right foot is up) and is 0 at both hand-overs.
    phase = torch.tensor([0.0, 0.25, 0.5, 0.75])
    reference = microduck_mdp.bounce_roll_reference(phase, BOUNCE_SWAY_AMPLITUDE)
    assert reference.tolist() == pytest.approx(
        [0.0, BOUNCE_SWAY_AMPLITUDE, 0.0, -BOUNCE_SWAY_AMPLITUDE], abs=1e-6
    )


def test_lift_schedule_alternates_by_half_cycle():
    # First half: RIGHT is the foot that should be off the ground; second half:
    # LEFT. Nothing is said about the other foot — this task is a bounce, and
    # both feet airborne together is intended.
    phase = torch.tensor([0.0, 0.25, 0.5, 0.75])
    schedule = microduck_mdp.bounce_lift_schedule(phase)
    # columns are (LEFT should be up, RIGHT should be up)
    assert schedule.tolist() == [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]


def test_centre_gate_measures_net_drift_not_the_sway():
    # 1 where the sway crosses zero (robot should be centred), 0 at the sway
    # peaks (the trunk is legitimately 3-5 cm to the side).
    phase = torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert microduck_mdp.bounce_centre_gate(phase).tolist() == pytest.approx(
        [1.0, 0.0, 1.0, 0.0], abs=1e-6
    )


def test_lift_term_does_not_require_the_other_foot_planted():
    """The bounce must stay a bounce: only the scheduled foot is graded.

    Regression guard for the requirement "就是要类似原地弹跳这种的": if this term
    ever starts rewarding the other foot being DOWN, the both-feet-airborne
    bounce disappears.
    """
    import types

    command = torch.tensor([[1.0, 0.0, 0.0]])  # φ = 0 → RIGHT should be up

    class _Cmd:
        def get_command(self, _name):
            return command

    class _Sensor:
        class data:
            # BOTH feet off the ground — should still score 1.0.
            current_air_time = torch.tensor([[0.20, 0.20]])

    env = types.SimpleNamespace(
        command_manager=_Cmd(), scene={"feet_ground_contact": _Sensor()}
    )
    score = microduck_mdp.bounce_lift_tracking(
        env, command_name="twist", sensor_name="feet_ground_contact"
    )
    assert score.item() == pytest.approx(1.0)


def test_sway_amplitude_is_large_enough_to_unload_a_foot():
    # Geometry (robot_walk.xml, HOME): the stance foot's inner edge is 21.2 mm
    # from the centreline and the CoM sits 148 mm up, so ~8° of lean is the
    # minimum that can lift the other foot and ~15° centres the CoM over the
    # stance foot. "Big amplitude" has to mean bigger than the minimum.
    assert 8.0 <= BOUNCE_SWAY_DEG <= 20.0
    assert BOUNCE_SWAY_AMPLITUDE == pytest.approx(math.radians(BOUNCE_SWAY_DEG))


# --------------------------------------------------------------------------- #
# Cfg invariants                                                               #
# --------------------------------------------------------------------------- #
def test_cfg_reuses_the_shared_phase_command():
    cfg = make_microduck_dance_env_cfg()
    command = cfg.commands["twist"]
    # No new command class: GroundPickPhaseCommand already emits [cos, sin, 0].
    assert isinstance(command, microduck_mdp.GroundPickPhaseCommandCfg)
    assert command.period == BOUNCE_PERIOD_S
    # Every episode starts at φ=0 (standing hand-over), like the button press.
    assert command.randomize_phase is False


def test_the_move_is_exactly_two_task_terms():
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["bounce_roll_tracking"].weight == pytest.approx(5.0)
    assert cfg.rewards["bounce_roll_tracking"].params["amplitude"] == pytest.approx(
        BOUNCE_SWAY_AMPLITUDE
    )
    assert cfg.rewards["bounce_lift_tracking"].weight == pytest.approx(3.0)
    # The sway term must dominate every non-task term.
    task = cfg.rewards["bounce_roll_tracking"].weight
    for other in (
        "bounce_lift_tracking",
        "bounce_pitch_balance",
        "bounce_stay_in_place",
        "bounce_heading_l1",
        "bounce_heading_hold",
        "air_time",
        "pose",
        "bounce_head_hold",
    ):
        assert task >= 1.5 * cfg.rewards[other].weight, other


def test_upright_is_replaced_by_a_pitch_only_balance_term():
    # `upright` penalises roll, which is the move; it must not come back.
    cfg = make_microduck_dance_env_cfg()
    assert "upright" not in cfg.rewards
    assert cfg.rewards["bounce_pitch_balance"].weight > 0.0


def test_velocity_tracking_terms_are_removed():
    # The twist slot carries [cos(2πφ), sin(2πφ), 0]; tracking it as a velocity
    # would reward moving at ~1 m/s.
    cfg = make_microduck_dance_env_cfg()
    assert "track_linear_velocity" not in cfg.rewards
    assert "track_angular_velocity" not in cfg.rewards


def test_self_negating_l1_term_keeps_a_positive_weight():
    # `bounce_roll_l1` returns ≤ 0 → POSITIVE weight (a negative weight would
    # double-negate into paying for the violation).
    cfg = make_microduck_dance_env_cfg()
    for name in ("bounce_roll_l1", "bounce_stay_in_place_l1", "bounce_heading_l1"):
        assert cfg.rewards[name].weight > 0.0, name


def test_turning_is_priced():
    """The first run logged heading_hold at 0.3 % of the positive reward mass
    with a flat Gaussian tail, and the robot ratcheted ~19° around on average.
    The Gaussian must be meaningful AND have an L1 companion that keeps a
    gradient however far the robot has already turned."""
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["bounce_heading_hold"].weight >= 1.0
    assert cfg.rewards["bounce_heading_l1"].weight >= 1.5


def test_diagnostics_are_registered_as_episode_metrics():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        # mean of 2·roll·sin(2πφ) = sway amplitude in phase with the reference
        "bounce_sway_amp_deg": "mean",
        "bounce_roll_ref_deg": "last",
        "bounce_roll_error_deg": "mean",
        "bounce_left_air_frac": "mean",
        "bounce_right_air_frac": "mean",
        "bounce_drift_m": "last",
        "bounce_pitch_deg": "mean",
    }
    for name, reduce in expected.items():
        assert name in cfg.metrics, name
        assert cfg.metrics[name].reduce == reduce, name
        assert callable(cfg.metrics[name].func), name


def test_play_mode_prints_per_episode_diagnostics():
    train_cfg = make_microduck_dance_env_cfg(play=False)
    play_cfg = make_microduck_dance_env_cfg(play=True)
    assert "bounce_print_diagnostics" not in train_cfg.events
    assert "bounce_print_diagnostics" in play_cfg.events
    assert "bounce_reset_origin" in train_cfg.events


def test_episode_holds_several_bounce_cycles():
    cfg = make_microduck_dance_env_cfg()
    assert EPISODE_LENGTH_S == pytest.approx(BOUNCE_PERIOD_S * 6)
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S)
    assert EPISODE_LENGTH_S / BOUNCE_PERIOD_S == pytest.approx(6.0)


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
