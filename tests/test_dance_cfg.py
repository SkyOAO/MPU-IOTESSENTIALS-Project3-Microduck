"""Cfg + MDP invariants for the in-place march (dance) task.

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
    MARCH_PERIOD_S,
    MARCH_SWAY_AMPLITUDE,
    MARCH_SWAY_DEG,
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
    assert torch.allclose(microduck_mdp.march_phase(command), phases, atol=1e-6)


def test_sway_is_left_first_then_right():
    # +roll = leaning LEFT. The reference peaks in the MIDDLE of each stance
    # half (φ=0.25 while the right foot is up) and is 0 at both hand-overs.
    phase = torch.tensor([0.0, 0.25, 0.5, 0.75])
    reference = microduck_mdp.march_roll_reference(phase, MARCH_SWAY_AMPLITUDE)
    assert reference.tolist() == pytest.approx(
        [0.0, MARCH_SWAY_AMPLITUDE, 0.0, -MARCH_SWAY_AMPLITUDE], abs=1e-6
    )


def test_sway_amplitude_is_large_enough_to_unload_a_foot():
    # Geometry (robot_walk.xml, HOME): the stance foot's inner edge is 21.2 mm
    # from the centreline and the CoM sits 148 mm up, so ~8° of lean is the
    # minimum that can lift the other foot and ~15° centres the CoM over the
    # stance foot. "Big amplitude" has to mean bigger than the minimum.
    assert 8.0 <= MARCH_SWAY_DEG <= 20.0
    assert MARCH_SWAY_AMPLITUDE == pytest.approx(math.radians(MARCH_SWAY_DEG))


# --------------------------------------------------------------------------- #
# Cfg invariants                                                               #
# --------------------------------------------------------------------------- #
def test_cfg_reuses_the_shared_phase_command():
    cfg = make_microduck_dance_env_cfg()
    command = cfg.commands["twist"]
    # No new command class: GroundPickPhaseCommand already emits [cos, sin, 0].
    assert isinstance(command, microduck_mdp.GroundPickPhaseCommandCfg)
    assert command.period == MARCH_PERIOD_S
    # Every episode starts at φ=0 (standing hand-over), like the button press.
    assert command.randomize_phase is False


def test_the_move_is_exactly_two_task_terms():
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["march_roll_tracking"].weight == pytest.approx(5.0)
    assert cfg.rewards["march_roll_tracking"].params["amplitude"] == pytest.approx(
        MARCH_SWAY_AMPLITUDE
    )
    assert cfg.rewards["march_step_tracking"].weight == pytest.approx(3.0)
    # The sway term must dominate every non-task term.
    task = cfg.rewards["march_roll_tracking"].weight
    for other in (
        "march_step_tracking",
        "march_pitch_balance",
        "march_stay_in_place",
        "air_time",
        "pose",
        "march_head_hold",
    ):
        assert task >= 1.5 * cfg.rewards[other].weight, other


def test_upright_is_replaced_by_a_pitch_only_balance_term():
    # `upright` penalises roll, which is the move; it must not come back.
    cfg = make_microduck_dance_env_cfg()
    assert "upright" not in cfg.rewards
    assert cfg.rewards["march_pitch_balance"].weight > 0.0


def test_velocity_tracking_terms_are_removed():
    # The twist slot carries [cos(2πφ), sin(2πφ), 0]; tracking it as a velocity
    # would reward moving at ~1 m/s.
    cfg = make_microduck_dance_env_cfg()
    assert "track_linear_velocity" not in cfg.rewards
    assert "track_angular_velocity" not in cfg.rewards


def test_self_negating_l1_term_keeps_a_positive_weight():
    # `march_roll_l1` returns ≤ 0 → POSITIVE weight (a negative weight would
    # double-negate into paying for the violation).
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["march_roll_l1"].weight > 0.0
    assert cfg.rewards["march_stay_in_place_l1"].weight > 0.0


def test_diagnostics_are_registered_as_episode_metrics():
    cfg = make_microduck_dance_env_cfg()
    expected = {
        "march_sway_amp_deg": "last",  # amplitude actually reached
        "march_roll_ref_deg": "last",
        "march_roll_error_deg": "mean",
        "march_left_air_frac": "mean",
        "march_right_air_frac": "mean",
        "march_drift_m": "last",
        "march_pitch_deg": "mean",
    }
    for name, reduce in expected.items():
        assert name in cfg.metrics, name
        assert cfg.metrics[name].reduce == reduce, name
        assert callable(cfg.metrics[name].func), name


def test_play_mode_prints_per_episode_diagnostics():
    train_cfg = make_microduck_dance_env_cfg(play=False)
    play_cfg = make_microduck_dance_env_cfg(play=True)
    assert "march_print_diagnostics" not in train_cfg.events
    assert "march_print_diagnostics" in play_cfg.events
    assert "march_reset_origin" in train_cfg.events


def test_episode_holds_several_march_cycles():
    cfg = make_microduck_dance_env_cfg()
    assert EPISODE_LENGTH_S == pytest.approx(MARCH_PERIOD_S * 6)
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S)
    assert EPISODE_LENGTH_S / MARCH_PERIOD_S == pytest.approx(6.0)


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
