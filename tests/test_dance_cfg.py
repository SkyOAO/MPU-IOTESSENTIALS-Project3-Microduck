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
    BOUNCE_TURN_DEG,
    BOUNCE_TURN_RAMP_S,
    BOUNCE_TURN_RATE_DEG_S,
    BOUNCE_TURN_S,
    BOUNCE_TURN_SIGN,
    BOUNCE_SETTLE_S,
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


def test_turn_budget_is_one_clockwise_revolution_at_the_accidental_speed():
    # One full revolution, so the robot finishes facing the heading it started
    # on. (Three was tried first and made the episode 42 s — too long.)
    assert BOUNCE_TURN_DEG == pytest.approx(360.0)
    assert BOUNCE_TURN_SIGN == -1.0  # clockwise, matching the measured -190°/ep
    # The RATE is primary and the duration is derived, so the integral is exact.
    assert BOUNCE_TURN_S == pytest.approx(
        BOUNCE_TURN_DEG / BOUNCE_TURN_RATE_DEG_S + BOUNCE_TURN_RAMP_S
    )
    # The speed reference is the ACCIDENTAL turn this task produces on its own
    # (measured 23.6-24.7°/s); 27°/s is that rate rounded up.
    assert 24.0 <= BOUNCE_TURN_RATE_DEG_S <= 30.0


def test_turn_schedule_integrates_to_the_requested_degrees():
    t = torch.linspace(0.0, EPISODE_LENGTH_S, 2001)
    target = microduck_mdp.bounce_turn_target_deg(
        t,
        rate_deg_s=BOUNCE_TURN_RATE_DEG_S,
        total_deg=BOUNCE_TURN_DEG,
        ramp_s=BOUNCE_TURN_RAMP_S,
        turn_sign=BOUNCE_TURN_SIGN,
    )
    assert target[0].item() == pytest.approx(0.0, abs=1e-6)
    # Monotone clockwise, and it HOLDS the finish (no coasting past 3 turns).
    assert torch.all(target[1:] <= target[:-1] + 1e-6)
    assert target[-1].item() == pytest.approx(-BOUNCE_TURN_DEG, abs=1e-6)
    held = microduck_mdp.bounce_turn_target_deg(
        torch.tensor([1.0e4]),
        rate_deg_s=BOUNCE_TURN_RATE_DEG_S,
        total_deg=BOUNCE_TURN_DEG,
        ramp_s=BOUNCE_TURN_RAMP_S,
        turn_sign=BOUNCE_TURN_SIGN,
    )
    assert held.item() == pytest.approx(-BOUNCE_TURN_DEG, abs=1e-6)


def test_heading_hold_terms_are_gone():
    """The move is three COMMANDED revolutions: keeping the start-heading pair
    would charge the task itself (~19 rad/step by the last cycle)."""
    cfg = make_microduck_dance_env_cfg()
    assert "bounce_heading_hold" not in cfg.rewards
    assert "bounce_heading_l1" not in cfg.rewards
    assert not hasattr(microduck_mdp, "bounce_heading_hold")


def test_the_move_is_three_task_terms():
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["bounce_roll_tracking"].weight == pytest.approx(5.0)
    assert cfg.rewards["bounce_roll_tracking"].params["amplitude"] == pytest.approx(
        BOUNCE_SWAY_AMPLITUDE
    )
    assert cfg.rewards["bounce_lift_tracking"].weight == pytest.approx(3.0)
    assert cfg.rewards["bounce_turn_tracking"].weight == pytest.approx(5.0)
    # The move (sway + lift + turn, with the stock gait shaping that feeds the
    # bounce) must out-bid everything that only says "do not misbehave".
    smallest_task = min(
        cfg.rewards[n].weight
        for n in ("bounce_roll_tracking", "bounce_lift_tracking", "bounce_turn_tracking", "air_time")
    )
    largest_containment = max(
        cfg.rewards[n].weight
        for n in (
            "bounce_pitch_balance",
            "bounce_stay_in_place",
            "bounce_stay_in_place_l1",
            "bounce_turn_l1",
            "bounce_roll_l1",
            "pose",
            "bounce_head_hold",
        )
    )
    assert smallest_task >= largest_containment


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
    for name in ("bounce_roll_l1", "bounce_stay_in_place_l1", "bounce_turn_l1"):
        assert cfg.rewards[name].weight > 0.0, name


def test_containment_terms_do_not_dominate_the_reward_mass():
    """2026-10-01 lesson: at heading_l1 = 1.5 + stay_l1 = 3.0 the containment
    terms cost ~30 % of the positive reward and achieved NEITHER goal (the robot
    still turned 84° and travelled 1.1 m). What is left is a gentle bias on the
    travel only; the turn is now commanded instead of fought."""
    cfg = make_microduck_dance_env_cfg()
    containment = (
        cfg.rewards["bounce_stay_in_place"].weight
        + cfg.rewards["bounce_stay_in_place_l1"].weight
    )
    assert containment <= cfg.rewards["bounce_roll_tracking"].weight
    assert cfg.rewards["bounce_stay_in_place_l1"].weight <= 1.0


def test_yaw_accumulator_separates_net_turn_from_wobble():
    """The signed accumulators are what the "can we control the turns?" decision
    rests on, so pin their semantics: net cancels, path accumulates."""
    import types

    def quat(yaw):
        return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])

    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(root_link_quat_w=quat(0.0))
    )
    env = types.SimpleNamespace(
        num_envs=1, device=torch.device("cpu"), common_step_counter=0
    )
    microduck_mdp._update_bounce_yaw(env, asset)  # initialise at yaw 0
    assert env._bounce_yaw_net.item() == pytest.approx(0.0)

    for step, yaw in enumerate((0.10, 0.0), start=1):  # out to +0.1 rad, back
        env.common_step_counter = step
        asset.data.root_link_quat_w = quat(yaw)
        microduck_mdp._update_bounce_yaw(env, asset)

    # Net rotation cancels (it ends facing where it started); the yaw PATH does
    # not — 0.1 rad out plus 0.1 rad back.
    assert env._bounce_yaw_net.item() == pytest.approx(0.0, abs=1e-6)
    assert env._bounce_yaw_path.item() == pytest.approx(0.20, abs=1e-6)


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
        # Rotation measurement: signed net turn, total turn path, signed rate.
        "bounce_yaw_total_deg": "last",
        "bounce_yaw_path_deg": "last",
        "bounce_yaw_rate_deg_s": "mean",
        # Commanded rotation and the pass/fail line (measured − commanded).
        "bounce_turn_target_deg": "last",
        "bounce_turn_error_deg": "last",
    }
    for name, reduce in expected.items():
        assert name in cfg.metrics, name
        assert cfg.metrics[name].reduce == reduce, name
        assert callable(cfg.metrics[name].func), name


def test_every_registered_term_takes_env_as_its_first_parameter():
    """mjlab calls reward/metric functions as ``func(env, **params)``.

    Registering a pure helper by mistake binds the env to whatever its first
    argument is and only explodes at run time — the 2026-10-01 crash was
    ``bounce_turn_target_deg(t, ...)`` registered as a metric, so ``t`` became
    the env and ``t.clamp`` raised AttributeError. Signature binding alone
    cannot catch that; the parameter NAME can.
    """
    import inspect

    cfg = make_microduck_dance_env_cfg()
    for kind, terms in (("reward", cfg.rewards), ("metric", cfg.metrics)):
        for name, term in terms.items():
            if not inspect.isfunction(term.func):
                continue  # class-based stock terms are instantiated with (cfg, env)
            first = next(iter(inspect.signature(term.func).parameters))
            assert first == "env", f"{kind} '{name}' must take env first, got {first!r}"


def test_turn_target_metric_is_callable_the_way_mjlab_calls_it():
    """Call the registered wrapper exactly as the metrics manager does.

    The static audit above can be satisfied by a wrapper that still crashes
    inside; this actually runs it against a stub env.
    """
    import types

    cfg = make_microduck_dance_env_cfg()
    term = cfg.metrics["bounce_turn_target_deg"]
    env = types.SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        step_dt=0.02,
    )
    value = term.func(env, **term.params)
    assert value.shape == (2,)
    assert value[0].item() == pytest.approx(0.0, abs=1e-6)

    # Half way through the turn it must be a real fraction of the target.
    # NB: the clock is our own step counter, not `episode_length_buf`.
    env._bounce_step_count = torch.full((2,), int(7.0 / 0.02), dtype=torch.long)
    value = term.func(env, **term.params)
    assert -BOUNCE_TURN_DEG < value[0].item() < 0.0


def test_turn_clock_ignores_the_randomized_episode_length_buffer():
    """The runner randomizes `episode_length_buf` at startup to decorrelate
    episode phases, so it must NOT be the turn schedule's clock: on the first
    episode of every env the schedule would start deep inside its ramp (logged
    2026-10-01: target -324° while the episode was 0.5 s old)."""
    import types

    env = types.SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        step_dt=0.02,
        episode_length_buf=torch.tensor([651, 900]),  # ← the randomized buffer
    )
    # Our own clock starts at 0 regardless of what the buffer says.
    assert microduck_mdp.bounce_episode_time(env).tolist() == [0.0, 0.0]


def test_l1_penalties_stay_O_of_one():
    """An L1 term that scales with DEGREES is a trap: one revolution behind
    costs 360×weight per step, GROWING with episode time, which pays the robot
    to terminate early (the 2026-10-01 collapse: episodes 24 → 6.7 steps,
    negative return, every task term 0). Worst-case weighted magnitudes here
    must stay O(1)."""
    import types

    cfg = make_microduck_dance_env_cfg()
    term = cfg.rewards["bounce_turn_l1"]
    # Worst case: the robot has not turned at all while the target is complete.
    env = types.SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ))},
        common_step_counter=0,
        _bounce_step_count=torch.full((1,), 10_000, dtype=torch.long),
    )
    worst = term.func(env, **term.params) * term.weight
    assert worst.item() > -3.0, f"turn L1 worst case {worst.item():.1f} is not O(1)"


def test_play_mode_prints_per_episode_diagnostics():
    train_cfg = make_microduck_dance_env_cfg(play=False)
    play_cfg = make_microduck_dance_env_cfg(play=True)
    assert "bounce_print_diagnostics" not in train_cfg.events
    assert "bounce_print_diagnostics" in play_cfg.events
    assert "bounce_reset_origin" in train_cfg.events


def test_episode_holds_several_bounce_cycles():
    cfg = make_microduck_dance_env_cfg()
    # One revolution at the accidental speed, plus the settle, plus a little
    # margin. The episode must outlast the turn or the finish never trains.
    assert EPISODE_LENGTH_S == pytest.approx(18.0)
    assert EPISODE_LENGTH_S > BOUNCE_TURN_S
    assert EPISODE_LENGTH_S - BOUNCE_TURN_S >= BOUNCE_SETTLE_S
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S)
    # A whole number of bounce cycles, so the phase starts at 0 every episode.
    assert EPISODE_LENGTH_S / BOUNCE_PERIOD_S == pytest.approx(
        round(EPISODE_LENGTH_S / BOUNCE_PERIOD_S)
    )


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
