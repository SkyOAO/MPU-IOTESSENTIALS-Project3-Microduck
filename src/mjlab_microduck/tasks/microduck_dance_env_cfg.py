"""Microduck dance routine — walk forward at DANCE_FORWARD_SPEED, swaying per step.

Episodic phase policy replacing the in-place bounce (2026-10-03: the bounce
itself tracked well, but a move whose feet are mostly airborne cannot hold a
heading — it drifted ~0.9 m per 5 s episode). Walking fixes that: the feet have
traction, so the heading is controllable.

    one period (``DANCE_PERIOD_S``) = one walk, then a short closing stand
      [0, WALK_END)      walk at DANCE_FORWARD_SPEED, N_STEPS swaying steps
      [WALK_END, 1.0)    closing stand, both feet planted

The sway is FUSED into the walk, which is the only structure that has ever
produced it: the trunk leans ``DANCE_SWAY_DEG`` left and right once per step and
the right foot lifts while the trunk leans left, so a lift and a lean are the
same event (``dance_sway_reference`` and ``dance_step_schedule`` run off one
phase). Splitting it — walk level, then sway on the spot — was tried 2026-10-04
and never produced a lean in any run, because it asks for a single-support lean
from a standstill where the robot settles at 0.4 deg of trunk roll instead.
Walking already rolls the trunk a few degrees, so the reference only has to
amplify a motion the robot is making anyway.

Speed is the spec (``DANCE_FORWARD_SPEED``), not distance; ``N_STEPS`` sets the
cadence that speed is covered at. ``dance_forward_progress`` pays new forward
ground only up to that speed (potential-based, so marching in place pays nothing
and racing pays no more than walking does) and ``dance_heading_l1`` keeps the run
straight. The gait is shaped by ``dance_step_tracking``, the only term that knows
WHICH foot should be up WHEN and that refuses to pay for a lift taken with a level
trunk, plus the velocity recipe's `foot_clearance` / `foot_swing_height` (target
0.02) and `foot_slip`. `air_time` is deliberately NOT among them: its command gate
cannot close on a unit-circle phase, so it would go on paying for a lifted foot
all through the closing stand. The remaining terms are containment:
``dance_pitch_balance`` and ``dance_head_hold``.

The squats an earlier version ended with were dropped (2026-10-04): with a 25 mm
dip on a 0.45 s cycle the height term's Gaussian is already saturated ~20 mm away
from the target, so it gives no gradient, and neither the dip reward nor any
amount of training ever moved the trunk off its standing height.

Phase travels in the twist slot over the unit circle (``[cos, sin, 0]``) — the
runtime one-shot contract — so `GroundPickPhaseCommand` is reused unchanged.
Read ``Episode_Metrics/dance_*`` before touching a reward.
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

# Symmetry
ENABLE_SYMMETRY = False

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False # Was True
ENABLE_KD_RANDOMIZATION = False # Was True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = False  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Head/body pose command infrastructure from the velocity template is not used
# by this dance task. The 4D head_pose and 6D body_pose observation slots are
# kept and zero-padded to preserve the shared 61D runtime layout.

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped to ±8mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was ±0.5 — an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
# letting a calmer gait be optimal.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # ±0.86° per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

# Routine timing. One period is one full walk and the episode is exactly one
# period, so the phase starts and ends at 0 (the standing hand-over the runtime
# button presses give).
#
# The walk is the whole routine; the closing stand is only the period's tail. It
# needs no windowed reward terms of its own — the phase decode puts the sway
# reference level and the step schedule planted there, so the posture, pitch and
# heading terms that already exist are what hold the robot upright. What the
# stand DOES need is for nothing to pay for lifting a foot in it; see the note
# on `air_time` below.
WALK_S = 4.0      # N_STEPS swaying steps at 0.5 s — a normal walking cadence
STAND_S = 0.5
DANCE_PERIOD_S = WALK_S + STAND_S              # 4.5 s
EPISODE_LENGTH_S = DANCE_PERIOD_S              # 225 steps @ 50 Hz
WALK_END = 1.0    # the walk fills the period up to the stand
REP_END = WALK_S / DANCE_PERIOD_S
N_STEPS = 8

# Reverse-curriculum spawn mix: the fraction of episodes that start at phase 0,
# the real deployment hand-over. The rest start partway through the walk, so the
# last steps get on-policy data even while early episodes still end before they
# reach them. Kept low: a mid-walk spawn drops a standing robot into a phase that
# asks it to be mid-step, and the measured yaw drift is largely per-episode
# scatter rather than a fixed gait bias.
DANCE_START_PHASE_PROB = 0.1

# Trunk lean at each swing's peak, unchanged from the version this look was
# signed off in. One left-right swing per step, so the look and the step rhythm
# are the same clock.
DANCE_SWAY_DEG = 15.0
DANCE_SWAY_AMPLITUDE = math.radians(DANCE_SWAY_DEG)

# How far into the commanded lean a lift has to be before it pays, as a fraction
# of the reference. 0.3 = the trunk must be at least 30% of the way there while
# the scheduled foot is up; it costs nothing near the swing's zero crossings,
# where the reference itself is small.
DANCE_LEAN_FRAC = 0.3

# The speed the walk is asked to hold, in m/s. Not a distance: the routine is
# speed-commanded, so the ground covered is this times WALK_S (0.10 m/s over 4 s
# = 40 cm). Paid as potential-based progress that saturates here, so walking
# faster earns no more than this and the policy has no reason to race — see
# `dance_forward_progress`. A quarter of the velocity recipe's ±0.4 m/s.
DANCE_FORWARD_SPEED = 0.10

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# Microduck-specific rough terrain: much gentler than the default ROUGH_TERRAINS_CFG.
# The robot can only lift its feet ~1-2 cm, so steps are capped at 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # max 1.5 cm (vs 10 cm default)
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # NOTE: BoxInvertedPyramidStairsTerrainCfg removed — it sets env_origin_z to the pit
        # bottom (negative), causing resets at root_z = 0.12 + env_origin_z ≈ −0.10 m which
        # places the robot below the pit floor and makes it fall through the ground.
        # Uneven cobblestone-like ground: random per-cell height offsets.
        # grid_width=0.12 on an 8m patch = 66×66 = 4 356 boxes/patch → ~261 K total → OOM.
        # 0.45 m gives 17×17 = 289 boxes/patch → ~17 K total (border = 0.35 m ✓).
        # Must not divide evenly into terrain size (8.0 m): 0.45 × 17 = 7.65 ✓
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # max 1 cm
            platform_width=1.5,
        ),
        # Gentle slopes (heightfield pyramid, platform on TOP — robot spawns on
        # the flat platform and walks down/up/across the slope as commands
        # resample). slope_range is rise/run: 0.03→0.10 ≈ 1.7°→5.7° by
        # difficulty — small robot, small slopes. NOT inverted (see the
        # inverted-pyramid env_origin note above — same pit-spawn risk class).
        # vertical_scale=0.001 keeps quantization steps at 1 mm so a gentle
        # slope is smooth instead of a staircase of 5 mm ledges.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften terrain box geom contacts to reduce edge-contact NaN instability.

    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.

    Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
    2× softer — enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]   # 2× softer time constant (default: 0.02)
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # slightly softer impedance
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def make_microduck_dance_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck lateral-patrol dance environment configuration."""

    std_standing = {
        # Lower body — tighter to keep the robot in home pose when standing
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance (sole sits flat), stop leg splay
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance, stop the leg splay to vertical
        r".*hip_pitch.*": 0.4,
        # Knee/ankle are looser than the walking recipe (0.4 / 0.25): this gait
        # lifts the feet high and the tighter tolerance charged the step for
        # posing the legs the way the task asks.
        r".*knee.*": 0.6,
        r".*ankle.*": 0.4,
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT foot first, RIGHT foot second
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
    # now driven by a per-foot terrain-height ray sensor (was site_pos based).
    # Mirrors microban's foot_height_scan.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()

    cfg.episode_length_s = EPISODE_LENGTH_S

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = std_standing  # tight when command=0
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # Pose reward operates on LEG joints only: the head is 38 % of the mass and
    # gets its own, gentler hold (`dance_head_hold`) so it can still act as a
    # counterweight.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # NOTE: the stock `upright` term is dropped further down — it penalises roll
    # as well as pitch, and roll is exactly what the sway is asked to produce.
    # Its replacement, `dance_pitch_balance`, keeps the pitch half only.

    # foot_clearance and foot_slip still read foot sites from asset_cfg.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Body-specific configurations
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    cfg.rewards.pop("soft_landing", None)

    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # --- The routine: one walk that sways with its own steps -------------------
    # Every term below is phase-referenced, so none of them can be farmed by
    # standing still, and each only pays inside its own window.
    dance_cmd = {"command_name": "twist", "rep_end": REP_END}
    walk_params = {**dance_cmd, "walk_end": WALK_END}
    step_params = {**walk_params, "n_steps": N_STEPS}

    # The look. Both terms are window-gated, so neither scores in the closing
    # stand. The Gaussian is the look, the L1 is the gradient: at 15° of lean the
    # Gaussian has saturated, so each swing would otherwise start with no slope to
    # follow. The L1 carries 2.0 rather than the bounce's 0.6 because 0.6/rad was
    # the WHOLE gradient at the policy's own operating point (roll ~0, where the
    # Gaussian's slope is only ~0.3/rad) — a 400-iteration run simply sat there.
    cfg.rewards["dance_sway_tracking"] = RewardTermCfg(
        func=microduck_mdp.dance_sway_tracking,
        weight=5.0,
        params={**step_params, "amplitude": DANCE_SWAY_AMPLITUDE, "std": 0.10},
    )
    cfg.rewards["dance_sway_l1"] = RewardTermCfg(
        func=microduck_mdp.dance_sway_l1,
        weight=2.0,
        params={**step_params, "amplitude": DANCE_SWAY_AMPLITUDE},
    )

    # Which foot is up, for the whole walk. A lift only pays while the trunk is
    # leaning the way that step asks (`lean_frac` of the reference), which is what
    # stops the sway from being farmed by stepping on a level trunk — that was the
    # 1000-iteration run whose hips twisted instead of leaning.
    cfg.rewards["dance_step_tracking"] = RewardTermCfg(
        func=microduck_mdp.dance_step_tracking,
        weight=3.0,
        params={
            **step_params,
            "sensor_name": feet_ground_cfg.name,
            "amplitude": DANCE_SWAY_AMPLITUDE,
            "lean_frac": DANCE_LEAN_FRAC,
        },
    )

    cfg.rewards["dance_forward_progress"] = RewardTermCfg(
        func=microduck_mdp.dance_forward_progress,
        weight=0.0,  # final 4.0, ramped by the forward_progress_weight curriculum
        params={**walk_params, "setpoint_speed": DANCE_FORWARD_SPEED},
    )

    # The walker drifts 6–8°/s open loop, and the sway turns the robot just as
    # readily by shifting the weight side to side, so without this the routine
    # visibly curves: a 1000-iteration run ended 52° off its spawn heading
    # (≤ 0 → POSITIVE weight). L1, so oscillation cancels and only the heading
    # error itself is charged. 3.0 rather than the bounce's 1.0: at 1.0 the same
    # run measured a 32° mean error, i.e. the term was not enforcing anything.
    cfg.rewards["dance_heading_l1"] = RewardTermCfg(
        func=microduck_mdp.dance_heading_l1,
        weight=3.0,
        params={"command_name": "twist"},
    )

    # --- Containment ----------------------------------------------------------
    # Balance = pitch only. The stock `upright` penalises roll too, and roll is
    # what the sway is asked to produce, so it is replaced rather than tuned.
    # Held at 4.0 / 0.10 (the walking recipe uses 2.0 / 0.15): fore/aft is the
    # axis this robot actually falls on, and the tighter term is what carried the
    # 6000-iteration run to full-length episodes.
    cfg.rewards.pop("upright", None)
    cfg.rewards["dance_pitch_balance"] = RewardTermCfg(
        func=microduck_mdp.dance_pitch_balance,
        weight=4.0,
        params={"std": 0.10},
    )

    # One head term instead of four. The head must not flail (it is 38 % of the
    # mass), but a generous std leaves it free to counterbalance the walk.
    cfg.rewards["dance_head_hold"] = RewardTermCfg(
        func=microduck_mdp.dance_head_hold,
        weight=0.4,
        params={"std": 0.30},
    )

    # `track_*_velocity` compare the twist slot to real velocities. The slot now
    # carries the phase pair [cos, sin, 0], so both would be tracking a unit
    # circle as if it were m/s — removed, not re-weighted. Forward travel and
    # heading are priced by the dance_* terms above.
    cfg.rewards.pop("track_linear_velocity", None)
    cfg.rewards.pop("track_angular_velocity", None)

    # `pose` is the leg-posture anchor that stops the splayed-leg freeze; the
    # weight is the sway version's own.
    cfg.rewards["pose"].weight = 2.0

    # Gait shaping straight from the velocity recipe, values and all:
    # `foot_clearance` and `foot_swing_height` shape how high the swing gets,
    # `foot_slip` damps scraping. Lift TIMING is `dance_step_tracking`'s job.
    #
    # `air_time` is removed, not tuned. Its `command_threshold` gate can never
    # close on a unit-circle phase (the command norm is always 1.0), so it stays
    # armed in the closing stand, where every dance term is off and nothing else
    # competes: a 1000-iteration run kept one foot airborne for 89 % of the stand
    # and never stood at all, because lifting a foot there paid 3.0/step. Nothing
    # needs to be added in its place — `dance_step_tracking` already schedules the
    # lift, and the two foot-shape terms above still price the swing.
    cfg.rewards.pop("air_time", None)
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02
    for _term in ("foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[_term].params["command_threshold"] = 0.01
    cfg.rewards["foot_slip"].weight = -0.4

    # Trunk pitch/roll rate: damps the sway instead of forbidding it. Kept small
    # on purpose — the sway IS a trunk rotation, so this must not out-bid it.
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # Action smoothness (the curriculum below keeps tightening it). A routine is
    # a rhythm, not a whip, so smoothing is priced from the start.
    cfg.rewards["action_rate_l2"].weight = -0.08

    # --- Diagnostics --------------------------------------------------------
    # mjlab's MetricsManager: no reward weight, no dt scaling, logged as
    # `Episode_Metrics/dance_*`. `reduce="last"` = value at the FINAL step of the
    # episode (where the routine ended up); `reduce="mean"` for the per-step
    # curves. These are the numbers to read before touching a reward again.
    cfg.metrics["dance_sway_amp_deg"] = MetricsTermCfg(
        func=microduck_mdp.dance_metric_sway_amp_deg,
        params=step_params,
        reduce="mean",
    )
    cfg.metrics["dance_forward_m"] = MetricsTermCfg(
        func=microduck_mdp.dance_metric_forward_m, reduce="last"
    )
    cfg.metrics["dance_yaw_drift_deg"] = MetricsTermCfg(
        func=microduck_mdp.dance_metric_yaw_drift_deg, reduce="last"
    )

    # Events
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["dance_reset_origin"] = EventTermCfg(
        func=microduck_mdp.dance_reset_origin,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # Grippier footpad — narrowed from (0.3, 1.2)
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity-based pushes for robustness training
    if ENABLE_VELOCITY_PUSHES:
        # In play mode, use shorter interval for better visibility
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # required by domain_randomization=True
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.

    # Base orientation randomization (forces reactive behavior)
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # Observations
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Add base_lin_vel to critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # Determine gravity/accelerometer term name based on flag
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Replace the base velocity command with the routine phase. The twist slot
    # stays 3D but now carries the runtime one-shot contract over the unit
    # circle; phase 0 is the standing hand-over point, which is where the routine
    # starts walking.
    command = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0

    command_kwargs = vars(command)
    command_kwargs.pop("rel_turn_in_place_envs", None)

    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **command_kwargs,
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": DANCE_PERIOD_S,
            # Deployment starts at phase 0 (standing) and the phase then runs on
            # its own clock, so that slice keeps starting there. The rest are
            # scattered anywhere in the walk, because the policy otherwise only
            # practises the last steps by surviving the ones that precede them.
            "randomize_phase": True,
            "zero_phase_prob": DANCE_START_PHASE_PROB,
        }
    )

    # Append head + body command obs terms to both policy and critic groups.
    # This routine does not use either slot — the runtime feeds zeros for this
    # skill — so they are zero-padded to keep the shared 61D layout.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # Terrain
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # Soften terrain box contacts: adjacent boxes at different heights create
        # hard edges that destabilise the contact solver and produce NaN forces.
        cfg.scene.spec_fn = _soften_terrain_contacts

        # The velocity env default nconmax=35 is tight for rough terrain: when the
        # robot falls and multiple body links hit multiple boxes simultaneously,
        # contacts overflow → some are silently dropped → sudden decompression → NaN.
        cfg.sim.nconmax = 200   # was 35

        # The velocity env uses only 10 solver iterations (vs the default 100),
        # which is too few to resolve edge contacts on rough box terrain.
        # Tripling iterations significantly reduces contact resolution failures
        # with a modest compute cost on GPU (MJWarp parallelises across envs).
        cfg.sim.mujoco.iterations = 30    # was 10
        cfg.sim.mujoco.ls_iterations = 50  # was 20

        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # action_rate weight ramp. The previous turn task chattered badly at −0.05
    # (logged raw ~29.5 summed over 14 action dims ≈ 1.45 rad of target change
    # per joint per 20 ms step), so smoothing starts higher now. A routine is a
    # rhythm, and a rhythm cannot be learned through a jittery controller.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.08},
                {"step": 500 * 24,   "weight": -0.12},
                {"step": 1500 * 24,  "weight": -0.15},
            ],
        },
    )

    # Forward progress is introduced in stages, NOT from step 0. A headless
    # rollout of a 250-iteration run showed 90% of robots falling on the second
    # step with no episode surviving the walk window: pulled forward while it can
    # barely balance, the policy never gets far enough to learn. (The previous
    # run "survived" only because the forward term was effectively worth nothing
    # - its reward mass was 0.003 against the sway's 3.1 - so it learned a stable
    # march in place.) Stage 0 is zero: build the stepping and sway first, then
    # add translation once there is something to push from.
    cfg.curriculum["forward_progress_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "dance_forward_progress",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 600 * 24,   "weight": 1.0},
                {"step": 1200 * 24,  "weight": 2.5},
                {"step": 2000 * 24,  "weight": 4.0},
            ],
        },
    )

    # CoM randomization range curriculum - start small, ramp up
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm
                    # exceeded the foot support polygon (heel is only 20 mm behind
                    # the ankle) — the randomized CoM could sit entirely outside
                    # support, forcing a wide/fast hyper-reactive gait and making
                    # BACKWARD balance untrainable. Regression timeline matched the
                    # ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )

    # Head CoM randomization range curriculum - start small, ramp up
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # Capped at ±10 mm (2026-07 audit — same over-conservatism
                    # concern as trunk CoM; head is a large lever arm).
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # Disable default curriculum
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    return cfg


MicroduckDanceRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="dance",  # Directory name
    run_name="dance",  # Appended to datetime in wandb: <datetime>_velocity
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
