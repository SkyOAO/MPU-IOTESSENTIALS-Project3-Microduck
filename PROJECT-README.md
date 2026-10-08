New dance action: Trunk sway walking
Action environment configuration file: src/mjlab_microduck/tasks/microduck_dance_env_cfg.py
The action environment configuration file is based on src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py
New reward function:
All dance rewards live in the shared MDP module src/mjlab_microduck/tasks/mdp.py, in the dance_* block (roughly lines 5071-5480). The functions themselves carry no weight; the weights and parameters are registered in src/mjlab_microduck/tasks/microduck_dance_env_cfg.py (reward terms around lines 390-510, the weight curricula around 875-910, the metrics around 513-523).
Six terms are new: dance_sway_tracking and dance_sway_l1 (the left-right lean, i.e. the look and its gradient), dance_step_tracking (which foot is up while leaning), dance_forward_progress (distance walked; its weight is ramped from 0 to 4.0 by a curriculum), dance_heading_l1 (anti-drift), dance_pitch_balance (pitch only, replacing the stock upright term) and dance_head_hold (one head term instead of four).
One stock term is replaced rather than added: air_time becomes dance_air_time, gated to the walking half of the cycle. track_linear_velocity and track_angular_velocity are removed, because the twist command slot now carries the phase instead of a velocity. The three dance metrics (dance_sway_amp_deg, dance_forward_m, dance_yaw_drift_deg) are logged only and carry no weight.

Environment setup and reproduction (uv):
Run the local simulation end separately.
1. Install dependencies: uv sync (Python 3.12)
2. List tasks: uv run list-envs
3. Smoke test first: uv run train Mjlab-Dance-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
4. Train: uv run train Mjlab-Dance-Flat-MicroDuck --env.scene.num-envs 4096 (logs and checkpoints go to logs/rsl_rl/dance/; to resume add --agent.load-checkpoint model_XXXX.pt --agent.resume True)
5. Replay: uv run play Mjlab-Dance-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
6. Export ONNX: uv run scripts/export.py Mjlab-Dance-Flat-MicroDuck --wandb-run-path <entity/project/run_id> (the observation normalizer is baked into the ONNX; never hand-convert a checkpoint)
7. Deployment rehearsal on CPU: uv run scripts/infer_policy.py --walking onnx/alpha_walking.onnx --dance onnx/dance.onnx --dance-period 2.5 --new-cmd-obs (press D to trigger one 2.5 s cycle)
8. Run the tests: uv run --with pytest pytest tests/

Note: the dance policy is phase-encoded, so the runtime must write the phase [cos, sin, 0] into the twist slot of the 13D command block every control step (50 Hz, 0.02 s) and hand back to the walking policy when the cycle ends. 