import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
RAW_FILES = ("result_en.csv", "result.csv")
SUMMARY_FILE = "result_summary.csv"
OUTPUT_FILE = "latency_analysis.png"
GROUP_LABELS = {"main": "Main actions", "sitness": "Sit/stand"}
GROUP_COLORS = {"main": "#167d8d", "sitness": "#dc733e"}


def load_trials():
	"""Read raw trial fields by position because the CSV has duplicate headers."""
	source = next(
		(BASE_DIR / name for name in RAW_FILES if (BASE_DIR / name).exists()), None
	)
	if source is None:
		raise FileNotFoundError(f"Could not find a raw data file: {', '.join(RAW_FILES)}")

	trials = []
	with source.open(encoding="utf-8-sig", newline="") as stream:
		reader = csv.reader(stream)
		header = next(reader, [])
		expected = ["i", "round", "group", "action", "status", "ms"]
		if header[: len(expected)] != expected:
			raise ValueError(f"Unexpected raw CSV columns in {source.name}")

		for row in reader:
			if len(row) < 6 or row[4].strip().lower() != "finished":
				continue
			try:
				latency = float(row[5])
			except ValueError:
				continue
			trials.append(
				{"group": row[2].strip(), "action": row[3].strip(), "latency": latency}
			)

	if not trials:
		raise ValueError(f"No finished latency trials found in {source.name}")
	return source, trials


def load_summary():
	summary = defaultdict(dict)
	summary_path = BASE_DIR / SUMMARY_FILE
	if summary_path.exists():
		with summary_path.open(encoding="utf-8-sig", newline="") as stream:
			for row in csv.DictReader(stream):
				try:
					summary[row["group"]][row["metric"]] = float(row["value"])
				except (KeyError, ValueError):
					continue
	return summary


def percentile(values, fraction):
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	lower = math.floor(position)
	upper = math.ceil(position)
	return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def plot_latency(output_path, show=False):
	source, trials = load_trials()
	summary = load_summary()
	by_action = defaultdict(list)
	by_group = defaultdict(list)
	for trial in trials:
		by_action[(trial["group"], trial["action"])].append(trial["latency"])
		by_group[trial["group"]].append(trial["latency"])

	action_keys = sorted(by_action, key=lambda key: statistics.median(by_action[key]))
	figure, (action_axis, cdf_axis) = plt.subplots(
		1,
		2,
		figsize=(13, 6),
		gridspec_kw={"width_ratios": [1.35, 1]},
		constrained_layout=True,
	)

	rng = random.Random(7)
	for position, (group, action) in enumerate(action_keys):
		values = by_action[(group, action)]
		jitter = [position + rng.uniform(-0.12, 0.12) for _ in values]
		color = GROUP_COLORS.get(group, "#667085")
		action_axis.scatter(values, jitter, color=color, alpha=0.78, s=35, zorder=2)
		action_axis.scatter(
			statistics.median(values),
			position,
			color="#202b33",
			marker="D",
			s=48,
			zorder=3,
		)

	action_labels = [
		f"{GROUP_LABELS.get(group, group)} / {action.replace('_', ' ')}"
		for group, action in action_keys
	]
	action_axis.set_yticks(range(len(action_keys)), action_labels)
	action_axis.invert_yaxis()
	action_axis.set_xlabel("Latency (ms)")
	action_axis.set_title("Individual trials by action", loc="left", weight="bold")
	action_axis.grid(axis="x", color="#d8e0e3", linewidth=0.8)
	action_axis.set_axisbelow(True)
	action_axis.spines[["top", "right", "left"]].set_visible(False)
	action_axis.tick_params(axis="y", length=0)
	action_axis.legend(
		handles=[
			Line2D([0], [0], marker="o", linestyle="", color="#667085", label="Trial"),
			Line2D([0], [0], marker="D", linestyle="", color="#202b33", label="Median"),
		],
		loc="lower right",
		frameon=False,
	)

	group_order = [group for group in ("main", "sitness") if group in by_group]
	group_order.extend(group for group in by_group if group not in group_order)
	for group in group_order:
		values = sorted(by_group[group])
		color = GROUP_COLORS.get(group, "#667085")
		cumulative = [index / len(values) for index in range(1, len(values) + 1)]
		median = summary[group].get("ms_median", statistics.median(values))
		p95 = summary[group].get("ms_p95", percentile(values, 0.95))
		label = (
			f"{GROUP_LABELS.get(group, group)} n={len(values)} "
			f"| median {median:g} | p95 {p95:g} ms"
		)
		cdf_axis.step(values, cumulative, where="post", color=color, linewidth=2, label=label)
		cdf_axis.axvline(median, color=color, linewidth=1, alpha=0.75)
		cdf_axis.axvline(p95, color=color, linewidth=1, linestyle="--", alpha=0.75)

	cdf_axis.set_xlabel("Latency (ms)")
	cdf_axis.set_ylabel("Cumulative proportion")
	cdf_axis.set_ylim(0, 1.03)
	cdf_axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
	cdf_axis.set_title(
		"Group ECDF (solid: median; dashed: reported p95)",
		loc="left",
		weight="bold",
	)
	cdf_axis.grid(color="#d8e0e3", linewidth=0.8)
	cdf_axis.set_axisbelow(True)
	cdf_axis.spines[["top", "right"]].set_visible(False)
	cdf_axis.legend(loc="upper left", frameon=False, fontsize=8)

	success_notes = []
	for group in group_order:
		metrics = summary[group]
		if "success" in metrics and "trials" in metrics:
			rate = metrics.get("success_rate_percent", 100 * metrics["success"] / metrics["trials"])
			success_notes.append(
				f"{GROUP_LABELS.get(group, group)} "
				f"{metrics['success']:g}/{metrics['trials']:g} successful ({rate:g}%)"
			)
	success_text = f" | Success: {' | '.join(success_notes)}" if success_notes else ""
	figure.suptitle(
		f"Command latency | {len(trials)} finished trials{success_text}",
		x=0.02,
		ha="left",
		fontsize=15,
		weight="bold",
	)

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
	print(f"Saved chart to {output_path.resolve()}")
	if show:
		plt.show()
	else:
		plt.close(figure)


def main():
	parser = argparse.ArgumentParser(description="Plot command latency CSV data.")
	parser.add_argument(
		"--output",
		type=Path,
		default=BASE_DIR / OUTPUT_FILE,
		help=f"output image path (default: {OUTPUT_FILE})",
	)
	parser.add_argument("--show", action="store_true", help="also open the chart window")
	args = parser.parse_args()
	plot_latency(args.output, show=args.show)


if __name__ == "__main__":
	main()
