
from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = "result_latency.csv"
OUTPUT_FILE = "latency_report.png"

# action -> (display label, group)
ACTIONS = {
    "walk_forward": ("Walk forward", "main"),
    "walk_backward": ("Walk backward", "main"),
    "turn_left": ("Turn left", "main"),
    "turn_right": ("Turn right", "main"),
    "stop": ("Stop", "main"),
    "dance": ("Dance", "main"),
    "toggle_sit": ("Sit", "sitness"),
    "toggle_stand": ("Stand", "sitness"),
}
GROUP_LABELS = {"main": "Main actions", "sitness": "Sit / stand"}
GROUP_COLORS = {"main": "#167d8d", "sitness": "#dc733e"}

# metric key -> (display label, colour); ordered small -> large contribution
METRICS = (
    ("server_ms", "Server", "#9fc3ca"),
    ("api_ms", "API", "#4f9db0"),
    ("e2e_ms", "End-to-end", "#12495a"),
)


def load_records(path):
    """Read ``result_latency.csv`` into a list of trial dictionaries."""
    raw = path.read_bytes()
    text = None
    # The file is UTF-8, but a couple of Chinese characters in the note column
    # were truncated when it was written, so fall back to a tolerant decode.
    for encoding, errors in (
        ("utf-8-sig", "strict"),
        ("utf-8", "strict"),
        ("utf-8", "replace"),
        ("gbk", "replace"),
    ):
        try:
            text = raw.decode(encoding, errors=errors)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Could not decode {path.name} as UTF-8 or GBK")

    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    index = {name.strip(): position for position, name in enumerate(header) if name.strip()}
    required = ("trial", "action", "api_ms", "e2e_ms", "server_ms", "status")
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

    records = []
    for order, row in enumerate(reader):
        if len(row) <= index["status"]:
            continue
        if row[index["status"]].strip().lower() != "finished":
            continue
        try:
            action = row[index["action"]].strip()
            step = ""
            if "step" in index and len(row) > index["step"]:
                step = row[index["step"]].strip()
            record = {
                "order": order,
                "action": normalize_action(action, step),
                "step": step,
                "api_ms": float(row[index["api_ms"]]),
                "e2e_ms": float(row[index["e2e_ms"]]),
                "server_ms": float(row[index["server_ms"]]),
            }
            record["trial"] = int(float(row[index["trial"]]))
        except (ValueError, IndexError):
            continue
        note_index = index.get("note")
        record["note"] = row[note_index].strip() if note_index is not None and len(row) > note_index else ""
        record["success"] = not note_marks_failure(record["note"])
        records.append(record)

    if not records:
        raise ValueError(f"No finished latency rows found in {path.name}")
    return records


def normalize_action(action, step=""):
    """Map the combined ``toggle_sitstand`` rows onto sit/stand actions.

    Newer logs record the pair as a single action with a ``step`` column
    ("蹲下" for sitting down, "起立" for standing up) instead of one action per
    row. Older logs already use ``toggle_sit`` / ``toggle_stand``.
    """
    if action in ("toggle_sitstand", "sit_stand", "toggle_sit_stand"):
        text = (step or "").lower()
        if "起立" in text or "stand" in text:
            return "toggle_stand"
        return "toggle_sit"
    return action


FAILURE_MARKERS = ("失败", "澶辫触", "婢惰精瑙", "婢", "fail")


def note_marks_failure(note):
    """True when a manual judgement note marks the action as failed."""
    if not note:
        return False
    text = note.lower()
    return any(marker in text for marker in FAILURE_MARKERS)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def group_of(action):
    return ACTIONS.get(action, (action, "main"))[1]


def label_of(action):
    return ACTIONS.get(action, (action.replace("_", " ").title(), "main"))[0]


def sit_stand_outcome(records):
    """Return (successes, total) sit/stand pairs.

    A pair counts as a success when both the sit and the stand command
    completed and no note flagged a failure. Newer logs record the pair as one
    row per step, so the trial number is what links the two commands.
    """
    pairs = {}
    for record in records:
        if record["action"] not in ("toggle_sit", "toggle_stand"):
            continue
        pairs.setdefault(record["trial"], {})[record["action"]] = record["success"]
    if not pairs:
        return None
    total = len(pairs)
    ok = sum(
        1
        for steps in pairs.values()
        if steps.get("toggle_stand", False) and steps.get("toggle_sit", True)
    )
    return ok, total


def panel_metric_bars(axis, by_action, action_order, rng):
    height = 0.24
    offsets = {"server_ms": -height, "api_ms": 0.0, "e2e_ms": height}
    positions = np.arange(len(action_order))

    for key, label, colour in METRICS:
        medians = [statistics.median(by_action[action][key]) for action in action_order]
        axis.barh(
            positions + offsets[key],
            medians,
            height=height,
            color=colour,
            label=label,
            zorder=2,
        )
        for position, action in zip(positions, action_order):
            values = by_action[action][key]
            jitter = rng.uniform(-0.06, 0.06, size=len(values))
            axis.scatter(
                values,
                np.full(len(values), position + offsets[key]) + jitter,
                s=10,
                color="#0d2a33",
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )

    axis.set_yticks(positions, [label_of(action) for action in action_order])
    axis.invert_yaxis()
    axis.set_xlabel("Latency (ms)")
    axis.set_title("Latency per action (bar: median, dots: trials)", loc="left", weight="bold")
    axis.set_xlim(0, max(value for action in action_order for value in by_action[action]["e2e_ms"]) * 1.3)
    axis.grid(axis="x", color="#d8e0e3", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for tick, action in zip(axis.get_yticklabels(), action_order):
        tick.set_color(GROUP_COLORS[group_of(action)])
    handles = [Patch(facecolor=colour, label=label) for _, label, colour in METRICS]
    handles.append(Line2D([0], [0], marker="o", linestyle="", color="#0d2a33", label="Trial"))
    axis.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)


def panel_group_ecdf(axis, by_group, records):
    group_order = [
        group
        for group in ("main", "sitness")
        if group in by_group
    ]
    group_order += [group for group in by_group if group not in group_order]

    for group in group_order:
        values = sorted(by_group[group])
        colour = GROUP_COLORS.get(group, "#667085")
        cumulative = [index / len(values) for index in range(1, len(values) + 1)]
        median = statistics.median(values)
        p95 = percentile(values, 0.95)
        axis.step(
            values,
            cumulative,
            where="post",
            color=colour,
            linewidth=2,
            label=f"{GROUP_LABELS.get(group, group)} n={len(values)} | median {median:g} | p95 {p95:g} ms",
        )
        axis.axvline(median, color=colour, linewidth=1, alpha=0.7)
        axis.axvline(p95, color=colour, linewidth=1, linestyle="--", alpha=0.7)

    axis.set_xlabel("End-to-end latency (ms)")
    axis.set_ylabel("Cumulative proportion")
    axis.set_ylim(0, 1.03)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_title("Group ECDF of end-to-end latency (dashed: p95)", loc="left", weight="bold")
    axis.grid(color="#d8e0e3", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False, fontsize=8)


def panel_timeline(axis, records):
    ordered = sorted(records, key=lambda record: record["order"])
    x = np.arange(1, len(ordered) + 1)
    for key, label, colour in METRICS:
        values = [record[key] for record in ordered]
        axis.plot(x, values, color=colour, linewidth=1.2, alpha=0.9, label=label)
        axis.scatter(x, values, color=colour, s=12, zorder=3, linewidths=0)

    window = 9
    e2e = np.array([record["e2e_ms"] for record in ordered])
    if len(e2e) >= window:
        kernel = np.ones(window) / window
        axis.plot(
            x,
            np.convolve(e2e, kernel, mode="same"),
            color="#12495a",
            linewidth=2.2,
            linestyle="--",
            alpha=0.85,
            label=f"End-to-end {window}-point mean",
            zorder=4,
        )

    axis.set_xlabel("Command order during the session")
    axis.set_ylabel("Latency (ms)")
    axis.set_xlim(0, len(ordered) + 1)
    axis.set_title("Latency over the session", loc="left", weight="bold")
    axis.grid(color="#d8e0e3", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False, fontsize=8, ncols=2)


def panel_composition(axis, by_action, action_order):
    positions = np.arange(len(action_order))
    server = [statistics.median(by_action[action]["server_ms"]) for action in action_order]
    api = [statistics.median(by_action[action]["api_ms"]) for action in action_order]
    e2e = [statistics.median(by_action[action]["e2e_ms"]) for action in action_order]
    overhead = [total - base for total, base in zip(e2e, server)]

    axis.barh(positions, server, height=0.55, color="#3f7d8c", label="Server processing")
    axis.barh(
        positions,
        overhead,
        left=server,
        height=0.55,
        color="#c9d6d9",
        label="Network + client overhead",
    )
    axis.hlines(positions, server, api, color="#dc733e", linewidth=2.4)
    axis.scatter(api, positions, color="#dc733e", zorder=4, s=34, marker="|", linewidths=2)

    for position, (total, value) in enumerate(zip(e2e, api)):
        axis.text(total + 2, position, f"{total:g}", va="center", fontsize=8, color="#12495a")
        axis.text(value, position - 0.42, f"API {value:g}", va="bottom", ha="center", fontsize=7, color="#dc733e")

    axis.set_yticks(positions, [label_of(action) for action in action_order])
    axis.invert_yaxis()
    axis.set_xlabel("Median latency (ms)")
    axis.set_title("End-to-end composition per action", loc="left", weight="bold")
    axis.set_xlim(0, max(e2e) * 1.45)
    axis.grid(axis="x", color="#d8e0e3", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for tick, action in zip(axis.get_yticklabels(), action_order):
        tick.set_color(GROUP_COLORS[group_of(action)])
    handles = [
        Patch(facecolor="#3f7d8c", label="Server processing"),
        Patch(facecolor="#c9d6d9", label="Network + client overhead"),
        Line2D([0], [0], color="#dc733e", linewidth=2.4, label="API round trip"),
    ]
    axis.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)


def plot_latency(data_path, output_path, show=False):
    records = load_records(data_path)
    by_action = defaultdict(lambda: defaultdict(list))
    by_group = defaultdict(list)
    for record in records:
        for key, _, _ in METRICS:
            by_action[record["action"]][key].append(record[key])
        by_group[group_of(record["action"])].append(record["e2e_ms"])

    known = [action for action in ACTIONS if action in by_action]
    extras = sorted(action for action in by_action if action not in ACTIONS)
    action_order = sorted(
        known + extras, key=lambda action: statistics.median(by_action[action]["e2e_ms"])
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), constrained_layout=True)
    rng = np.random.default_rng(7)
    panel_metric_bars(axes[0, 0], by_action, action_order, rng)
    panel_group_ecdf(axes[0, 1], by_group, records)
    panel_timeline(axes[1, 0], records)
    panel_composition(axes[1, 1], by_action, action_order)

    trials = max(record["trial"] for record in records)
    e2e_all = [record["e2e_ms"] for record in records]
    title = (
        f"Command latency | {data_path.name} | {trials} trials / {len(records)} commands "
        f"| e2e median {statistics.median(e2e_all):.1f} ms, p95 {percentile(e2e_all, 0.95):.1f} ms"
    )
    outcome = sit_stand_outcome(records)
    if outcome:
        successes, total = outcome
        title += f" | Sit/stand success {successes}/{total} ({100 * successes / total:.0f}%)"
    figure.suptitle(title, x=0.01, ha="left", fontsize=14, weight="bold")
    figure.text(
        0.01,
        -0.01,
        "Metric definitions: server = server-side processing, API = client API round trip, "
        "end-to-end = issue-to-completion latency.",
        ha="left",
        fontsize=8,
        color="#5a6b70",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved chart to {output_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return records, by_action, by_group


def print_summary(records):
    print(f"rows: {len(records)}  trials: {max(record['trial'] for record in records)}")
    for action in sorted({record["action"] for record in records}):
        subset = [record for record in records if record["action"] == action]
        print(
            f"  {action:<14} n={len(subset):<3} "
            f"server={statistics.median(r['server_ms'] for r in subset):6.1f} "
            f"api={statistics.median(r['api_ms'] for r in subset):6.1f} "
            f"e2e={statistics.median(r['e2e_ms'] for r in subset):6.1f} ms"
        )
    outcome = sit_stand_outcome(records)
    if outcome:
        successes, total = outcome
        print(f"  sit/stand manual judgement: {successes}/{total} successful")


def main():
    parser = argparse.ArgumentParser(description="Plot result_latency.csv command latency data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / DATA_FILE,
        help=f"input CSV path (default: {DATA_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / OUTPUT_FILE,
        help=f"output image path (default: {OUTPUT_FILE})",
    )
    parser.add_argument("--show", action="store_true", help="also open the chart window")
    parser.add_argument("--summary", action="store_true", help="print per-action statistics")
    args = parser.parse_args()

    records, _, _ = plot_latency(args.input, args.output, show=args.show)
    if args.summary:
        print_summary(records)


if __name__ == "__main__":
    main()
