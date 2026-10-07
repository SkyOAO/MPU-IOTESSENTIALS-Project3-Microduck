
from __future__ import annotations

import argparse
import csv
import io
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = "result_latency.csv"
OUTPUT_FILE = "action_success_rate.png"

# action key -> display label, in the order the commands are exercised.
ACTIONS = OrderedDict(
    (
        ("walk_forward", "Walk forward"),
        ("walk_backward", "Walk backward"),
        ("turn_left", "Turn left"),
        ("turn_right", "Turn right"),
        ("stop", "Stop"),
        ("dance", "Dance"),
        ("toggle_sit", "Sit"),
        ("toggle_stand", "Stand"),
    )
)

SUCCESS_COLOR = "#167d8d"
FAILURE_COLOR = "#dc733e"
SUCCESS_EDGE = "#0f5866"
FAILURE_EDGE = "#b3561f"

# Note markers used by the manual judgement. Both the original Chinese and the
# mojibake that appears when the note is read with the wrong encoding are listed.
FAILURE_MARKERS = ("失败", "澶辫触", "澶?", "fail")
SUCCESS_MARKERS = ("成功", "鎴愬姛", "鎴?", "success")


def read_rows(path):
    """Decode the CSV bytes and return the raw rows (list of lists)."""
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    return list(csv.reader(io.StringIO(text)))


def repair_note(note):
    """Undo the double-encoded Chinese that the log writes into the note field."""
    if not note:
        return note
    try:
        return note.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return note


def note_is_failure(note):
    """True when the judgement note marks the action as failed."""
    text = repair_note(note).lower()
    return any(marker in text for marker in FAILURE_MARKERS)


def normalize_action(action, step=""):
    """Map combined ``toggle_sitstand`` rows onto sit/stand actions."""
    if action in ("toggle_sitstand", "sit_stand", "toggle_sit_stand"):
        text = (step or "").lower()
        if "起立" in text or "stand" in text:
            return "toggle_stand"
        return "toggle_sit"
    return action


def load_outcomes(path):
    """Return an ordered mapping action -> [successes, total]."""
    rows = read_rows(path)
    rows = iter(rows)
    header = next(rows, [])
    index = {name.strip(): position for position, name in enumerate(header) if name.strip()}
    required = ("action", "status")
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

    outcomes = OrderedDict((action, [0, 0]) for action in ACTIONS)
    for row in rows:
        if len(row) <= index["action"]:
            continue
        step = row[index["step"]].strip() if "step" in index and len(row) > index["step"] else ""
        action = normalize_action(row[index["action"]].strip(), step)
        if not action:
            continue
        if action not in outcomes:
            outcomes[action] = [0, 0]

        status = row[index["status"]].strip().lower() if "status" in index and len(row) > index["status"] else ""
        note = row[index["note"]].strip() if "note" in index and len(row) > index["note"] else ""

        # A command counts as a success unless it never finished or its manual
        # judgement note flags a failure (the stand-up trials that did not complete).
        success = status == "finished" and not note_is_failure(note)

        outcomes[action][1] += 1
        if success:
            outcomes[action][0] += 1

    return OrderedDict((action, counts) for action, counts in outcomes.items() if counts[1])


def plot_accuracy(outcomes, output_path, show=False):
    labels = [ACTIONS.get(action, action.replace("_", " ").title()) for action in outcomes]
    successes = [counts[0] for counts in outcomes.values()]
    totals = [counts[1] for counts in outcomes.values()]
    rates = [100.0 * ok / total for ok, total in zip(successes, totals)]
    failures = [total - ok for ok, total in zip(successes, totals)]

    overall_ok = sum(successes)
    overall_total = sum(totals)
    overall_rate = 100.0 * overall_ok / overall_total

    positions = list(range(len(labels)))
    # Small canvas on purpose: the PNG is placed at ~12 cm in report.docx, so the
    # type must be sized relative to the image, not to a full-screen figure.
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)

    axis.barh(
        positions,
        rates,
        height=0.62,
        color=SUCCESS_COLOR,
        edgecolor=SUCCESS_EDGE,
        linewidth=0.8,
        zorder=3,
    )
    axis.barh(
        positions,
        [100.0 - rate for rate in rates],
        left=rates,
        height=0.62,
        color=FAILURE_COLOR,
        edgecolor=FAILURE_EDGE,
        linewidth=0.8,
        zorder=3,
    )

    for position, rate, ok, total, bad in zip(positions, rates, successes, totals, failures):
        axis.text(
            rate / 2,
            position,
            f"{ok}/{total}  ({rate:.0f}%)",
            va="center",
            ha="center",
            fontsize=11.5,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        if bad:
            axis.text(
                rate + (100.0 - rate) / 2,
                position,
                f"{bad} failed",
                va="center",
                ha="center",
                fontsize=10.5,
                fontweight="bold",
                color="white",
                zorder=5,
            )

    axis.axvline(
        overall_rate,
        color="#5a6b70",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Success rate", fontsize=11.5)
    axis.grid(axis="x", color="#d8e0e3", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=11.5)
    axis.tick_params(axis="x", labelsize=11)

    total_failures = overall_total - overall_ok
    handles = [
        Patch(facecolor=SUCCESS_COLOR, edgecolor=SUCCESS_EDGE, label="Completed"),
        Line2D([], [], color="#5a6b70", linestyle="--", linewidth=1.2,
               label=f"Overall success {overall_rate:.1f}%"),
    ]
    if total_failures:
        handles.insert(
            1,
            Patch(facecolor=FAILURE_COLOR, edgecolor=FAILURE_EDGE,
                  label="Failed / not completed"),
        )
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncols=3,
        frameon=False,
        fontsize=10.5,
        handlelength=1.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    if total_failures:
        title = (f"Action success rate | {overall_ok}/{overall_total} commands completed "
                 f"({overall_rate:.1f}%), {total_failures} stand-up failures")
        footnote = ("Failures are stand-up actions (Stand) that did not complete; "
                    "every other command finished successfully.")
    else:
        title = (f"Action success rate | {overall_ok}/{overall_total} commands completed "
                 f"({overall_rate:.1f}%), no failures")
        footnote = ("Every command finished successfully, including all sit and stand pairs; "
                    "the robot stood up on every attempt.")
    figure.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    figure.text(0.01, -0.06, footnote, ha="left", fontsize=9.5, color="#5a6b70")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved chart to {output_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return outcomes


def main():
    parser = argparse.ArgumentParser(
        description="Plot the per-action success rate from result_latency.csv."
    )
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
    args = parser.parse_args()

    outcomes = load_outcomes(args.input)
    for action, (ok, total) in outcomes.items():
        print(f"  {action:<14} {ok}/{total} = {100.0 * ok / total:5.1f}%")
    plot_accuracy(outcomes, args.output, show=args.show)


if __name__ == "__main__":
    main()
