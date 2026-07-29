import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


DEFAULT_BAG_DIR = (
    "ros2/tracking_test_2_apr/best_results/"
    "test_trak_05hz_loadmodel_05hz_0.15A_v2_stopgood"
)
DEFAULT_LOG_PATH = "mpc_tracking_prediction_log.json"


def load_prediction_log(log_path: Path) -> list[dict]:
    with open(log_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of log entries in {log_path}")

    entries = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if "replay_time" not in entry:
            continue
        current_key = "bag_eef_pos_current" if "bag_eef_pos_current" in entry else "eef_pos_current"
        desired_key = "bag_eef_pos_desired" if "bag_eef_pos_desired" in entry else "eef_pos_desired"
        if current_key not in entry or desired_key not in entry:
            continue
        if "predicted_eef_horizon" not in entry:
            continue

        pred = np.asarray(entry["predicted_eef_horizon"], dtype=float)
        if pred.ndim != 2 or pred.shape[1] != 3:
            continue

        entries.append(
            {
                "time": float(entry["replay_time"]),
                "dt": float(entry.get("t", 0.0)),
                "eef_current": np.asarray(entry[current_key], dtype=float),
                "eef_desired": np.asarray(entry[desired_key], dtype=float),
                "eef_pred": pred,
            }
        )

    if not entries:
        raise ValueError(f"No valid prediction entries found in {log_path}")

    entries.sort(key=lambda e: e["time"])
    t0 = entries[0]["time"]
    for entry in entries:
        entry["time_rel"] = entry["time"] - t0

    return entries


def slice_entries(entries: list[dict], t_start: float, t_end: float) -> list[dict]:
    sliced = [e for e in entries if t_start <= e["time_rel"] <= t_end]
    if not sliced:
        t_min = entries[0]["time_rel"]
        t_max = entries[-1]["time_rel"]
        raise ValueError(
            f"No entries found in [{t_start}, {t_end}] s. "
            f"Available replay range is [{t_min:.3f}, {t_max:.3f}] s."
        )
    return sliced


def downsample_entries(entries: list[dict], fps: int) -> list[dict]:
    if len(entries) <= 1:
        return entries

    t_start = entries[0]["time_rel"]
    t_end = entries[-1]["time_rel"]
    duration = max(t_end - t_start, 0.0)
    if duration <= 0.0:
        return [entries[0]]

    target_times = np.arange(t_start, t_end + 0.5 / fps, 1.0 / fps)
    source_times = np.asarray([e["time_rel"] for e in entries], dtype=float)
    source_idx = np.arange(len(entries))
    sampled_indices = np.interp(target_times, source_times, source_idx)
    sampled_indices = np.clip(np.rint(sampled_indices).astype(int), 0, len(entries) - 1)

    downsampled = []
    last_idx = None
    for idx in sampled_indices:
        if idx != last_idx:
            downsampled.append(entries[idx])
            last_idx = idx

    return downsampled


def compute_axis_limits(entries: list[dict]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    current = np.asarray([e["eef_current"] for e in entries])
    desired = np.asarray([e["eef_desired"] for e in entries])
    pred = np.concatenate([e["eef_pred"] for e in entries], axis=0)

    all_pts = np.vstack([current, desired, pred])

    def lim(vals: np.ndarray, margin: float = 0.01) -> tuple[float, float]:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if np.isclose(vmin, vmax):
            vmin -= margin
            vmax += margin
        else:
            vmin -= margin
            vmax += margin
        return vmin, vmax

    return lim(all_pts[:, 0]), lim(all_pts[:, 1]), lim(all_pts[:, 2])


def build_animation(entries: list[dict], bag_name: str, output_path: Path, fps: int) -> None:
    _, y_lim, z_lim = compute_axis_limits(entries)
    current_all = np.asarray([e["eef_current"] for e in entries])
    desired_all = np.asarray([e["eef_desired"] for e in entries])

    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    fig.suptitle(f"MPC Tracking Prediction: {bag_name}")

    ax.set_xlim(y_lim)
    ax.set_ylim(z_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world y [m]")
    ax.set_ylabel("world z [m]")
    ax.grid(True, alpha=0.2)

    desired_path, = ax.plot(
        desired_all[:, 1], desired_all[:, 2],
        color="tab:blue", lw=2.0, alpha=0.35, label="desired path"
    )
    current_trail, = ax.plot([], [], color="tab:green", lw=2.0, alpha=0.85, label="current trail")
    pred_snake, = ax.plot([], [], "o", color="limegreen", ms=5, alpha=0.9, label="mpc horizon")
    pred_head, = ax.plot([], [], "o", color="gold", ms=8, alpha=0.95)
    current_marker, = ax.plot([], [], "o", color="tab:green", ms=10, alpha=1.0, label="current")
    desired_marker, = ax.plot([], [], "o", color="tab:blue", ms=10, alpha=1.0, label="desired")
    ax.legend(loc="upper right")

    time_text = ax.text(
        0.03, 0.97, "", transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8")
    )

    def update(frame_idx: int):
        entry = entries[frame_idx]
        pred = entry["eef_pred"]
        trail_start = max(0, frame_idx - 25)
        trail = current_all[trail_start:frame_idx + 1]

        current_trail.set_data(trail[:, 1], trail[:, 2])
        pred_snake.set_data(pred[:, 1], pred[:, 2])
        pred_head.set_data([pred[0, 1]], [pred[0, 2]])
        current_marker.set_data([entry["eef_current"][1]], [entry["eef_current"][2]])
        desired_marker.set_data([entry["eef_desired"][1]], [entry["eef_desired"][2]])
        time_text.set_text(f"t = {entry['time_rel']:.2f} s")

        return (
            desired_path,
            current_trail,
            pred_snake,
            pred_head,
            current_marker,
            desired_marker,
            time_text,
        )

    anim = FuncAnimation(fig, update, frames=len(entries), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def select_snapshot_entry(entries: list[dict], snapshot_time: float | None) -> dict:
    if snapshot_time is None:
        return entries[len(entries) // 2]

    return min(entries, key=lambda e: abs(e["time_rel"] - snapshot_time))


def build_snapshot(entries: list[dict], bag_name: str, output_path: Path, snapshot_time: float | None) -> None:
    entry = select_snapshot_entry(entries, snapshot_time)
    current = np.asarray([e["eef_current"] for e in entries])
    desired = np.asarray([e["eef_desired"] for e in entries])
    pred = entry["eef_pred"]
    _, y_lim, z_lim = compute_axis_limits(entries)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    fig.suptitle(f"MPC Tracking Prediction Snapshot: {bag_name}")
    ax.set_xlim(y_lim)
    ax.set_ylim(z_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world y [m]")
    ax.set_ylabel("world z [m]")
    ax.grid(True, alpha=0.2)

    trail_idx = min(range(len(entries)), key=lambda i: abs(entries[i]["time_rel"] - entry["time_rel"]))
    trail_start = max(0, trail_idx - 25)
    trail = current[trail_start:trail_idx + 1]

    ax.plot(desired[:, 1], desired[:, 2], color="tab:blue", lw=2.0, alpha=0.35, label="desired path")
    ax.plot(trail[:, 1], trail[:, 2], color="tab:green", lw=2.0, alpha=0.9, label="current trail")
    ax.plot(pred[:, 1], pred[:, 2], "o", color="limegreen", ms=5, alpha=0.9, label="mpc horizon")
    ax.plot([pred[0, 1]], [pred[0, 2]], "o", color="gold", ms=8, alpha=0.95)
    ax.plot([entry["eef_current"][1]], [entry["eef_current"][2]], "o", color="tab:green", ms=10, label="current")
    ax.plot([entry["eef_desired"][1]], [entry["eef_desired"][2]], "o", color="tab:blue", ms=10, label="desired")
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.97,
        f"snapshot t = {entry['time_rel']:.2f} s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an MPC tracking prediction video from replay JSON.")
    parser.add_argument("--bag-dir", default=DEFAULT_BAG_DIR, help="Bag directory used only for naming/output defaults.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to mpc_tracking_prediction_log.json from replay.")
    parser.add_argument("--t-start", type=float, default=5.0, help="Start time in seconds from replay start.")
    parser.add_argument("--t-end", type=float, default=17.0, help="End time in seconds from replay start.")
    parser.add_argument("--fps", type=int, default=20, help="Output video frame rate.")
    parser.add_argument("--output", default=None, help="Output mp4 path.")
    parser.add_argument("--image", action="store_true", help="Save a static PNG snapshot instead of an MP4.")
    parser.add_argument("--snapshot-time", type=float, default=None, help="Snapshot time in seconds from replay start.")
    return parser.parse_args()


def main():
    args = parse_args()
    bag_dir = Path(args.bag_dir)
    log_path = Path(args.log_path)

    if args.output is None:
        suffix = "png" if args.image else "mp4"
        output_path = Path.cwd() / f"{bag_dir.name}_prediction_{int(args.t_start)}_{int(args.t_end)}s.{suffix}"
    else:
        output_path = Path(args.output)

    entries = load_prediction_log(log_path)
    entries = slice_entries(entries, args.t_start, args.t_end)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.image:
        build_snapshot(entries, bag_dir.name, output_path, args.snapshot_time)
        print(f"Saved image to {output_path}")
    else:
        entries = downsample_entries(entries, args.fps)
        build_animation(entries, bag_dir.name, output_path, args.fps)
        print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
