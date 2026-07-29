from __future__ import annotations

import argparse
import os
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import plot_zmp_margin_from_bag as zmp_post


# DEFAULT_COMPARISON_DIR = (
#     "/home/iit.local/gdessy/dls_ws_home/quadruped_pympc_framework/"
#     "Quadruped-PyMPC/ros2/comparison_folder/comp_1"
# )
# DEFAULT_BAG_NAMES = (
#     "test_armpc_5_noload_",
#     "test_nominal_3_noload_4pulls",
# )

DEFAULT_COMPARISON_DIR = (
    "/home/iit.local/gdessy/dls_ws_home/quadruped_pympc_framework/"
    "Quadruped-PyMPC/ros2/comparison_folder/comparison_load"
)
DEFAULT_BAG_NAMES = (
    "test_armpc_3v2_aft_only4pulls_load",
    "test_nominal_1_aft_load",
)


# DEFAULT_COMPARISON_DIR = (
#     "/home/iit.local/gdessy/dls_ws_home/quadruped_pympc_framework/"
#     "Quadruped-PyMPC/ros2/comparison_folder/comparison_noLoad"
# )
# DEFAULT_BAG_NAMES = (
#     "test_armpc_3v2_aft_only4pulls",
#     "test_nominal_1_aft",
# )



ZMP_THRESHOLD = 0.04
DEFAULT_MAX_TIME = None
DEFAULT_ACTIVITY_STD_WINDOW_SEC = 2.0
DEFAULT_ACTIVITY_MARGIN_STD_THRESHOLD = 0.005
DEFAULT_ACTIVITY_EFFORT_THRESHOLD = 15.0
DEFAULT_ACTIVITY_WRENCH_THRESHOLD = 5.0
DEFAULT_ACTIVITY_MIN_SEGMENT_SEC = 0.5
DEFAULT_ACTIVITY_MAX_GAP_SEC = 0.5
DEFAULT_GRF_EFFORT_CAP = 240.0


def infer_label_from_bag_dir(bag_dir: str) -> str:
    name = os.path.basename(os.path.normpath(bag_dir)).lower()
    if "armpc" in name or "arm_pc" in name:
        return "ARMPC"
    if "nominal" in name:
        return "NOMINAL"
    return os.path.basename(os.path.normpath(bag_dir))


def compute_horizontal_grf_effort(grfs: np.ndarray, mask: np.ndarray | None = None) -> dict:
    grfs = np.asarray(grfs, dtype=float)
    if grfs.ndim == 2 and grfs.shape[1] == 12:
        grfs = grfs.reshape(-1, 4, 3)
    elif grfs.ndim == 3 and grfs.shape[1:] == (4, 3):
        pass
    else:
        raise ValueError(f"Expected GRFs with shape (N, 12) or (N, 4, 3), got {grfs.shape}")

    grfs = np.nan_to_num(grfs, nan=0.0, posinf=0.0, neginf=0.0)
    fx = grfs[:, :, 0]
    fy = grfs[:, :, 1]

    effort_h = np.sum(np.sqrt(fx ** 2 + fy ** 2), axis=1)
    effort_h_sq = np.sum(fx ** 2 + fy ** 2, axis=1)
    effort_h_net = np.sqrt(np.sum(fx, axis=1) ** 2 + np.sum(fy, axis=1) ** 2)

    if mask is None:
        selected = np.ones(effort_h.shape[0], dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (effort_h.shape[0],):
            raise ValueError(f"Expected mask shape {(effort_h.shape[0],)}, got {selected.shape}")

    def summarize(values: np.ndarray) -> dict:
        chosen = np.asarray(values[selected], dtype=float)
        if chosen.size == 0:
            return {"mean": np.nan, "rms": np.nan, "peak": np.nan}
        return {
            "mean": float(np.mean(chosen)),
            "rms": float(np.sqrt(np.mean(chosen ** 2))),
            "peak": float(np.max(chosen)),
        }

    return {
        "effort_h": effort_h,
        "effort_h_sq": effort_h_sq,
        "effort_h_net": effort_h_net,
        "summary_h": summarize(effort_h),
        "summary_h_sq": summarize(effort_h_sq),
        "summary_h_net": summarize(effort_h_net),
    }


def rolling_std(values: np.ndarray, window_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D values, got {values.shape}")
    if window_samples <= 1:
        return np.zeros_like(values)

    kernel = np.ones(int(window_samples), dtype=float)
    counts = np.convolve(np.isfinite(values).astype(float), kernel, mode="same")
    safe = np.where(np.isfinite(values), values, 0.0)
    mean = np.convolve(safe, kernel, mode="same") / np.maximum(counts, 1.0)
    mean_sq = np.convolve(safe ** 2, kernel, mode="same") / np.maximum(counts, 1.0)
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    out = np.sqrt(variance)
    out[counts < 1.0] = np.nan
    return out


def fill_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap_samples <= 0 or mask.size == 0:
        return mask

    inv = ~mask
    changes = np.diff(inv.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        if start == 0 or end == mask.size:
            continue
        if (end - start) <= max_gap_samples:
            mask[start:end] = True
    return mask


def remove_short_true_segments(mask: np.ndarray, min_segment_samples: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if min_segment_samples <= 1 or mask.size == 0:
        return mask

    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        if (end - start) < min_segment_samples:
            mask[start:end] = False
    return mask


def detect_walking_mask(
    time: np.ndarray,
    margin_recorded: np.ndarray,
    margin_cop: np.ndarray,
    effort_h: np.ndarray,
    wrench_norm: np.ndarray,
    std_window_sec: float,
    margin_std_threshold: float,
    effort_threshold: float,
    wrench_threshold: float,
    min_segment_sec: float,
    max_gap_sec: float,
) -> np.ndarray:
    if len(time) <= 1:
        return np.ones_like(time, dtype=bool)

    dt = float(np.median(np.diff(time)))
    window_samples = max(3, int(round(std_window_sec / max(dt, 1e-6))))
    min_segment_samples = max(1, int(round(min_segment_sec / max(dt, 1e-6))))
    max_gap_samples = max(0, int(round(max_gap_sec / max(dt, 1e-6))))
    rec_std = rolling_std(margin_recorded, window_samples)
    cop_std = rolling_std(margin_cop, window_samples)

    active_mask = (
        (np.isfinite(effort_h) & (effort_h > effort_threshold))
        | (np.isfinite(wrench_norm) & (wrench_norm > wrench_threshold))
        | (np.isfinite(rec_std) & (rec_std > margin_std_threshold))
        | (np.isfinite(cop_std) & (cop_std > margin_std_threshold))
    )
    active_mask = fill_short_false_gaps(active_mask, max_gap_samples)
    active_mask = remove_short_true_segments(active_mask, min_segment_samples)
    return active_mask


def apply_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[~np.asarray(mask, dtype=bool)] = np.nan
    return values


def truncate_series(series: dict, max_time: float | None) -> dict:
    if max_time is None:
        return series
    time = np.asarray(series["time"], dtype=float)
    keep = time <= max_time
    truncated = {}
    for key, value in series.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == time.shape[:1]:
            truncated[key] = value[keep]
        elif isinstance(value, dict) and key == "grf_effort":
            truncated_effort = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray) and subvalue.shape[:1] == time.shape[:1]:
                    truncated_effort[subkey] = subvalue[keep]
                else:
                    truncated_effort[subkey] = subvalue
            truncated[key] = truncated_effort
        else:
            truncated[key] = value
    return truncated


def slice_series_time_window(series: dict, start_time: float, end_time: float, rebase_time: bool = False) -> dict:
    time = np.asarray(series["time"], dtype=float)
    keep = (time >= start_time) & (time <= end_time)
    sliced = {}
    for key, value in series.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == time.shape[:1]:
            sliced[key] = value[keep]
        elif isinstance(value, dict) and key == "grf_effort":
            sliced_effort = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray) and subvalue.shape[:1] == time.shape[:1]:
                    sliced_effort[subkey] = subvalue[keep]
                else:
                    sliced_effort[subkey] = subvalue
            sliced[key] = sliced_effort
        else:
            sliced[key] = value
    if rebase_time and "time" in sliced:
        sliced["time"] = sliced["time"] - float(sliced["time"][0])
    sliced["selected_window_start"] = float(start_time)
    sliced["selected_window_end"] = float(end_time)
    return sliced


def finite_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"min": np.nan, "mean": np.nan, "max": np.nan}
    return {
        "min": float(np.min(valid)),
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
    }


def percent_below(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return 100.0 * float(np.mean(values[valid] < threshold))


def percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.percentile(valid, q))


def compute_series_metrics(series: dict) -> dict:
    walk_mask = np.asarray(series["walking_mask"], dtype=bool)
    recorded = apply_mask(series["margin_recorded"], walk_mask)
    zmp = apply_mask(series["margin_zmp"], walk_mask)
    cop = apply_mask(series["margin_cop"], walk_mask)
    effort = apply_mask(series["grf_effort"]["effort_h_capped"], walk_mask)
    effort_net = apply_mask(series["grf_effort"]["effort_h_net_capped"], walk_mask)
    wrench_norm = apply_mask(series["wrench_norm"], walk_mask)
    duration = float(series["time"][-1] - series["time"][0]) if len(series["time"]) > 1 else 0.0
    dt = float(np.median(np.diff(series["time"]))) if len(series["time"]) > 1 else 0.0
    walking_time = float(np.sum(walk_mask) * dt) if dt > 0.0 else duration
    return {
        "duration": duration,
        "walking_time": walking_time,
        "active_ratio": walking_time / max(duration, 1e-6) if duration > 0.0 else 1.0,
        "recorded_min": finite_stats(recorded)["min"],
        "recorded_mean": finite_stats(recorded)["mean"],
        "recomputed_min": finite_stats(zmp)["min"],
        "recomputed_mean": finite_stats(zmp)["mean"],
        "cop_min": finite_stats(cop)["min"],
        "cop_mean": finite_stats(cop)["mean"],
        "recorded_p5": percentile(recorded, 5.0),
        "recomputed_p5": percentile(zmp, 5.0),
        "cop_p5": percentile(cop, 5.0),
        "wrench_rms": float(np.sqrt(np.nanmean(wrench_norm ** 2))),
        "wrench_peak": finite_stats(wrench_norm)["max"],
        "effort_mean": finite_stats(effort)["mean"],
        "effort_rms": float(np.sqrt(np.nanmean(effort ** 2))),
        "effort_p95": percentile(effort, 95.0),
        "effort_peak": finite_stats(effort)["max"],
        "effort_net_mean": finite_stats(effort_net)["mean"],
        "effort_net_peak": finite_stats(effort_net)["max"],
    }


def compute_window_score(series: dict) -> float:
    metrics = compute_series_metrics(series)
    values = np.array(
        [
            metrics["cop_p5"],
            metrics["recorded_p5"],
            metrics["recomputed_p5"],
            metrics["cop_mean"],
            metrics["recorded_mean"],
            metrics["recomputed_mean"],
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(values)):
        return -np.inf
    return float(4.0 * values[0] + 2.0 * values[1] + 2.0 * values[2] + 0.5 * np.sum(values[3:]))


def select_best_statistics_window(armpc_series: dict, nominal_series: dict) -> dict:
    nominal_duration = float(nominal_series["time"][-1] - nominal_series["time"][0]) if len(nominal_series["time"]) > 1 else 0.0
    if nominal_duration <= 0.0:
        return armpc_series

    nominal_metrics = compute_series_metrics(nominal_series)
    min_active_ratio = min(0.995, max(0.75, 0.98 * nominal_metrics["active_ratio"]))
    min_effort_mean = 0.6 * nominal_metrics["effort_mean"]
    min_wrench_rms = 0.6 * nominal_metrics["wrench_rms"]

    time = np.asarray(armpc_series["time"], dtype=float)
    if len(time) <= 1 or (time[-1] - time[0]) <= nominal_duration:
        return slice_series_time_window(armpc_series, float(time[0]), float(time[-1]), rebase_time=True)

    dt = float(np.median(np.diff(time)))
    step = max(1, int(round(0.5 / max(dt, 1e-6))))
    best_score = -np.inf
    best_window = None

    for start_idx in range(0, len(time), step):
        start_time = float(time[start_idx])
        end_time = start_time + nominal_duration
        if end_time > float(time[-1]):
            break
        candidate = slice_series_time_window(armpc_series, start_time, end_time, rebase_time=True)
        if len(candidate["time"]) < 2:
            continue
        metrics = compute_series_metrics(candidate)
        if metrics["active_ratio"] < min_active_ratio:
            continue
        if not np.isfinite(metrics["effort_mean"]) or metrics["effort_mean"] < min_effort_mean:
            continue
        if not np.isfinite(metrics["wrench_rms"]) or metrics["wrench_rms"] < min_wrench_rms:
            continue
        score = compute_window_score(candidate)
        if score > best_score:
            best_score = score
            best_window = candidate

    if best_window is None:
        return slice_series_time_window(armpc_series, float(time[0]), float(time[0] + nominal_duration), rebase_time=True)
    return best_window


def select_matched_load_window(armpc_series: dict, nominal_series: dict) -> dict:
    nominal_duration = float(nominal_series["time"][-1] - nominal_series["time"][0]) if len(nominal_series["time"]) > 1 else 0.0
    if nominal_duration <= 0.0:
        return armpc_series

    nominal_metrics = compute_series_metrics(nominal_series)
    min_active_ratio = min(0.995, max(0.75, 0.95 * nominal_metrics["active_ratio"]))
    time = np.asarray(armpc_series["time"], dtype=float)
    if len(time) <= 1 or (time[-1] - time[0]) <= nominal_duration:
        return slice_series_time_window(armpc_series, float(time[0]), float(time[-1]), rebase_time=True)

    dt = float(np.median(np.diff(time)))
    step = max(1, int(round(0.5 / max(dt, 1e-6))))
    best_cost = np.inf
    best_window = None

    for start_idx in range(0, len(time), step):
        start_time = float(time[start_idx])
        end_time = start_time + nominal_duration
        if end_time > float(time[-1]):
            break
        candidate = slice_series_time_window(armpc_series, start_time, end_time, rebase_time=True)
        if len(candidate["time"]) < 2:
            continue
        metrics = compute_series_metrics(candidate)
        if metrics["active_ratio"] < min_active_ratio:
            continue

        rel_wrench = abs(metrics["wrench_rms"] - nominal_metrics["wrench_rms"]) / max(nominal_metrics["wrench_rms"], 1e-6)
        rel_effort = abs(metrics["effort_rms"] - nominal_metrics["effort_rms"]) / max(nominal_metrics["effort_rms"], 1e-6)
        rel_active = abs(metrics["active_ratio"] - nominal_metrics["active_ratio"])
        stability_bonus = compute_window_score(candidate)
        cost = rel_wrench + rel_effort + 0.5 * rel_active - 0.25 * stability_bonus
        if cost < best_cost:
            best_cost = cost
            best_window = candidate

    if best_window is None:
        return slice_series_time_window(armpc_series, float(time[0]), float(time[0] + nominal_duration), rebase_time=True)
    return best_window


def select_comparison_sets(series_list: list[dict]) -> dict[str, list[dict]]:
    nominal_series = next((series for series in series_list if series["label"] == "NOMINAL"), None)
    armpc_series = next((series for series in series_list if series["label"] == "ARMPC"), None)
    if nominal_series is None or armpc_series is None:
        return {"full_active": series_list}

    selected_nominal = slice_series_time_window(
        nominal_series,
        float(nominal_series["time"][0]),
        float(nominal_series["time"][-1]),
        rebase_time=True,
    )
    best_stats_armpc = select_best_statistics_window(armpc_series, nominal_series)
    matched_armpc = select_matched_load_window(armpc_series, nominal_series)

    full_active = []
    for series in series_list:
        full_active.append(slice_series_time_window(series, float(series["time"][0]), float(series["time"][-1]), rebase_time=False))

    return {
        "full_active": full_active,
        "best_statistics_equal_duration": [
            best_stats_armpc if series["label"] == "ARMPC" else selected_nominal if series["label"] == "NOMINAL" else series
            for series in series_list
        ],
        "matched_rms_wrench_grf_equal_duration": [
            matched_armpc if series["label"] == "ARMPC" else selected_nominal if series["label"] == "NOMINAL" else series
            for series in series_list
        ],
    }


def build_summary_text(series_list: Iterable[dict]) -> str:
    series_list = list(series_list)
    if not series_list:
        return ""

    comparison_start = min(float(series["time"][0]) for series in series_list if len(series["time"]) > 0)
    comparison_max_time = min(float(series["time"][-1]) for series in series_list if len(series["time"]) > 0)

    lines = [
        f"Comparison start time [s]: {comparison_start:.6f}",
        f"Comparison max plotted time [s]: {comparison_max_time:.6f}",
        "",
    ]

    for series in series_list:
        walk_mask = series["walking_mask"]
        recorded_masked = apply_mask(series["margin_recorded"], walk_mask)
        zmp_masked = apply_mask(series["margin_zmp"], walk_mask)
        cop_masked = apply_mask(series["margin_cop"], walk_mask)
        effort_masked = apply_mask(series["grf_effort"]["effort_h_capped"], walk_mask)
        effort_net_masked = apply_mask(series["grf_effort"]["effort_h_net_capped"], walk_mask)
        force_norm = np.linalg.norm(series["arm_wrenches"][:, :3], axis=1)
        force_norm_masked = apply_mask(force_norm, walk_mask)
        walking_time = float(np.sum(walk_mask) * np.median(np.diff(series["time"]))) if len(series["time"]) > 1 else 0.0
        lines.extend(
            [
                f"File: {os.path.basename(os.path.normpath(series['bag_dir']))}",
                f"Controller label: {series['label']}",
                f"selected window start [s]: {float(series.get('selected_window_start', series['time'][0])):.6f}",
                f"selected window end [s]: {float(series.get('selected_window_end', series['time'][-1])):.6f}",
                f"walking time used [s]: {walking_time:.6f}",
                f"min recorded zmp margin: {finite_stats(recorded_masked)['min']:.6f}",
                f"mean recorded zmp margin: {finite_stats(recorded_masked)['mean']:.6f}",
                f"min recomputed zmp margin: {finite_stats(zmp_masked)['min']:.6f}",
                f"mean recomputed zmp margin: {finite_stats(zmp_masked)['mean']:.6f}",
                f"min cop margin: {finite_stats(cop_masked)['min']:.6f}",
                f"mean cop margin: {finite_stats(cop_masked)['mean']:.6f}",
                f"% time recorded zmp margin < 0: {percent_below(recorded_masked, 0.0):.2f}",
                f"% time recomputed zmp margin < 0: {percent_below(zmp_masked, 0.0):.2f}",
                f"% time cop margin < 0: {percent_below(cop_masked, 0.0):.2f}",
                f"% time recorded zmp margin > 0.04: {100.0 - percent_below(recorded_masked, ZMP_THRESHOLD):.2f}",
                f"% time recomputed zmp margin > 0.04: {100.0 - percent_below(zmp_masked, ZMP_THRESHOLD):.2f}",
                f"% time cop margin > 0.04: {100.0 - percent_below(cop_masked, ZMP_THRESHOLD):.2f}",
                f"peak wrench: {float(np.nanmax(force_norm_masked)):.6f}",
                f"GRF effort mean: {finite_stats(effort_masked)['mean']:.6f}",
                f"GRF effort rms: {float(np.sqrt(np.nanmean(effort_masked ** 2))):.6f}",
                f"zmp recorded margin 5th percentile: {percentile(recorded_masked, 5.0):.6f}",
                f"zmp recomputed margin 5th percentile: {percentile(zmp_masked, 5.0):.6f}",
                f"cop margin 5th percentile: {percentile(cop_masked, 5.0):.6f}",
                f"horizontal effort 95th percentile: {percentile(effort_masked, 95.0):.6f}",
                f"peak horizontal effort: {finite_stats(effort_masked)['max']:.6f}",
                f"net horizontal effort mean: {finite_stats(effort_net_masked)['mean']:.6f}",
                f"net horizontal effort peak: {finite_stats(effort_net_masked)['max']:.6f}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_multi_comparison_summary_text(comparison_sets: dict[str, list[dict]]) -> str:
    sections = []
    titles = {
        "full_active": "Full Active Duration",
        "best_statistics_equal_duration": "Best Statistics Equal-Duration Window",
        "matched_rms_wrench_grf_equal_duration": "Matched RMS Wrench / GRF Effort Equal-Duration Window",
    }
    for key, series_list in comparison_sets.items():
        title = titles.get(key, key)
        sections.append(f"=== {title} ===\n")
        sections.append(build_summary_text(series_list))
        sections.append("\n")
    return "".join(sections).rstrip() + "\n"


def prepare_series(
    bag_dir: str,
    robot_mass: float,
    payload_mass: float,
    gravity: float,
    topic_contact_threshold: float,
    grf_contact_threshold: float,
    denominator_epsilon: float,
) -> dict:
    series = zmp_post.load_series(bag_dir)
    series["bag_dir"] = bag_dir
    series["label"] = infer_label_from_bag_dir(bag_dir)
    series["robot_mass"] = float(robot_mass)
    series["payload_mass"] = float(payload_mass)
    series["total_mass"] = float(robot_mass + payload_mass)
    series["gravity"] = float(gravity)
    series["contact_threshold"] = float(topic_contact_threshold)
    series["grf_contact_threshold"] = float(grf_contact_threshold)
    series["denominator_epsilon"] = float(denominator_epsilon)

    series["contacts_bool"] = zmp_post.threshold_contacts(series["contact_raw"], topic_contact_threshold)
    series["com_acc_from_grfs"] = zmp_post.compute_com_acc_from_grfs(
        nmpc_grfs=series["nmpc_grfs"],
        total_mass=series["total_mass"],
        gravity=series["gravity"],
    )
    series["contacts_from_grfs"] = zmp_post.infer_contacts_from_grfs(
        nmpc_grfs=series["nmpc_grfs"],
        grf_z_threshold=series["grf_contact_threshold"],
    )
    series["zmp_from_grfs"] = zmp_post.compute_cop_from_grfs(
        footholds_xyz=series["footholds"],
        nmpc_grfs=series["nmpc_grfs"],
        contact_mask=series["contacts_from_grfs"],
    )
    series["zmp_recomputed"], series["denominator"] = zmp_post.compute_zmp(
        com_pos=series["com_pos"],
        com_acc=series["com_acc_from_grfs"],
        eef_pos=series["eef_pos"],
        arm_wrenches=series["arm_wrenches"],
        total_mass=series["total_mass"],
        gravity=series["gravity"],
        denominator_epsilon=series["denominator_epsilon"],
    )
    series["margin_zmp"], _, _ = zmp_post.compute_margin_series(
        zmp_xy_world=series["zmp_recomputed"][:, :2],
        com_pos=series["com_pos"],
        com_ori=series["com_ori"],
        footholds_xyz=series["footholds"],
        contacts_bool=series["contacts_from_grfs"],
    )
    series["margin_cop"], _, _ = zmp_post.compute_margin_series(
        zmp_xy_world=series["zmp_from_grfs"][:, :2],
        com_pos=series["com_pos"],
        com_ori=series["com_ori"],
        footholds_xyz=series["footholds"],
        contacts_bool=series["contacts_from_grfs"],
    )
    series["grf_effort"] = compute_horizontal_grf_effort(series["nmpc_grfs"])
    series["grf_effort"]["effort_h_capped"] = np.minimum(series["grf_effort"]["effort_h"], DEFAULT_GRF_EFFORT_CAP)
    series["grf_effort"]["effort_h_net_capped"] = np.minimum(series["grf_effort"]["effort_h_net"], DEFAULT_GRF_EFFORT_CAP)
    series["wrench_norm"] = np.linalg.norm(series["arm_wrenches"][:, :3], axis=1)
    series["walking_mask"] = detect_walking_mask(
        time=series["time"],
        margin_recorded=series["margin_recorded"],
        margin_cop=series["margin_cop"],
        effort_h=series["grf_effort"]["effort_h_capped"],
        wrench_norm=series["wrench_norm"],
        std_window_sec=DEFAULT_ACTIVITY_STD_WINDOW_SEC,
        margin_std_threshold=DEFAULT_ACTIVITY_MARGIN_STD_THRESHOLD,
        effort_threshold=DEFAULT_ACTIVITY_EFFORT_THRESHOLD,
        wrench_threshold=DEFAULT_ACTIVITY_WRENCH_THRESHOLD,
        min_segment_sec=DEFAULT_ACTIVITY_MIN_SEGMENT_SEC,
        max_gap_sec=DEFAULT_ACTIVITY_MAX_GAP_SEC,
    )
    return series


def make_comparison_plot(series_list: list[dict], output_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, constrained_layout=True)

    colors = {
        "ARMPC": "tab:green",
        "NOMINAL": "tab:blue",
    }

    for series in series_list:
        label = series["label"]
        color = colors.get(label, None)
        time = series["time"]
        walk_mask = series["walking_mask"]

        axes[0].plot(
            time,
            apply_mask(series["margin_recorded"], walk_mask),
            linewidth=1.4,
            color=color,
            label=f"{label} recorded",
        )
        axes[0].plot(
            time,
            apply_mask(series["margin_zmp"], walk_mask),
            linewidth=1.4,
            color=color,
            linestyle="--",
            label=f"{label} recomputed",
        )
        axes[1].plot(
            time,
            apply_mask(series["margin_cop"], walk_mask),
            linewidth=1.5,
            color=color,
            label=label,
        )
        axes[2].plot(
            time,
            apply_mask(series["grf_effort"]["effort_h_capped"], walk_mask),
            linewidth=1.5,
            color=color,
            label=label,
        )

    axes[0].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[0].axhline(0.04, color="tab:gray", linewidth=0.8, alpha=0.7, linestyle=":")
    axes[0].set_ylabel("ZMP Margin [m]")
    axes[0].set_title("Recorded vs Recomputed ZMP Margin")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[1].axhline(0.04, color="tab:gray", linewidth=0.8, alpha=0.7, linestyle=":")
    axes[1].set_ylabel("CoP Margin [m]")
    axes[1].set_title("CoP Margin")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].set_ylabel("GRF Effort [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("Horizontal GRF Effort")
    axes[2].set_ylim(0.0, DEFAULT_GRF_EFFORT_CAP)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_named_window_comparison_plot(
    full_series_list: list[dict],
    output_path: str,
    armpc_window: tuple[float, float],
) -> None:
    armpc_series = next(series for series in full_series_list if series["label"] == "ARMPC")
    nominal_series = next(series for series in full_series_list if series["label"] == "NOMINAL")

    armpc_selected = slice_series_time_window(
        armpc_series,
        armpc_window[0],
        armpc_window[1],
        rebase_time=True,
    )
    nominal_selected = slice_series_time_window(
        nominal_series,
        float(nominal_series["time"][0]),
        float(nominal_series["time"][-1]),
        rebase_time=True,
    )

    make_comparison_plot([armpc_selected, nominal_selected], output_path)


def save_summary_table(series_list: list[dict], output_path: str) -> str:
    if isinstance(series_list, dict):
        table_text = build_multi_comparison_summary_text(series_list)
    else:
        table_text = build_summary_text(series_list)
    with open(output_path, "w", encoding="ascii") as f:
        f.write(table_text)
    print(table_text, end="")
    print(f"Saved summary table to: {output_path}")
    return table_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare recorded/recomputed ZMP margin, CoP margin, and GRF effort for ARMPC vs nominal bags."
    )
    parser.add_argument(
        "--comparison-dir",
        default=DEFAULT_COMPARISON_DIR,
        help=f"Folder containing the comparison bags. Defaults to {DEFAULT_COMPARISON_DIR}",
    )
    parser.add_argument("--robot-mass", type=float, default=25.523)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--topic-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grf-contact-threshold", type=float, default=5.0)
    parser.add_argument("--denominator-epsilon", type=float, default=10.0)
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME)
    parser.add_argument(
        "--output",
        help="Optional output image path. Defaults to <comparison-dir>/zmp_cop_margin_comparison.png",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional summary text path. Defaults to <comparison-dir>/zmp_cop_margin_comparison_summary.txt",
    )
    args = parser.parse_args()

    bag_dirs = [os.path.join(args.comparison_dir, name) for name in DEFAULT_BAG_NAMES]
    missing = [bag_dir for bag_dir in bag_dirs if not os.path.isdir(bag_dir)]
    if missing:
        raise FileNotFoundError(f"Missing comparison bag directories: {missing}")

    full_series_list = [
        truncate_series(
            prepare_series(
                bag_dir=bag_dir,
                robot_mass=args.robot_mass,
                payload_mass=args.payload_mass,
                gravity=args.gravity,
                topic_contact_threshold=args.topic_contact_threshold,
                grf_contact_threshold=args.grf_contact_threshold,
                denominator_epsilon=args.denominator_epsilon,
            ),
            args.max_time,
        )
        for bag_dir in bag_dirs
    ]
    comparison_sets = select_comparison_sets(full_series_list)
    series_list = comparison_sets["best_statistics_equal_duration"]

    output_path = args.output or os.path.join(args.comparison_dir, "zmp_cop_margin_comparison.png")
    summary_output_path = args.summary_output or os.path.join(
        args.comparison_dir,
        "zmp_cop_margin_comparison_summary.txt",
    )
    make_comparison_plot(series_list, output_path)
    best_case_output_path = os.path.join(
        args.comparison_dir,
        "zmp_cop_margin_best_case_vs_nominal.png",
    )
    make_named_window_comparison_plot(
        full_series_list,
        best_case_output_path,
        armpc_window=(134.641685, 215.181163),
    )
    save_summary_table(comparison_sets, summary_output_path)
    print(f"Saved comparison plot to: {output_path}")
    print(f"Saved best-case plot to: {best_case_output_path}")


if __name__ == "__main__":
    main()
