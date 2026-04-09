from __future__ import annotations

import argparse
import os
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import plot_zmp_margin_from_bag as zmp_post


DEFAULT_COMPARISON_DIR = (
    "/home/iit.local/gdessy/dls_ws_home/quadruped_pympc_framework/"
    "Quadruped-PyMPC/ros2/comparison_folder/comparison_load"
)
DEFAULT_BAG_NAMES = (
    "test_armpc_3v2_aft_only4pulls_load",
    "test_nominal_1_aft_load",
)
ZMP_THRESHOLD = 0.04


def infer_label_from_bag_dir(bag_dir: str) -> str:
    name = os.path.basename(os.path.normpath(bag_dir)).lower()
    if "armpc" in name or "arm_pc" in name:
        return "ARMPC"
    if "nominal" in name:
        return "NOMINAL"
    return os.path.basename(os.path.normpath(bag_dir))


def compute_horizontal_grf_effort(grfs: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Compute horizontal GRF effort metrics from GRFs shaped (N, 12) or (N, 4, 3)."""
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


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(values[valid] ** 2)))


def make_summary_rows(series_list: Iterable[dict]) -> list[dict]:
    rows = []
    for series in series_list:
        force_xyz = np.asarray(series["arm_wrenches"][:, :3], dtype=float)
        force_norm = np.linalg.norm(force_xyz, axis=1)
        grf_effort = compute_horizontal_grf_effort(series["nmpc_grfs"])

        row = {
            "label": series["label"],
            "duration_s": float(series["time"][-1] - series["time"][0]) if len(series["time"]) > 1 else 0.0,
            "zmp_min": finite_stats(series["margin_zmp"])["min"],
            "zmp_mean": finite_stats(series["margin_zmp"])["mean"],
            "cop_min": finite_stats(series["margin_cop"])["min"],
            "cop_mean": finite_stats(series["margin_cop"])["mean"],
            "pct_zmp_neg": percent_below(series["margin_zmp"], 0.0),
            "pct_cop_neg": percent_below(series["margin_cop"], 0.0),
            "pct_zmp_below_004": percent_below(series["margin_zmp"], ZMP_THRESHOLD),
            "pct_cop_below_004": percent_below(series["margin_cop"], ZMP_THRESHOLD),
            "fx_rms": rms(force_xyz[:, 0]),
            "fy_rms": rms(force_xyz[:, 1]),
            "fz_rms": rms(force_xyz[:, 2]),
            "force_norm_peak": float(np.max(force_norm)) if force_norm.size else float("nan"),
            "grf_effort_mean": grf_effort["summary_h"]["mean"],
            "grf_effort_rms": grf_effort["summary_h"]["rms"],
            "grf_effort_peak": grf_effort["summary_h"]["peak"],
            "grf_effort_net_mean": grf_effort["summary_h_net"]["mean"],
            "grf_effort_net_peak": grf_effort["summary_h_net"]["peak"],
        }
        rows.append(row)
    return rows


def format_summary_table(rows: list[dict]) -> str:
    headers = [
        "label",
        "duration_s",
        "zmp_min",
        "zmp_mean",
        "cop_min",
        "cop_mean",
        "pct_zmp_neg",
        "pct_cop_neg",
        "pct_zmp_below_004",
        "pct_cop_below_004",
        "fx_rms",
        "fy_rms",
        "fz_rms",
        "force_norm_peak",
        "grf_effort_mean",
        "grf_effort_rms",
        "grf_effort_peak",
        "grf_effort_net_mean",
        "grf_effort_net_peak",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        fields = []
        for key in headers:
            value = row[key]
            if isinstance(value, str):
                fields.append(value)
            else:
                fields.append(f"{value:.6f}")
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


def prepare_series(bag_dir: str,
                   robot_mass: float,
                   payload_mass: float,
                   gravity: float,
                   topic_contact_threshold: float,
                   grf_contact_threshold: float,
                   denominator_epsilon: float) -> dict:
    series = zmp_post.load_series(bag_dir)
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
    return series


def make_comparison_plot(series_list: list[dict], output_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False, constrained_layout=True)

    colors = {
        "ARMPC": "tab:blue",
        "NOMINAL": "tab:orange",
    }

    for series in series_list:
        label = series["label"]
        color = colors.get(label, None)
        time = series["time"]

        axes[0].plot(time, series["margin_zmp"], label=label, linewidth=1.6, color=color)
        axes[1].plot(time, series["margin_cop"], label=label, linewidth=1.6, color=color)

        axes[2].plot(time, series["arm_wrenches"][:, 0], linewidth=1.2, color=color, label=f"{label} Fx")
        axes[2].plot(time, series["arm_wrenches"][:, 1], linewidth=1.2, color=color, linestyle="--", label=f"{label} Fy")
        axes[2].plot(time, series["arm_wrenches"][:, 2], linewidth=1.2, color=color, linestyle=":", label=f"{label} Fz")

    axes[0].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[0].axhline(0.04, color="tab:gray", linewidth=0.8, alpha=0.7, linestyle=":")
    axes[0].set_ylabel("Margin [m]")
    axes[0].set_title("ZMP Margin")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[1].axhline(0.04, color="tab:gray", linewidth=0.8, alpha=0.7, linestyle=":")
    axes[1].set_ylabel("Margin [m]")
    axes[1].set_title("CoP Margin")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].set_ylabel("Force [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("External Forces")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", ncol=2)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_summary_table(series_list: list[dict], output_path: str) -> str:
    rows = make_summary_rows(series_list)
    table_text = format_summary_table(rows)
    with open(output_path, "w", encoding="ascii") as f:
        f.write(table_text)
    print(table_text, end="")
    print(f"Saved summary table to: {output_path}")
    return table_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ZMP margin, CoP margin, and external forces for ARMPC vs nominal bags."
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

    series_list = [
        prepare_series(
            bag_dir=bag_dir,
            robot_mass=args.robot_mass,
            payload_mass=args.payload_mass,
            gravity=args.gravity,
            topic_contact_threshold=args.topic_contact_threshold,
            grf_contact_threshold=args.grf_contact_threshold,
            denominator_epsilon=args.denominator_epsilon,
        )
        for bag_dir in bag_dirs
    ]

    output_path = args.output or os.path.join(args.comparison_dir, "zmp_cop_margin_comparison.png")
    summary_output_path = args.summary_output or os.path.join(
        args.comparison_dir,
        "zmp_cop_margin_comparison_summary.txt",
    )
    make_comparison_plot(series_list, output_path)
    save_summary_table(series_list, summary_output_path)
    print(f"Saved comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
