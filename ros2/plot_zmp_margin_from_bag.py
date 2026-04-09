from __future__ import annotations

import argparse
import io
import itertools
import os
import sqlite3
import struct
from collections import Counter
from glob import glob

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.spatial import ConvexHull  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    ConvexHull = None
    SCIPY_AVAILABLE = False


DEFAULT_BAG_PATH = (
    "/home/iit.local/gdessy/dls_ws_home/quadruped_pympc_framework/"
    "Quadruped-PyMPC/ros2/comparison_folder/comparison_load/"
    "test_armpc_3v2_aft_only4pulls"
)
ZMP_TOPIC_NAME = "/zmp_topic"
LEG_NAMES = ("FL", "FR", "RL", "RR")


def resolve_bag_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Bag directory not found: {path}")


def resolve_db3_path(bag_dir: str) -> str:
    candidates = sorted(glob(os.path.join(bag_dir, "*.db3")))
    if not candidates:
        raise FileNotFoundError(f"No .db3 file found in bag directory: {bag_dir}")
    return candidates[0]


def get_topic_id(cur: sqlite3.Cursor, topic_name: str) -> int | None:
    row = cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name,)).fetchone()
    return None if row is None else int(row[0])


def unpack_margin_array(blob: bytes, offset: int) -> np.ndarray:
    if len(blob) < offset + 8:
        raise ValueError("Blob is too short to contain the zmp_margin length.")

    count = struct.unpack_from("<Q", blob, offset)[0]
    offset += 8
    expected_size = offset + count * 8
    if len(blob) != expected_size:
        raise ValueError(
            f"Unexpected zmp_margin payload size: got {len(blob)} bytes, expected {expected_size}."
        )
    if count == 0:
        return np.empty(0, dtype=float)
    return np.asarray(struct.unpack_from("<" + "d" * count, blob, offset), dtype=float)


def decode_zmp_compute_msg(blob: bytes) -> dict:
    """
    Decode the ZmpComputeMsg payload.

    The bags in this workspace contain both an older 49-double layout and a newer
    52-double layout with `eef_pos_desired`. This decoder accepts both.
    """
    legacy_fixed_count = 49
    current_fixed_count = 52

    try:
        fixed = np.asarray(struct.unpack_from(f"<{current_fixed_count}d", blob, 4), dtype=float)
        zmp_margin = unpack_margin_array(blob, 4 + current_fixed_count * 8)
        return {
            "com_pos": fixed[0:3],
            "com_acc": fixed[3:6],
            "com_ori": fixed[6:9],
            "arm_wrenches": fixed[9:15],
            "eef_pos": fixed[15:18],
            "eef_pos_desired": fixed[18:21],
            "footholds": fixed[21:33],
            "contact": fixed[33:37],
            "nmpc_grfs": fixed[37:49],
            "zmp": fixed[49:52],
            "zmp_margin": zmp_margin,
        }
    except (struct.error, ValueError):
        fixed = np.asarray(struct.unpack_from(f"<{legacy_fixed_count}d", blob, 4), dtype=float)
        zmp_margin = unpack_margin_array(blob, 4 + legacy_fixed_count * 8)
        return {
            "com_pos": fixed[0:3],
            "com_acc": fixed[3:6],
            "com_ori": fixed[6:9],
            "arm_wrenches": fixed[9:15],
            "eef_pos": fixed[15:18],
            "eef_pos_desired": np.full(3, np.nan, dtype=float),
            "footholds": fixed[18:30],
            "contact": fixed[30:34],
            "nmpc_grfs": fixed[34:46],
            "zmp": fixed[46:49],
            "zmp_margin": zmp_margin,
        }


def load_series(bag_dir: str) -> dict:
    db3_path = resolve_db3_path(resolve_bag_dir(bag_dir))

    timestamps = []
    com_pos = []
    com_acc = []
    com_ori = []
    arm_wrenches = []
    eef_pos = []
    footholds = []
    contacts = []
    nmpc_grfs = []
    zmp_recorded = []
    margin_recorded = []

    with sqlite3.connect(db3_path) as con:
        cur = con.cursor()
        topic_id = get_topic_id(cur, ZMP_TOPIC_NAME)
        if topic_id is None:
            available = [name for (name,) in cur.execute("SELECT name FROM topics ORDER BY id").fetchall()]
            raise ValueError(
                f"Topic {ZMP_TOPIC_NAME!r} not found in rosbag. Available topics: {', '.join(available)}"
            )

        for timestamp_ns, data in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        ):
            msg = decode_zmp_compute_msg(data)
            timestamps.append(timestamp_ns)
            com_pos.append(np.asarray(msg["com_pos"], dtype=float))
            com_acc.append(np.asarray(msg["com_acc"], dtype=float))
            com_ori.append(np.asarray(msg["com_ori"], dtype=float))
            arm_wrenches.append(np.asarray(msg["arm_wrenches"], dtype=float))
            eef_pos.append(np.asarray(msg["eef_pos"], dtype=float))
            footholds.append(np.asarray(msg["footholds"], dtype=float).reshape(4, 3))
            contacts.append(np.asarray(msg["contact"], dtype=float))
            nmpc_grfs.append(np.asarray(msg["nmpc_grfs"], dtype=float).reshape(4, 3))
            zmp_recorded.append(np.asarray(msg["zmp"], dtype=float))
            margin_recorded.append(
                float(msg["zmp_margin"][0]) if len(msg["zmp_margin"]) > 0 else np.nan
            )

    if not timestamps:
        raise ValueError(f"No {ZMP_TOPIC_NAME} messages found in {db3_path}")

    timestamps = np.asarray(timestamps, dtype=np.int64)
    time = (timestamps - timestamps[0]) * 1e-9

    return {
        "bag_dir": bag_dir,
        "db3_path": db3_path,
        "timestamps_ns": timestamps,
        "time": time,
        "com_pos": np.asarray(com_pos, dtype=float),
        "com_acc": np.asarray(com_acc, dtype=float),
        "com_ori": np.asarray(com_ori, dtype=float),
        "arm_wrenches": np.asarray(arm_wrenches, dtype=float),
        "eef_pos": np.asarray(eef_pos, dtype=float),
        "footholds": np.asarray(footholds, dtype=float),
        "contact_raw": np.asarray(contacts, dtype=float),
        "nmpc_grfs": np.asarray(nmpc_grfs, dtype=float),
        "zmp_recorded": np.asarray(zmp_recorded, dtype=float),
        "margin_recorded": np.asarray(margin_recorded, dtype=float),
    }


def threshold_contacts(contact_raw: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(contact_raw, dtype=float) >= float(threshold)


def world_to_horizontal_frame_xy(points_xy: np.ndarray,
                                 base_xy: np.ndarray,
                                 yaw: float) -> np.ndarray:
    """
    Mirror the MPC support-constraint frame:
        p_H = h_R_w @ (p_W - base_W)
    where h_R_w depends only on yaw.
    """
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    h_R_w = np.array([[c, s], [-s, c]], dtype=float)
    shifted = np.asarray(points_xy, dtype=float) - np.asarray(base_xy, dtype=float)
    return shifted @ h_R_w.T


def compute_com_acc_from_grfs(nmpc_grfs: np.ndarray, total_mass: float, gravity: float) -> np.ndarray:
    """
    Rebuild CoM linear acceleration the same way the MPC centroidal model does:

        a_com = (sum_i f_i) / m + [0, 0, -g]

    assuming the GRFs in the rosbag are already expressed in the world frame.
    """
    nmpc_grfs = np.asarray(nmpc_grfs, dtype=float)
    total_force = np.sum(nmpc_grfs, axis=1)
    gravity_vec = np.array([0.0, 0.0, -float(gravity)], dtype=float)
    return total_force / float(total_mass) + gravity_vec


def infer_contacts_from_grfs(nmpc_grfs: np.ndarray, grf_z_threshold: float) -> np.ndarray:
    """
    Infer stance contacts directly from planned vertical GRFs.
    """
    nmpc_grfs = np.asarray(nmpc_grfs, dtype=float)
    return nmpc_grfs[:, :, 2] > float(grf_z_threshold)


def compute_cop_from_grfs(footholds_xyz: np.ndarray,
                          nmpc_grfs: np.ndarray,
                          contact_mask: np.ndarray,
                          force_threshold: float = 1e-6) -> np.ndarray:
    """
    Flat-ground CoP/ZMP proxy from footholds and vertical GRFs:

        cop_xy = sum_i (f_i,z * p_i,xy) / sum_i f_i,z

    This gives a support-consistent reference point derived only from the stance
    geometry and GRFs, useful to compare against the formula-based ZMP.
    """
    footholds_xyz = np.asarray(footholds_xyz, dtype=float)
    nmpc_grfs = np.asarray(nmpc_grfs, dtype=float)
    contact_mask = np.asarray(contact_mask, dtype=bool)

    cop = np.full((footholds_xyz.shape[0], 3), np.nan, dtype=float)
    for i in range(footholds_xyz.shape[0]):
        active = contact_mask[i]
        if not np.any(active):
            continue
        fz = np.asarray(nmpc_grfs[i, active, 2], dtype=float)
        positive = fz > float(force_threshold)
        if not np.any(positive):
            continue
        pts_xy = footholds_xyz[i, active, :2][positive]
        weights = fz[positive]
        total_weight = float(np.sum(weights))
        if total_weight <= force_threshold:
            continue
        cop_xy = np.sum(pts_xy * weights[:, None], axis=0) / total_weight
        cop[i, :2] = cop_xy
        cop[i, 2] = 0.0
    return cop


def monotonic_chain_convex_hull(points_xy: np.ndarray) -> np.ndarray:
    """
    Return convex hull vertices in CCW order using Andrew's monotonic chain.
    """
    if points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    rounded = np.round(points_xy, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    pts = np.asarray(points_xy[np.sort(unique_idx)], dtype=float)
    if pts.shape[0] <= 2:
        return pts

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if hull.shape[0] == 0:
        return pts[:1]
    return hull


def polygon_signed_area(polygon_xy: np.ndarray) -> float:
    if polygon_xy.shape[0] < 3:
        return 0.0
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def build_support_polygon(active_feet_xyz: np.ndarray) -> np.ndarray | None:
    """
    Build the stance support polygon from the active feet.

    Feet are projected to the horizontal plane and the convex hull is taken.
    The result is returned as CCW vertices in XY.
    """
    active_feet_xyz = np.asarray(active_feet_xyz, dtype=float)
    if active_feet_xyz.shape[0] < 3:
        return None

    points_xy = active_feet_xyz[:, :2]
    rounded = np.round(points_xy, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    points_xy = points_xy[np.sort(unique_idx)]
    if points_xy.shape[0] < 3:
        return None

    try:
        if SCIPY_AVAILABLE:
            hull = ConvexHull(points_xy)
            polygon = points_xy[hull.vertices]
        else:
            polygon = monotonic_chain_convex_hull(points_xy)
    except Exception:
        return None

    if polygon.shape[0] < 3:
        return None

    if abs(polygon_signed_area(polygon)) < 1e-10:
        return None

    if polygon_signed_area(polygon) < 0.0:
        polygon = polygon[::-1]
    return polygon


def point_to_segment_distance(point_xy: np.ndarray, seg_a_xy: np.ndarray, seg_b_xy: np.ndarray) -> float:
    ab = seg_b_xy - seg_a_xy
    ab_norm_sq = float(np.dot(ab, ab))
    if ab_norm_sq <= 1e-16:
        return float(np.linalg.norm(point_xy - seg_a_xy))

    t = float(np.dot(point_xy - seg_a_xy, ab) / ab_norm_sq)
    t = np.clip(t, 0.0, 1.0)
    projection = seg_a_xy + t * ab
    return float(np.linalg.norm(point_xy - projection))


def point_in_convex_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray, tol: float = 1e-9) -> bool:
    """
    A point is inside a CCW convex polygon if it lies to the left of every edge.
    """
    if polygon_xy.shape[0] < 3:
        return False

    for i in range(polygon_xy.shape[0]):
        a = polygon_xy[i]
        b = polygon_xy[(i + 1) % polygon_xy.shape[0]]
        edge = b - a
        rel = point_xy - a
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -tol:
            return False
    return True


def signed_distance_to_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    """
    Signed margin to a convex support polygon.

    The magnitude is the minimum Euclidean distance to any polygon edge.
    The sign is positive inside, zero on the boundary, negative outside.
    """
    if polygon_xy is None or polygon_xy.shape[0] < 3:
        return np.nan

    distances = [
        point_to_segment_distance(point_xy, polygon_xy[i], polygon_xy[(i + 1) % polygon_xy.shape[0]])
        for i in range(polygon_xy.shape[0])
    ]
    min_distance = float(np.min(distances))
    inside = point_in_convex_polygon(point_xy, polygon_xy)

    if min_distance <= 1e-9:
        return 0.0
    return min_distance if inside else -min_distance


def compute_zmp(com_pos: np.ndarray,
                com_acc: np.ndarray,
                eef_pos: np.ndarray,
                arm_wrenches: np.ndarray,
                total_mass: float,
                gravity: float,
                denominator_epsilon: float) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Flat-ground ZMP approximation in the world frame:

        x_zmp = (m g x_com - z_com m xddot_com + x_eef Fz - z_eef Fx) / (m g + Fz)
        y_zmp = (m g y_com - z_com m yddot_com + y_eef Fz - z_eef Fy) / (m g + Fz)

    with z-axis aligned with the ground normal n = [0, 0, 1].
    """
    com_pos = np.asarray(com_pos, dtype=float)
    com_acc = np.asarray(com_acc, dtype=float)
    eef_pos = np.asarray(eef_pos, dtype=float)
    arm_wrenches = np.asarray(arm_wrenches, dtype=float)

    fx = arm_wrenches[:, 0]
    fy = arm_wrenches[:, 1]
    fz = arm_wrenches[:, 2] 

    mg = total_mass * gravity
    safe_denominator = mg + fz
    denominator = safe_denominator
    # small_mask = np.abs(denominator) < denominator_epsilon
    # safe_denominator = denominator.copy()
    # safe_denominator[small_mask] = np.where(
    #     denominator[small_mask] >= 0.0,
    #     denominator_epsilon,
    #     -denominator_epsilon,
    # )

    x_num = mg * com_pos[:, 0] - com_pos[:, 2] * total_mass * com_acc[:, 0] 
    x_num += eef_pos[:, 0] * fz - eef_pos[:, 2] * fx

    y_num = mg * com_pos[:, 1] - com_pos[:, 2] * total_mass * com_acc[:, 1]
    y_num += eef_pos[:, 1] * fz - eef_pos[:, 2] * fy

    zmp_xy = np.column_stack((x_num / safe_denominator, y_num / safe_denominator))
    zmp = np.column_stack((zmp_xy, np.zeros(zmp_xy.shape[0], dtype=float)))
    return zmp, denominator


def compute_margin_series(zmp_xy_world: np.ndarray,
                          com_pos: np.ndarray,
                          com_ori: np.ndarray,
                          footholds_xyz: np.ndarray,
                          contacts_bool: np.ndarray) -> tuple[np.ndarray, list[np.ndarray | None], dict]:
    margins = np.full(zmp_xy_world.shape[0], np.nan, dtype=float)
    polygons: list[np.ndarray | None] = []

    insufficient_contact_count = 0
    hull_failure_count = 0
    degenerate_polygon_count = 0

    for i in range(zmp_xy_world.shape[0]):
        active_mask = contacts_bool[i]
        if np.count_nonzero(active_mask) < 3:
            insufficient_contact_count += 1
            polygons.append(None)
            continue

        base_xy = com_pos[i, :2]
        yaw = com_ori[i, 2]

        footholds_xy_h = world_to_horizontal_frame_xy(footholds_xyz[i, :, :2], base_xy, yaw)
        active_feet_h = np.column_stack((footholds_xy_h[active_mask], footholds_xyz[i, active_mask, 2]))
        polygon = build_support_polygon(active_feet_h)
        polygons.append(polygon)

        if polygon is None:
            if np.count_nonzero(active_mask) >= 3:
                hull_failure_count += 1
            continue

        if polygon.shape[0] < 3 or abs(polygon_signed_area(polygon)) < 1e-10:
            degenerate_polygon_count += 1
            continue

        zmp_xy_h = world_to_horizontal_frame_xy(zmp_xy_world[i:i + 1], base_xy, yaw)[0]
        margins[i] = signed_distance_to_polygon(zmp_xy_h, polygon)

    diagnostics = {
        "insufficient_contact_count": insufficient_contact_count,
        "hull_failure_count": hull_failure_count,
        "degenerate_polygon_count": degenerate_polygon_count,
        "nan_margin_count": int(np.count_nonzero(~np.isfinite(margins))),
    }
    return margins, polygons, diagnostics


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    return float(np.min(values[mask])), float(np.mean(values[mask])), float(np.max(values[mask]))


def format_contact_pattern(pattern: np.ndarray) -> str:
    return "".join("1" if bool(v) else "0" for v in pattern)


def choose_snapshot_indices(time: np.ndarray,
                            margins: np.ndarray,
                            arm_wrenches: np.ndarray,
                            max_snapshots: int) -> list[int]:
    candidates: list[int] = []

    finite_margin_idx = np.where(np.isfinite(margins))[0]
    if finite_margin_idx.size:
        margin_sorted = finite_margin_idx[np.argsort(margins[finite_margin_idx])]
        candidates.extend(margin_sorted[:max_snapshots].tolist())

    force_norm = np.linalg.norm(arm_wrenches[:, :3], axis=1)
    force_sorted = np.argsort(force_norm)[::-1]
    candidates.extend(force_sorted[:max_snapshots].tolist())

    if time.size:
        for fraction in (0.2, 0.5, 0.8):
            target = fraction * time[-1]
            idx = int(np.argmin(np.abs(time - target)))
            candidates.append(idx)

    selected: list[int] = []
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_snapshots:
            break
    return selected


def make_debug_plots(series: dict, output_path: str, include_denominator: bool = True) -> None:
    time = series["time"]
    contacts_bool = series["contacts_from_grfs"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True, constrained_layout=True)

    axes[0].plot(time, series["zmp_recomputed"][:, 0], label="ZMP x recomputed", linewidth=1.4)
    axes[0].plot(time, series["zmp_recorded"][:, 0], "--", label="ZMP x recorded", linewidth=1.0)
    if "zmp_from_grfs" in series:
        axes[0].plot(time, series["zmp_from_grfs"][:, 0], ":", label="ZMP x from GRFs", linewidth=1.2)
    axes[0].plot(time, series["zmp_recomputed"][:, 1], label="ZMP y recomputed", linewidth=1.4)
    axes[0].plot(time, series["zmp_recorded"][:, 1], "--", label="ZMP y recorded", linewidth=1.0)
    if "zmp_from_grfs" in series:
        axes[0].plot(time, series["zmp_from_grfs"][:, 1], ":", label="ZMP y from GRFs", linewidth=1.2)
    axes[0].set_ylabel("ZMP [m]")
    axes[0].set_title("Recomputed Vs Recorded ZMP")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", ncol=3)

    axes[1].plot(time, series["margin_recomputed"], label="Margin formula ZMP", linewidth=1.4)
    if "margin_cop_grf" in series:
        axes[1].plot(time, series["margin_cop_grf"], label="Margin CoP from GRFs", linewidth=1.2)
    if "margin_topic_contact" in series:
        axes[1].plot(time, series["margin_topic_contact"], "--", label="Margin from topic contacts", linewidth=1.0)
    if "margin_recorded" in series:
        axes[1].plot(time, series["margin_recorded"], ":", label="Margin recorded", linewidth=1.0)
    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[1].axhline(0.04, color="tab:gray", linewidth=0.8, alpha=0.6, linestyle=":")
    axes[1].set_ylabel("Margin [m]")
    axes[1].set_title("Signed ZMP Margin Comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # for leg_idx, leg_name in enumerate(LEG_NAMES):
    #     axes[2].plot(time, contacts_bool[:, leg_idx].astype(float), label=leg_name)
    # axes[2].set_ylabel("Contact")
    # axes[2].set_ylim(-0.1, 1.1)
    # axes[2].set_title("Thresholded Contact Pattern")
    # axes[2].grid(True, alpha=0.3)
    # axes[2].legend(loc="best", ncol=4)

    axes[2].plot(time, series["arm_wrenches"][:, 0], label="Fx")
    axes[2].plot(time, series["arm_wrenches"][:, 1], label="Fy")
    axes[2].plot(time, series["arm_wrenches"][:, 2], label="Fz")
    axes[2].set_ylabel("Force [N]")
    axes[2].set_title("External Force At End-Effector")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", ncol=3)
    axes[2].set_xlabel("Time [s]")

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_polygon_snapshot_plots(series: dict, output_path: str, snapshot_indices: list[int]) -> None:
    if not snapshot_indices:
        return

    ncols = 3
    nrows = int(np.ceil(len(snapshot_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    for ax, idx in zip(axes_array, snapshot_indices):
        polygon = series["support_polygons"][idx]
        base_xy = series["com_pos"][idx, :2]
        yaw = series["com_ori"][idx, 2]
        feet_xy = world_to_horizontal_frame_xy(series["footholds"][idx, :, :2], base_xy, yaw)
        active_mask = series["contacts_from_grfs"][idx]
        zmp_new = world_to_horizontal_frame_xy(series["zmp_recomputed"][idx:idx + 1, :2], base_xy, yaw)[0]
        zmp_old = world_to_horizontal_frame_xy(series["zmp_recorded"][idx:idx + 1, :2], base_xy, yaw)[0]
        zmp_grf = None
        if "zmp_from_grfs" in series and np.all(np.isfinite(series["zmp_from_grfs"][idx, :2])):
            zmp_grf = world_to_horizontal_frame_xy(series["zmp_from_grfs"][idx:idx + 1, :2], base_xy, yaw)[0]

        ax.scatter(feet_xy[:, 0], feet_xy[:, 1], c="lightgray", marker="x", s=50, label="All feet")
        ax.scatter(
            feet_xy[active_mask, 0],
            feet_xy[active_mask, 1],
            c="tab:blue",
            s=55,
            label="Active feet",
        )

        for leg_idx, leg_name in enumerate(LEG_NAMES):
            color = "tab:blue" if active_mask[leg_idx] else "gray"
            alpha = 1.0 if active_mask[leg_idx] else 0.7
            ax.annotate(
                leg_name,
                (feet_xy[leg_idx, 0], feet_xy[leg_idx, 1]),
                xytext=(6, 4),
                textcoords="offset points",
                color=color,
                alpha=alpha,
                fontsize=9,
            )

        if polygon is not None and polygon.shape[0] >= 3:
            polygon_closed = np.vstack((polygon, polygon[0]))
            ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], color="tab:green", linewidth=1.5, label="Hull")
            ax.fill(polygon_closed[:, 0], polygon_closed[:, 1], color="tab:green", alpha=0.12)

            active_leg_names = [LEG_NAMES[i] for i in np.where(active_mask)[0]]
            active_points = feet_xy[active_mask]
            for vertex in polygon:
                if active_points.shape[0] > 0:
                    match_idx = int(np.argmin(np.linalg.norm(active_points - vertex, axis=1)))
                    label = active_leg_names[match_idx]
                    ax.annotate(
                        f"H:{label}",
                        (vertex[0], vertex[1]),
                        xytext=(6, -10),
                        textcoords="offset points",
                        color="tab:green",
                        fontsize=8,
                    )

        ax.scatter(zmp_new[0], zmp_new[1], c="tab:red", s=70, label="ZMP recomputed")
        ax.scatter(zmp_old[0], zmp_old[1], c="tab:orange", s=70, marker="^", label="ZMP recorded")
        if zmp_grf is not None:
            ax.scatter(zmp_grf[0], zmp_grf[1], c="tab:purple", s=70, marker="s", label="ZMP from GRFs")
        ax.set_title(
            f"t={series['time'][idx]:.2f}s | grf={format_contact_pattern(active_mask)} | topic={format_contact_pattern(series['contacts_bool'][idx])}\n"
            f"margin_zmp={series['margin_recomputed'][idx]:.4f} m | margin_cop={series.get('margin_cop_grf', np.full_like(series['margin_recomputed'], np.nan))[idx]:.4f} m"
        )
        ax.set_xlabel("x_H [m]")
        ax.set_ylabel("y_H [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    for ax in axes_array[len(snapshot_indices):]:
        ax.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_all_polygon_config_plots(series: dict, output_path: str, sample_idx: int) -> None:
    """
    Plot all candidate support polygons that can be built from the four feet at one sample.

    This is useful to debug contact ordering / leg-assignment issues: if one alternative
    triangle makes physical sense while the nominal one does not, the contact semantics are
    likely wrong rather than the polygon geometry itself.
    """
    base_xy = series["com_pos"][sample_idx, :2]
    yaw = series["com_ori"][sample_idx, 2]
    feet_xy = world_to_horizontal_frame_xy(series["footholds"][sample_idx, :, :2], base_xy, yaw)
    zmp_new = world_to_horizontal_frame_xy(series["zmp_recomputed"][sample_idx:sample_idx + 1, :2], base_xy, yaw)[0]
    zmp_old = world_to_horizontal_frame_xy(series["zmp_recorded"][sample_idx:sample_idx + 1, :2], base_xy, yaw)[0]
    zmp_grf = None
    if "zmp_from_grfs" in series and np.all(np.isfinite(series["zmp_from_grfs"][sample_idx, :2])):
        zmp_grf = world_to_horizontal_frame_xy(series["zmp_from_grfs"][sample_idx:sample_idx + 1, :2], base_xy, yaw)[0]
    reported_pattern = format_contact_pattern(series["contacts_bool"][sample_idx])
    grf_pattern = format_contact_pattern(series["contacts_from_grfs"][sample_idx])

    candidates: list[tuple[str, tuple[int, ...]]] = []
    candidates.append(("All feet", (0, 1, 2, 3)))
    for combo in itertools.combinations(range(4), 3):
        label = ",".join(LEG_NAMES[i] for i in combo)
        candidates.append((label, combo))

    ncols = 3
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.0 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    for ax, (title, combo) in zip(axes_array, candidates):
        active_mask = np.zeros(4, dtype=bool)
        active_mask[list(combo)] = True
        active_points = feet_xy[active_mask]
        active_labels = [LEG_NAMES[i] for i in np.where(active_mask)[0]]
        active_xyz = np.column_stack((active_points, series["footholds"][sample_idx, active_mask, 2]))
        polygon = build_support_polygon(active_xyz)
        margin = signed_distance_to_polygon(zmp_new, polygon) if polygon is not None else np.nan

        ax.scatter(feet_xy[:, 0], feet_xy[:, 1], c="lightgray", marker="x", s=50, label="All feet")
        ax.scatter(active_points[:, 0], active_points[:, 1], c="tab:blue", s=55, label="Candidate feet")

        for leg_idx, leg_name in enumerate(LEG_NAMES):
            color = "tab:blue" if active_mask[leg_idx] else "gray"
            alpha = 1.0 if active_mask[leg_idx] else 0.7
            ax.annotate(
                leg_name,
                (feet_xy[leg_idx, 0], feet_xy[leg_idx, 1]),
                xytext=(6, 4),
                textcoords="offset points",
                color=color,
                alpha=alpha,
                fontsize=9,
            )

        if polygon is not None and polygon.shape[0] >= 3:
            polygon_closed = np.vstack((polygon, polygon[0]))
            ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], color="tab:green", linewidth=1.5, label="Hull")
            ax.fill(polygon_closed[:, 0], polygon_closed[:, 1], color="tab:green", alpha=0.12)

            for vertex in polygon:
                match_idx = int(np.argmin(np.linalg.norm(active_points - vertex, axis=1)))
                ax.annotate(
                    f"H:{active_labels[match_idx]}",
                    (vertex[0], vertex[1]),
                    xytext=(6, -10),
                    textcoords="offset points",
                    color="tab:green",
                    fontsize=8,
                )

        ax.scatter(zmp_new[0], zmp_new[1], c="tab:red", s=70, label="ZMP recomputed")
        ax.scatter(zmp_old[0], zmp_old[1], c="tab:orange", s=70, marker="^", label="ZMP recorded")
        if zmp_grf is not None:
            ax.scatter(zmp_grf[0], zmp_grf[1], c="tab:purple", s=70, marker="s", label="ZMP from GRFs")
        ax.set_title(
            f"{title}\ntopic={reported_pattern} | grf={grf_pattern} | margin={margin:.4f} m"
        )
        ax.set_xlabel("x_H [m]")
        ax.set_ylabel("y_H [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    for ax in axes_array[len(candidates):]:
        ax.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"All Candidate Support Polygons at t={series['time'][sample_idx]:.2f}s", fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_summary_text(series: dict, diagnostics: dict) -> str:
    buf = io.StringIO()

    time = series["time"]
    denominator = series["denominator"]
    zmp = series["zmp_recomputed"]
    margins = series["margin_recomputed"]
    cop_margins = series.get("margin_cop_grf", np.full_like(margins, np.nan))
    contacts_bool = series["contacts_from_grfs"]

    denom_min, denom_mean, denom_max = finite_stats(denominator)
    zmp_x_min, zmp_x_mean, zmp_x_max = finite_stats(zmp[:, 0])
    zmp_y_min, zmp_y_mean, zmp_y_max = finite_stats(zmp[:, 1])
    margin_min, margin_mean, margin_max = finite_stats(margins)
    cop_margin_min, cop_margin_mean, cop_margin_max = finite_stats(cop_margins)

    finite_margin = np.isfinite(margins)
    pct_margin_neg = 100.0 * np.mean(margins[finite_margin] < 0.0) if np.any(finite_margin) else np.nan
    pct_margin_low = 100.0 * np.mean(margins[finite_margin] < 0.04) if np.any(finite_margin) else np.nan

    patterns = Counter(format_contact_pattern(row) for row in contacts_bool)
    topic_patterns = Counter(format_contact_pattern(row) for row in series["contacts_bool"])

    print(f"Bag: {series['bag_dir']}", file=buf)
    print(f"Database: {series['db3_path']}", file=buf)
    print(
        f"Time range: {time[0]:.3f} s to {time[-1]:.3f} s "
        f"({time[-1] - time[0]:.3f} s, {time.size} samples)",
        file=buf,
    )
    print(f"Robot mass used: {series['robot_mass']:.6f} kg", file=buf)
    print(f"Payload mass used: {series['payload_mass']:.6f} kg", file=buf)
    print(f"Total mass used: {series['total_mass']:.6f} kg", file=buf)
    print(f"Contact threshold used: {series['contact_threshold']:.6f}", file=buf)
    print(f"GRF contact threshold used: {series['grf_contact_threshold']:.6f}", file=buf)
    print(f"Gravity magnitude used: {series['gravity']:.6f} m/s^2", file=buf)
    print(f"Denominator epsilon: {series['denominator_epsilon']:.6f}", file=buf)
    print("", file=buf)

    print(
        f"Denominator mg + Fz: min={denom_min:.6f}, mean={denom_mean:.6f}, max={denom_max:.6f}",
        file=buf,
    )
    print(f"Small-denominator warning count: {diagnostics['small_denominator_count']}", file=buf)
    print(
        f"Recomputed ZMP x stats: min={zmp_x_min:.6f}, mean={zmp_x_mean:.6f}, max={zmp_x_max:.6f}",
        file=buf,
    )
    print(
        f"Recomputed ZMP y stats: min={zmp_y_min:.6f}, mean={zmp_y_mean:.6f}, max={zmp_y_max:.6f}",
        file=buf,
    )
    print(
        f"Recomputed margin stats: min={margin_min:.6f}, mean={margin_mean:.6f}, max={margin_max:.6f}",
        file=buf,
    )
    print(
        f"CoP margin stats: min={cop_margin_min:.6f}, mean={cop_margin_mean:.6f}, max={cop_margin_max:.6f}",
        file=buf,
    )
    print(f"% margin < 0: {pct_margin_neg:.3f}", file=buf)
    print(f"% margin < 0.04: {pct_margin_low:.3f}", file=buf)
    print("", file=buf)

    print("Unique GRF-derived contact patterns and counts:", file=buf)
    for pattern, count in sorted(patterns.items()):
        print(f"  {pattern}: {count}", file=buf)
    print("", file=buf)

    print("Unique topic contact patterns and counts:", file=buf)
    for pattern, count in sorted(topic_patterns.items()):
        print(f"  {pattern}: {count}", file=buf)
    print("", file=buf)

    print(
        f"Samples with fewer than 3 active contacts: {diagnostics['insufficient_contact_count']}",
        file=buf,
    )
    print(f"Hull computation failures: {diagnostics['hull_failure_count']}", file=buf)
    print(f"Degenerate polygon count: {diagnostics['degenerate_polygon_count']}", file=buf)
    print(f"NaN margins: {diagnostics['nan_margin_count']}", file=buf)
    print("", file=buf)

    print(
        f"RMS error ZMP x: {rms_error(series['zmp_recomputed'][:, 0], series['zmp_recorded'][:, 0]):.6f}",
        file=buf,
    )
    print(
        f"RMS error ZMP y: {rms_error(series['zmp_recomputed'][:, 1], series['zmp_recorded'][:, 1]):.6f}",
        file=buf,
    )
    print(
        f"RMS error margin: {rms_error(series['margin_recomputed'], series['margin_recorded']):.6f}",
        file=buf,
    )
    if "margin_topic_contact" in series:
        print(
            f"RMS error margin (topic-contact polygon): {rms_error(series['margin_topic_contact'], series['margin_recorded']):.6f}",
            file=buf,
        )
    if "margin_cop_grf" in series:
        print(
            f"RMS error CoP margin (vs recorded margin): {rms_error(series['margin_cop_grf'], series['margin_recorded']):.6f}",
            file=buf,
        )

    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline ROS 2 bag post-processing for robust ZMP and signed ZMP margin reconstruction."
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        default=DEFAULT_BAG_PATH,
        help=f"Path to the rosbag directory. Defaults to {DEFAULT_BAG_PATH}",
    )
    parser.add_argument("--robot-mass", type=float, default=25.523, help="Robot mass in kg.")
    parser.add_argument("--payload-mass", type=float, default=0.0, help="Optional payload mass in kg.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude in m/s^2.")
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert contact values into booleans.",
    )
    parser.add_argument(
        "--denominator-epsilon",
        type=float,
        default=10.0,
        help="Minimum absolute denominator used for safe ZMP division.",
    )
    parser.add_argument(
        "--grf-contact-threshold",
        type=float,
        default=5.0,
        help="Vertical GRF threshold used to infer contacts from nmpc_grfs.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=6,
        help="Number of support-polygon snapshots to save in debug mode.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable XY support polygon snapshot plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the main debug figure interactively.",
    )
    args = parser.parse_args()

    bag_dir = resolve_bag_dir(args.bag_path)
    series = load_series(bag_dir)

    series["robot_mass"] = float(args.robot_mass)
    series["payload_mass"] = float(args.payload_mass)
    series["total_mass"] = float(args.robot_mass + args.payload_mass)
    series["gravity"] = float(args.gravity)
    series["contact_threshold"] = float(args.contact_threshold)
    series["denominator_epsilon"] = float(args.denominator_epsilon)
    series["grf_contact_threshold"] = float(args.grf_contact_threshold)

    series["contacts_bool"] = threshold_contacts(series["contact_raw"], args.contact_threshold)
    series["com_acc_from_grfs"] = compute_com_acc_from_grfs(
        nmpc_grfs=series["nmpc_grfs"],
        total_mass=series["total_mass"],
        gravity=series["gravity"],
    )
    series["contacts_from_grfs"] = infer_contacts_from_grfs(
        nmpc_grfs=series["nmpc_grfs"],
        grf_z_threshold=series["grf_contact_threshold"],
    )
    series["zmp_from_grfs"] = compute_cop_from_grfs(
        footholds_xyz=series["footholds"],
        nmpc_grfs=series["nmpc_grfs"],
        contact_mask=series["contacts_from_grfs"],
    )
    zmp_recomputed, denominator = compute_zmp(
        com_pos=series["com_pos"],
        com_acc=series["com_acc_from_grfs"],
        eef_pos=series["eef_pos"],
        arm_wrenches=series["arm_wrenches"],
        total_mass=series["total_mass"],
        gravity=series["gravity"],
        denominator_epsilon=series["denominator_epsilon"],
    )
    series["zmp_recomputed"] = zmp_recomputed
    series["denominator"] = denominator

    margin_topic_contact, support_polygons_topic, margin_diagnostics_topic = compute_margin_series(
        zmp_xy_world=zmp_recomputed[:, :2],
        com_pos=series["com_pos"],
        com_ori=series["com_ori"],
        footholds_xyz=series["footholds"],
        contacts_bool=series["contacts_bool"],
    )
    margin_recomputed, support_polygons, margin_diagnostics = compute_margin_series(
        zmp_xy_world=zmp_recomputed[:, :2],
        com_pos=series["com_pos"],
        com_ori=series["com_ori"],
        footholds_xyz=series["footholds"],
        contacts_bool=series["contacts_from_grfs"],
    )
    margin_cop_grf, _, _ = compute_margin_series(
        zmp_xy_world=series["zmp_from_grfs"][:, :2],
        com_pos=series["com_pos"],
        com_ori=series["com_ori"],
        footholds_xyz=series["footholds"],
        contacts_bool=series["contacts_from_grfs"],
    )
    series["margin_topic_contact"] = margin_topic_contact
    series["support_polygons_topic"] = support_polygons_topic
    series["margin_recomputed"] = margin_recomputed
    series["margin_cop_grf"] = margin_cop_grf
    series["support_polygons"] = support_polygons

    diagnostics = {
        "small_denominator_count": 0,
        **margin_diagnostics,
    }

    main_plot_path = os.path.join(bag_dir, "zmp_recomputed_debug.png")
    snapshot_plot_path = os.path.join(bag_dir, "zmp_polygon_snapshots.png")
    summary_path = os.path.join(bag_dir, "zmp_recomputed_summary.txt")

    make_debug_plots(series, main_plot_path, include_denominator=True)

    if not args.no_snapshots:
        snapshot_indices = choose_snapshot_indices(
            time=series["time"],
            margins=series["margin_recomputed"],
            arm_wrenches=series["arm_wrenches"],
            max_snapshots=max(1, int(args.snapshot_count)),
        )
        make_polygon_snapshot_plots(series, snapshot_plot_path, snapshot_indices)

    summary_text = build_summary_text(series, diagnostics)
    print(summary_text, end="")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    if args.show:
        fig = plt.figure(figsize=(1, 1))
        plt.close(fig)
        img = plt.imread(main_plot_path)
        plt.figure(figsize=(14, 12))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
