#!/usr/bin/env python3
"""Fast postprocessing for ARMPC/Nominal pull-release and walk-in-place test bags.

Reports, per bag:
  - data quality: per-topic message counts, achieved rate, dropouts/gaps
  - ZMP margin stats (mean/min/pct below threshold), inside vs outside pull windows
  - auto-detected pull-release events from third-joint (index 2) deflection, with
    peak amplitude, peak estimated force, duration and timing for each -- the
    basis for checking pulls are amplitude-matched across trials/controllers
  - reference base velocity (/trajectory_generator com_vel) contamination check,
    since /joy itself isn't recorded

Usage:
    source /opt/ros/jazzy/setup.bash
    source ros2/msgs_ws/install/setup.bash
    python3 ros2/postprocess_bag.py <bag_dir> [--zmp-threshold 0.04] [--pull-threshold-deg 3.0]

Optionally pass --csv <path> to dump one row per detected pull event, to make it
easy to compare pull amplitude/duration across trials or ARMPC-vs-Nominal bags.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.signal import find_peaks
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ZMP_TOPIC = '/zmp_topic'
ARM_TOPIC = '/mpc_arm_infos'
TRAJ_TOPIC = '/trajectory_generator'
WANT_TOPICS = [ZMP_TOPIC, ARM_TOPIC, TRAJ_TOPIC]


def load_bag(bag_dir: str) -> dict:
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in WANT_TOPICS if t not in type_map]
    if missing:
        print(f"WARNING: topics not found in bag, skipping: {missing}", file=sys.stderr)
    msg_types = {name: get_message(type_map[name]) for name in WANT_TOPICS if name in type_map}

    raw = {name: {'t': [], 'msgs': []} for name in msg_types}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in msg_types:
            raw[topic]['t'].append(t)
            raw[topic]['msgs'].append(deserialize_message(data, msg_types[topic]))
    return raw


def report_data_quality(raw: dict, t0: int) -> None:
    print("=== Data quality ===")
    for name, d in raw.items():
        n = len(d['t'])
        if n < 2:
            print(f"  {name:22s} n={n} (too few messages to assess rate)")
            continue
        t = (np.asarray(d['t']) - t0) * 1e-9
        dt = np.diff(t)
        rate = 1.0 / np.median(dt) if np.median(dt) > 0 else float('nan')
        max_gap = dt.max()
        n_dropouts = int(np.sum(dt > 5 * np.median(dt)))
        print(f"  {name:22s} n={n:6d}  duration={t[-1]:7.2f}s  median_rate={rate:6.1f}Hz"
              f"  max_gap={max_gap*1000:7.1f}ms  gaps>5x_median={n_dropouts}")


def extract_zmp(raw: dict, t0: int) -> dict:
    d = raw.get(ZMP_TOPIC)
    if not d or not d['t']:
        return {}
    t = (np.asarray(d['t']) - t0) * 1e-9
    com_pos = np.array([[m.com_pos[0], m.com_pos[1], m.com_pos[2]] for m in d['msgs']])
    com_ori = np.array([[m.com_ori[0], m.com_ori[1], m.com_ori[2]] for m in d['msgs']])
    margin = np.array([m.zmp_margin[0] if len(m.zmp_margin) > 0 else np.nan for m in d['msgs']])
    contact = np.array([list(m.contact) for m in d['msgs']])
    return dict(t=t, com_pos=com_pos, com_ori=com_ori, margin=margin, contact=contact)


def extract_arm(raw: dict, t0: int) -> dict:
    d = raw.get(ARM_TOPIC)
    if not d or not d['t']:
        return {}
    t = (np.asarray(d['t']) - t0) * 1e-9
    qpos = np.array([list(m.passive_arm_joint_position) for m in d['msgs']])
    qpos0 = np.array([list(m.passive_arm_joint_position0) for m in d['msgs']])
    qvel = np.array([list(m.passive_arm_joint_velocity) for m in d['msgs']])
    eef = np.array([list(m.passive_arm_eef_position) for m in d['msgs']])
    wrench = np.array([list(m.passive_arm_external_wrenches) for m in d['msgs']])
    return dict(t=t, qpos=qpos, qpos0=qpos0, qvel=qvel, eef=eef, wrench=wrench)


def clean_force_mag(arm: dict, bound_N: float = 50.0) -> np.ndarray:
    """|estimated end-effector force|, with implausible spikes set to NaN.

    A hand pulling a ~1.5kg passive arm cannot plausibly sustain much more than a
    few tens of N; a standalone re-derivation of one such spike (same q, qdot,
    rest position, and current config gains fed through
    Passive_Arm_Interface.calculate_force_estimates_damping) reproduced ~27N,
    not the ~1850N actually recorded at that timestamp -- so this is not simply
    the estimator being noisy/ill-conditioned at large deflection, something in
    the live filtering/gain pipeline (wb_interface's wrench filter, or gains
    differing from current config.py) is amplifying it further. Root cause not
    fully pinned down yet; filtering by magnitude is a stopgap, not a fix.
    """
    force_mag = np.linalg.norm(arm['wrench'][:, 0:3], axis=1)
    clean = force_mag.copy()
    clean[force_mag > bound_N] = np.nan
    return clean


def extract_traj(raw: dict, t0: int) -> dict:
    d = raw.get(TRAJ_TOPIC)
    if not d or not d['t']:
        return {}
    t = (np.asarray(d['t']) - t0) * 1e-9
    lin = np.array([[m.com_vel.linear[0], m.com_vel.linear[1], m.com_vel.linear[2]] for m in d['msgs']])
    ang = np.array([[m.com_vel.angular[0], m.com_vel.angular[1], m.com_vel.angular[2]] for m in d['msgs']])
    return dict(t=t, lin=lin, ang=ang)


def clean_zmp_margin(zmp: dict, plausible_bound: float = 1.0) -> np.ndarray:
    """ZMP margin, with numerical-blowup samples set to NaN.

    The ZMP formula divides by (m*g - f_z); near-zero denominators during hard
    contact transitions produce spurious +-huge values that aren't physical
    (support polygon is O(0.3-0.5m)) and shouldn't dominate stats or a plot.
    """
    margin = zmp['margin']
    clean = margin.copy()
    clean[np.abs(margin) > plausible_bound] = np.nan
    return clean


def report_zmp(zmp: dict, threshold: float, plausible_bound: float = 1.0) -> None:
    if not zmp:
        print("=== ZMP margin: no data ===")
        return
    margin = zmp['margin']
    valid = ~np.isnan(margin)
    m = margin[valid]
    implausible = np.abs(m) > plausible_bound
    m_clean = m[~implausible]
    print("=== ZMP margin ===")
    print(f"  raw:   mean={np.nanmean(margin):.4f}  min={np.nanmin(margin):.4f}  max={np.nanmax(margin):.4f}")
    print(f"  {np.sum(implausible)}/{len(m)} samples ({100*np.mean(implausible):.2f}%) exceed "
          f"+-{plausible_bound}m -- treated as numerical outliers (near-zero-denominator ZMP blowups), excluded below")
    if len(m_clean) == 0:
        print("  WARNING: no plausible samples left after filtering.")
        return
    print(f"  filtered: mean={m_clean.mean():.4f}  median={np.median(m_clean):.4f}"
          f"  p5={np.percentile(m_clean,5):.4f}  p95={np.percentile(m_clean,95):.4f}  min={m_clean.min():.4f}")
    print(f"  pct_below_{threshold:.2f} (of filtered samples) = {100*np.mean(m_clean < threshold):.1f}%")


def report_ref_velocity(traj: dict) -> None:
    print("=== Reference base velocity (com_vel) contamination check ===")
    if not traj:
        print("  no /trajectory_generator data")
        return
    t, lin, ang = traj['t'], traj['lin'], traj['ang']
    nz = np.where(np.any(np.abs(lin) > 1e-6, axis=1) | np.any(np.abs(ang) > 1e-6, axis=1))[0]
    if len(nz) == 0:
        print("  ALL ZERO -- no commanded reference velocity for the whole bag.")
        return
    # group contiguous nonzero indices into windows
    breaks = np.where(np.diff(nz) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(nz) - 1]))
    print(f"  NONZERO reference velocity found in {len(starts)} window(s):")
    for s, e in zip(starts, ends):
        i0, i1 = nz[s], nz[e]
        peak_lin = np.abs(lin[i0:i1+1]).max(axis=0)
        peak_ang = np.abs(ang[i0:i1+1]).max(axis=0)
        print(f"    t=[{t[i0]:7.2f}, {t[i1]:7.2f}]s  dur={t[i1]-t[i0]:5.2f}s"
              f"  peak_lin(x,y,z)={np.round(peak_lin,3)}  peak_ang(x,y,z)={np.round(peak_ang,3)}")


def detect_fall_events(zmp: dict, roll_pitch_rate_threshold_deg_s: float = 60.0,
                        height_drop_threshold_m: float = 0.03, merge_gap_s: float = 1.0,
                        aftermath_window_s: float = 3.0) -> list:
    """Flag likely fall/tip-over events from base orientation and height.

    Signature validated against a real fall (pacc_comp_test_new_5_rep,
    t=28.5-29.2s): roll went from +20.5deg to -48.9deg in 0.7s (~98deg/s) while
    height dropped ~0.06m in the same window. Flags samples where roll or pitch
    changes faster than roll_pitch_rate_threshold_deg_s AND height drops by at
    least height_drop_threshold_m within the same short window -- rate alone can
    false-positive on fast recovery steps, so both signals are required together.

    This is a coarse, cheap signal meant to point you at timestamps to check
    against video, not a certified fall classifier -- always cross-check.
    """
    if not zmp:
        return []
    t = zmp['t']
    roll = np.degrees(zmp['com_ori'][:, 0])
    pitch = np.degrees(zmp['com_ori'][:, 1])
    z = zmp['com_pos'][:, 2]
    dt = np.diff(t)
    dt[dt <= 0] = np.nan

    roll_rate = np.abs(np.diff(roll)) / dt
    pitch_rate = np.abs(np.diff(pitch)) / dt
    fast = (roll_rate > roll_pitch_rate_threshold_deg_s) | (pitch_rate > roll_pitch_rate_threshold_deg_s)

    idxs = np.where(fast)[0]
    if len(idxs) == 0:
        return []

    # merge nearby flagged samples into discrete candidate events
    groups = []
    start = idxs[0]
    prev = idxs[0]
    for i in idxs[1:]:
        if t[i] - t[prev] > merge_gap_s:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))

    out = []
    for s, e in groups:
        # look at a short window after onset for the height drop / peak excursion
        w_end = np.searchsorted(t, t[s] + aftermath_window_s)
        w_end = min(w_end, len(t) - 1)
        seg = slice(s, w_end + 1)
        z_before = z[max(0, s - 5):s + 1]
        height_drop = float(z_before.mean() - z[seg].min()) if len(z_before) else float('nan')
        if height_drop < height_drop_threshold_m:
            continue  # fast orientation change without a height drop -- likely just a quick recovery step
        peak_i = s + int(np.argmax(np.abs(roll[seg] - roll[s])))
        out.append(dict(
            start_t=float(t[s]),
            peak_roll_rate_deg_s=float(np.nanmax(roll_rate[s:e + 1])) if e >= s else float('nan'),
            peak_pitch_rate_deg_s=float(np.nanmax(pitch_rate[s:e + 1])) if e >= s else float('nan'),
            roll_before_deg=float(roll[s]),
            roll_after_deg=float(roll[peak_i]),
            height_drop_m=height_drop,
        ))
    return out


def report_fall_events(falls: list) -> None:
    print("=== Candidate fall/tip-over events (roll/pitch rate + height drop, cross-check with video) ===")
    if not falls:
        print("  none detected")
        return
    print(f"  {len(falls)} event(s):")
    for i, f in enumerate(falls):
        print(f"  {i:3d} t={f['start_t']:7.2f}s  roll {f['roll_before_deg']:7.1f}->{f['roll_after_deg']:7.1f}deg"
              f"  peak_roll_rate={f['peak_roll_rate_deg_s']:7.1f}deg/s"
              f"  peak_pitch_rate={f['peak_pitch_rate_deg_s']:7.1f}deg/s"
              f"  height_drop={f['height_drop_m']:.3f}m")


def detect_pull_events(arm: dict, threshold_deg: float, min_gap_s: float, joint_idx: int = 2,
                        force_bound_N: float = 50.0) -> list:
    """Detect discrete pull-release events from deflection of `joint_idx` (default: third joint, 0-indexed 2)."""
    if not arm:
        return []
    t = arm['t']
    deflection_deg = np.degrees(arm['qpos'][:, joint_idx] - arm['qpos0'][:, joint_idx])
    above = np.abs(deflection_deg) > threshold_deg
    idxs = np.where(above)[0]
    if len(idxs) == 0:
        return []

    # merge crossings separated by less than min_gap_s into one event
    events = []
    start = idxs[0]
    prev = idxs[0]
    for i in idxs[1:]:
        if t[i] - t[prev] > min_gap_s:
            events.append((start, prev))
            start = i
        prev = i
    events.append((start, prev))

    angle_deg = np.degrees(arm['qpos'][:, joint_idx])
    force_clean = clean_force_mag(arm, force_bound_N)
    out = []
    for s, e in events:
        seg = slice(s, e + 1)
        peak_i = s + int(np.argmax(np.abs(deflection_deg[seg])))
        seg_force = force_clean[seg]
        valid = seg_force[~np.isnan(seg_force)]
        out.append(dict(
            start_t=float(t[s]), end_t=float(t[e]), duration=float(t[e] - t[s]),
            peak_deflection_deg=float(deflection_deg[peak_i]),
            peak_angle_deg=float(angle_deg[peak_i]),   # raw joint angle (== deflection only if q_arm0 is ~0)
            peak_time=float(t[peak_i]),
            peak_force_N=float(valid.max()) if len(valid) else float('nan'),
            mean_force_N=float(valid.mean()) if len(valid) else float('nan'),
            n_force_outliers=int(np.isnan(seg_force).sum()),
        ))
    return out


def count_individual_pulls(arm: dict, threshold_deg: float, min_distance_s: float = 0.3,
                            joint_idx: int = 2, force_bound_N: float = 50.0) -> list:
    """Find every individual pull peak (local extremum past threshold), independent of how
    close together they are -- unlike detect_pull_events this does NOT merge a dense run of
    pulls into one blob, so it correctly counts e.g. 30 separate pulls done back-to-back.
    Finds peaks on BOTH sides (positive and negative excursions) -- important for tests that
    deliberately pull both directions (e.g. left/right yaw), not just whichever side happens
    to dominate the sample count in this particular bag.
    """
    if not arm:
        return []
    t = arm['t']
    angle_deg = np.degrees(arm['qpos'][:, joint_idx])
    dt = np.median(np.diff(t))
    distance = max(1, int(round(min_distance_s / dt)))

    pos_idxs, _ = find_peaks(angle_deg, height=threshold_deg, distance=distance)
    neg_idxs, _ = find_peaks(-angle_deg, height=threshold_deg, distance=distance)
    peak_idxs = np.sort(np.concatenate([pos_idxs, neg_idxs]))

    force_clean = clean_force_mag(arm, force_bound_N)
    out = []
    for i in peak_idxs:
        # if the exact peak sample is itself an outlier, fall back to the nearest
        # plausible sample within a small window rather than reporting NaN
        f = force_clean[i]
        if np.isnan(f):
            lo, hi = max(0, i - 5), min(len(force_clean), i + 6)
            window = force_clean[lo:hi]
            valid = window[~np.isnan(window)]
            f = float(valid[np.argmin(np.abs(np.arange(lo, hi)[~np.isnan(window)] - i))]) if len(valid) else float('nan')
        out.append(dict(time=float(t[i]), angle_deg=float(angle_deg[i]), force_N=float(f)))
    return out


def report_individual_pulls(pulls: list, threshold_deg: float, joint_idx: int = 2) -> None:
    print(f"=== Individual pull count (local peaks past +-{threshold_deg} deg, joint{joint_idx+1}) ===")
    if not pulls:
        print("  none detected")
        return
    angles = np.array([p['angle_deg'] for p in pulls])
    print(f"  {len(pulls)} individual pulls detected")
    print(f"  peak angle: mean={angles.mean():.2f} deg  std={angles.std():.2f} deg"
          f"  (std/|mean|={100*angles.std()/abs(angles.mean()):.1f}%)"
          f"  min={angles.min():.2f}  max={angles.max():.2f} deg")
    # histogram-style bucket count so you can see the amplitude spread at a glance
    bins = [3, 10, 20, 30, 45, 60, 90]
    abs_angles = np.abs(angles)
    print("  amplitude distribution:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(np.sum((abs_angles >= lo) & (abs_angles < hi)))
        if n:
            print(f"    [{lo:3d},{hi:3d}) deg: {n:3d} pulls")
    n_over = int(np.sum(abs_angles >= bins[-1]))
    if n_over:
        print(f"    >={bins[-1]:3d} deg: {n_over:3d} pulls")

    forces = np.array([p['force_N'] for p in pulls])
    valid_f = forces[~np.isnan(forces)]
    if len(valid_f):
        print(f"  peak force (outlier-filtered): mean={valid_f.mean():.2f}N std={valid_f.std():.2f}N"
              f"  min={valid_f.min():.2f}N max={valid_f.max():.2f}N"
              f"  ({len(forces)-len(valid_f)}/{len(forces)} pulls had no plausible force sample nearby)")


PAPER_STYLE = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Nimbus Roman', 'Times New Roman'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'normal',
    'axes.linewidth': 1.0,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.6,
    'grid.color': 'gray',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.fontsize': 9,
    'lines.linewidth': 1.4,
}


def _panel_label(ax, letter: str) -> None:
    ax.text(0.02, 0.93, f'({letter})', transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='left', va='top', bbox=dict(boxstyle='square,pad=0.25', facecolor='white',
                                            edgecolor='black', linewidth=0.8))


def plot_joint_angle(arm: dict, events: list, threshold_deg: float, joint_idx: int, out_path: str,
                      pulls: list = None, force_bound_N: float = 50.0, zmp: dict = None,
                      zmp_threshold: float = 0.04, zmp_bound: float = 1.0, falls: list = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = arm['t']
    angle_rad = arm['qpos'][:, joint_idx]
    threshold_rad = np.radians(threshold_deg)
    wrench = arm['wrench']  # [Fx,Fy,Fz,Mx,My,Mz], world/base frame (see passive_arm_interface.py)
    force_mag = clean_force_mag(arm, force_bound_N)   # NaN'd where implausible -> shows as a gap, not a spike
    n_outliers = int(np.sum(np.linalg.norm(wrench[:, 0:3], axis=1) > force_bound_N))

    with plt.rc_context(PAPER_STYLE):
        n_rows = 3 if zmp else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)
        ax, ax2 = axes[0], axes[1]
        ax3 = axes[2] if zmp else None

        ax.plot(t, angle_rad, lw=1.6, color='tab:blue', label=f'joint{joint_idx+1} angle')
        ax.axhline(threshold_rad, color='gray', ls='--', lw=0.8)
        ax.axhline(-threshold_rad, color='gray', ls='--', lw=0.8, label=f'+-{threshold_deg} deg threshold')
        for i, ev in enumerate(events):
            ax.annotate(str(i), (ev['peak_time'], np.radians(ev['peak_angle_deg'])), fontsize=8, ha='center',
                        xytext=(0, 6), textcoords='offset points')
        if pulls:
            ax.scatter([p['time'] for p in pulls], [np.radians(p['angle_deg']) for p in pulls],
                       color='black', marker='x', s=18, zorder=5, label=f'{len(pulls)} individual pulls')
            ax2.scatter([p['time'] for p in pulls], [p['force_N'] for p in pulls],
                        color='black', marker='x', s=18, zorder=5, label=f'{len(pulls)} individual pulls')
        ax.set_ylabel(f'joint{joint_idx+1} [rad]')
        ax.set_title(f'{len(pulls or [])} individual pulls, {len(events)} pull-release events (numbered)',
                     fontsize=9, fontweight='normal', color='dimgray', loc='left')
        ax.legend(loc='upper right')
        _panel_label(ax, 'a')

        # Force subplot: components + magnitude. The estimated wrench is at the arm
        # end-effector, not a per-joint torque -- it's the interaction force the pull
        # produces there (see Passive_Arm_Interface.calculate_force_estimates_damping).
        # Samples where |F| > force_bound_N are NaN'd out (shows as a gap) rather than
        # plotted -- they're not physically plausible for a hand pull on this arm; see
        # clean_force_mag() docstring for what's known/not known about their cause.
        outlier_mask = np.isnan(force_mag)
        fx = np.where(outlier_mask, np.nan, wrench[:, 0])
        fy = np.where(outlier_mask, np.nan, wrench[:, 1])
        fz = np.where(outlier_mask, np.nan, wrench[:, 2])
        ax2.plot(t, fx, lw=0.8, color='tab:blue', alpha=0.8, label='Fx')
        ax2.plot(t, fy, lw=0.8, color='tab:orange', alpha=0.8, label='Fy')
        ax2.plot(t, fz, lw=0.8, color='tab:green', alpha=0.8, label='Fz')
        ax2.plot(t, force_mag, lw=1.6, color='black', label='|F|')
        ax2.set_ylabel('force [N]')
        ax2.set_title(f'{n_outliers} samples ({100*n_outliers/len(t):.2f}%) with |F|>{force_bound_N}N removed',
                      fontsize=9, fontweight='normal', color='dimgray', loc='left')
        ax2.legend(loc='upper right', ncol=5)
        _panel_label(ax2, 'b')

        # ZMP margin subplot, same outlier treatment as report_zmp (numerical
        # near-zero-denominator blowups NaN'd out rather than plotted).
        if ax3 is not None:
            zmp_margin = clean_zmp_margin(zmp, zmp_bound)
            n_zmp_outliers = int(np.sum(np.abs(zmp['margin']) > zmp_bound))
            ax3.plot(zmp['t'], zmp_margin, lw=1.2, color='tab:orange', label='ZMP margin')
            ax3.axhline(zmp_threshold, color='gray', ls='--', lw=0.8, label=f'{zmp_threshold}m threshold')
            ax3.axhline(0.0, color='black', lw=0.5)
            ax3.set_xlabel('t [s]')
            ax3.set_ylabel('ZMP margin [m]')
            ax3.set_ylim(0.0, 0.2)
            ax3.set_title(f'{n_zmp_outliers} samples ({100*n_zmp_outliers/len(zmp["t"]):.2f}%) with '
                          f'|margin|>{zmp_bound}m removed', fontsize=9, fontweight='normal', color='dimgray',
                          loc='left')
            ax3.legend(loc='upper right')
            _panel_label(ax3, 'c')
        else:
            ax2.set_xlabel('t [s]')

        # Candidate fall/tip-over events (see detect_fall_events) -- marked across
        # every panel since a fall is visible in angle, force, and ZMP margin alike.
        if falls:
            for i, fall in enumerate(falls):
                for a in axes:
                    a.axvline(fall['start_t'], color='red', ls='-', lw=1.4, alpha=0.8, zorder=6)
                axes[0].annotate(f"fall {i}", (fall['start_t'], axes[0].get_ylim()[1]), fontsize=8,
                                  color='red', fontweight='bold', ha='left', va='top', rotation=90)

        for a in axes:
            a.grid(True)

        # zoom x-axis to where anything actually happens, instead of the full
        # (often mostly-flat) recording window
        edge_times = [ev['start_t'] for ev in events] + [ev['end_t'] for ev in events] \
            + [p['time'] for p in (pulls or [])] + [f['start_t'] for f in (falls or [])]
        if edge_times:
            pad = 2.0
            ax.set_xlim(min(edge_times) - pad, max(edge_times) + pad)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def report_pull_events(events: list, threshold_deg: float, tap_max_duration_s: float = 2.0, joint_idx: int = 2) -> None:
    print(f"=== Pull-release events (|joint{joint_idx+1} deflection| > {threshold_deg} deg) ===")
    if not events:
        print("  none detected")
        return
    # A discrete "tap" test (matched-amplitude protocol) vs. an "extended" event
    # (sustained/repeated disturbance, e.g. random shaking) look very different in
    # duration -- classify separately so one doesn't pollute the other's stats.
    print(f"  {len(events)} event(s)  (kind: tap = single pull-release under {tap_max_duration_s}s,"
          f" extended = sustained/repeated disturbance)")
    print(f"  {'#':>3} {'kind':>9} {'start[s]':>9} {'dur[s]':>7} {'peak_angle[deg]':>16} {'peak_F[N]':>10} {'mean_F[N]':>10}")
    for i, ev in enumerate(events):
        kind = 'tap' if ev['duration'] < tap_max_duration_s else 'extended'
        print(f"  {i:3d} {kind:>9} {ev['start_t']:9.2f} {ev['duration']:7.2f} {ev['peak_angle_deg']:16.2f}"
              f" {ev['peak_force_N']:10.2f} {ev['mean_force_N']:10.2f}")

    taps = [ev for ev in events if ev['duration'] < tap_max_duration_s]
    extended = [ev for ev in events if ev['duration'] >= tap_max_duration_s]
    if taps:
        peaks = np.array([ev['peak_angle_deg'] for ev in taps])
        forces = np.array([ev['peak_force_N'] for ev in taps])
        print(f"  taps ({len(taps)}): peak angle mean={peaks.mean():.2f} deg std={peaks.std():.2f} deg"
              f" (std/|mean|={100*peaks.std()/abs(peaks.mean()):.1f}%)"
              f"  |  peak force mean={forces.mean():.2f}N std={forces.std():.2f}N"
              f"  <- amplitude-matching check")
    if extended:
        print(f"  extended ({len(extended)}): durations={[round(e['duration'],1) for e in extended]}s"
              f" -- inspect individually, not amplitude-matched by construction")


def write_csv(events: list, path: str, bag_dir: str) -> None:
    fieldnames = ['bag', 'event_idx', 'start_t', 'end_t', 'duration',
                  'peak_deflection_deg', 'peak_angle_deg', 'peak_time', 'peak_force_N', 'mean_force_N',
                  'n_force_outliers']
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for i, ev in enumerate(events):
            row = {'bag': os.path.basename(bag_dir.rstrip('/')), 'event_idx': i, **ev}
            w.writerow(row)
    print(f"Appended {len(events)} row(s) to {path}")


def write_falls_csv(falls: list, path: str, bag_dir: str) -> None:
    fieldnames = ['bag', 'fall_idx', 'start_t', 'peak_roll_rate_deg_s', 'peak_pitch_rate_deg_s',
                  'roll_before_deg', 'roll_after_deg', 'height_drop_m']
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for i, fall in enumerate(falls):
            row = {'bag': os.path.basename(bag_dir.rstrip('/')), 'fall_idx': i, **fall}
            w.writerow(row)
    print(f"Appended {len(falls)} row(s) to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag_dir')
    ap.add_argument('--zmp-threshold', type=float, default=0.04)
    ap.add_argument('--pull-threshold-deg', type=float, default=3.0)
    ap.add_argument('--pull-min-gap-s', type=float, default=0.5)
    ap.add_argument('--force-bound-N', type=float, default=50.0,
                     help='estimated |F| samples above this are treated as implausible outliers and excluded')
    ap.add_argument('--joint-idx', type=int, default=2, help='0=joint1(yaw), 1=joint2, 2=joint3(pitch/pendulum)')
    ap.add_argument('--csv', type=str, default=None, help='append per-event rows to this CSV')
    ap.add_argument('--plot', type=str, default=None, help='save a joint-angle-vs-time PNG to this path')
    ap.add_argument('--tap-max-duration-s', type=float, default=2.0,
                     help='events under this duration are classified "tap" (matched-amplitude protocol) '
                          'rather than "extended" (sustained/repeated disturbance) -- raise this for a '
                          'deliberate hold-then-release protocol (e.g. 8.0 for a 5s hold)')
    ap.add_argument('--fall-rate-threshold-deg-s', type=float, default=60.0,
                     help='roll/pitch rate above this (deg/s) is a candidate fall onset')
    ap.add_argument('--fall-height-drop-m', type=float, default=0.03,
                     help='minimum height drop required alongside the rate spike to flag a fall')
    ap.add_argument('--falls-csv', type=str, default=None, help='append per-fall-event rows to this CSV')
    args = ap.parse_args()

    raw = load_bag(args.bag_dir)
    if not raw or not any(d['t'] for d in raw.values()):
        print("No data loaded -- check bag path / topic names.", file=sys.stderr)
        sys.exit(1)

    t0 = min(min(d['t']) for d in raw.values() if d['t'])

    print(f"bag: {args.bag_dir}")
    report_data_quality(raw, t0)
    print()

    zmp = extract_zmp(raw, t0)
    arm = extract_arm(raw, t0)
    traj = extract_traj(raw, t0)

    report_zmp(zmp, args.zmp_threshold)
    print()
    report_ref_velocity(traj)
    print()

    falls = detect_fall_events(zmp, args.fall_rate_threshold_deg_s, args.fall_height_drop_m)
    report_fall_events(falls)
    print()
    if args.falls_csv and falls:
        write_falls_csv(falls, args.falls_csv, args.bag_dir)

    events = detect_pull_events(arm, args.pull_threshold_deg, args.pull_min_gap_s, args.joint_idx,
                                 force_bound_N=args.force_bound_N)
    report_pull_events(events, args.pull_threshold_deg, tap_max_duration_s=args.tap_max_duration_s,
                        joint_idx=args.joint_idx)
    print()

    pulls = count_individual_pulls(arm, args.pull_threshold_deg, joint_idx=args.joint_idx,
                                    force_bound_N=args.force_bound_N)
    report_individual_pulls(pulls, args.pull_threshold_deg, joint_idx=args.joint_idx)

    if args.csv and events:
        write_csv(events, args.csv, args.bag_dir)

    if args.plot and arm:
        plot_joint_angle(arm, events, args.pull_threshold_deg, args.joint_idx, args.plot, pulls=pulls,
                          force_bound_N=args.force_bound_N, zmp=zmp, zmp_threshold=args.zmp_threshold,
                          falls=falls)


if __name__ == '__main__':
    main()
