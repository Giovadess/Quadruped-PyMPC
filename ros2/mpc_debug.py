       
''' # Saving for clarity a bunch of variables
        foot_velocity_fl = Ins[0:3]
        foot_velocity_fr = Ins[3:6]
        foot_velocity_rl = Ins[6:9]
        foot_velocity_rr = Ins[9:12]
        foot_force_fl = Ins[12:15]
        foot_force_fr = Ins[15:18]
        foot_force_rl = Ins[18:21]
        foot_force_rr = Ins[21:24]

        com_position = states[0:3]
        com_velocity = states[3:6]
        foot_position_fl = states[12:15]
        foot_position_fr = states[15:18]
        foot_position_rl = states[18:21]
        foot_position_rr = states[21:24]

        q_arm= states[30:33] #arm joint position
        q_dot_arm = states[33:36] #arm joint velocity'''
'''
Constraints shape:
shape friction constr:  (20, 1)
shape stability constr:  (6, 1)

'''
### Code for debugging the predicted MPC States


import json
import numpy as np
import matplotlib.pyplot as plt

with open("mpc_log_armpc.json", "r") as f:
    log = json.load(f)
qp_fail_steps = [i for i, s in enumerate(log) if s.get("solver_status") == 4]
first_qp_fail_step = qp_fail_steps[0] if qp_fail_steps else None

print("QP fail steps (solver_status == 4):", qp_fail_steps)
print("First QP fail step:", first_qp_fail_step)

step = log[-1]
## print the qp_status
print("QP status:", step["solver_status"])
x_pred = np.array(step["x_pred"])   # (N+1, nx)
u_pred = np.array(step["u_pred"])   # (N, nu)
state_current = np.array(step.get("state_current", []))
upper_bound = np.array(step.get("upper_bound", []))
lower_bound = np.array(step.get("lower_bound", []))
contact_sequence = np.array(step.get("contact_sequence", []))
# Helper: line + dot marker for each node
def plot_with_nodes(y, label, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(y, marker="o", markersize=4, linewidth=1.5, label=label)
    return ax

# Helper: 3-axis plot
def plot_xyz(series_xyz, title, ylabel):
    plt.figure()
    plot_with_nodes(series_xyz[:, 0], "x")
    plot_with_nodes(series_xyz[:, 1], "y")
    plot_with_nodes(series_xyz[:, 2], "z")
    plt.xlabel("Horizon index i")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Helper: 3-axis plot into provided axes (subplots)
def plot_xyz_subplot(ax, series_xyz, title, ylabel):
    ax.plot(series_xyz[:, 0], marker="o", markersize=3, linewidth=1.2, label="x")
    ax.plot(series_xyz[:, 1], marker="o", markersize=3, linewidth=1.2, label="y")
    ax.plot(series_xyz[:, 2], marker="o", markersize=3, linewidth=1.2, label="z")
    ax.set_title(title)
    ax.set_xlabel("Horizon index i")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

# Helper: build state labels based on the nominal state definition
def build_state_labels(integral_error_size=6):
    labels = [
        "pos_x", "pos_y", "pos_z",
        "lin_vel_x", "lin_vel_y", "lin_vel_z",
        "ori_roll", "ori_pitch", "ori_yaw",
        "ang_vel_x", "ang_vel_y", "ang_vel_z",
        "foot_FL_x", "foot_FL_y", "foot_FL_z",
        "foot_FR_x", "foot_FR_y", "foot_FR_z",
        "foot_RL_x", "foot_RL_y", "foot_RL_z",
        "foot_RR_x", "foot_RR_y", "foot_RR_z",
    ]
    labels += [f"int_err_{i}" for i in range(integral_error_size)]
    labels += ["arm_j1", "arm_j2", "arm_j3",
               "arm_j1_dot", "arm_j2_dot", "arm_j3_dot"]
    return labels

# Helper: plot state_current grouped into subplots
def plot_state_current(state_current, integral_error_size=6):
    if state_current.size == 0:
        print("Skipping state_current plot: missing state_current in log.")
        return
    expected_len = 24 + integral_error_size + 6
    if len(state_current) != expected_len:
        print(f"state_current size ({len(state_current)}) does not match expected ({expected_len}); "
              "skipping grouped state_current plot.")
        return

    idx = 0
    pos = state_current[idx:idx+3]; idx += 3
    lin_vel = state_current[idx:idx+3]; idx += 3
    ori = state_current[idx:idx+3]; idx += 3
    ang_vel = state_current[idx:idx+3]; idx += 3
    foot_fl = state_current[idx:idx+3]; idx += 3
    foot_fr = state_current[idx:idx+3]; idx += 3
    foot_rl = state_current[idx:idx+3]; idx += 3
    foot_rr = state_current[idx:idx+3]; idx += 3
    int_err = state_current[idx:idx+integral_error_size]; idx += integral_error_size
    arm_pos = state_current[idx:idx+3]; idx += 3
    arm_vel = state_current[idx:idx+3]; idx += 3

    fig_sc, axs_sc = plt.subplots(4, 2, figsize=(12, 10))
    plot_xyz_subplot(axs_sc[0, 0], pos.reshape(1, 3), "state_current: position", "value")
    plot_xyz_subplot(axs_sc[0, 1], lin_vel.reshape(1, 3), "state_current: linear_velocity", "value")
    plot_xyz_subplot(axs_sc[1, 0], ori.reshape(1, 3), "state_current: orientation", "value")
    plot_xyz_subplot(axs_sc[1, 1], ang_vel.reshape(1, 3), "state_current: angular_velocity", "value")
    plot_xyz_subplot(axs_sc[2, 0], foot_fl.reshape(1, 3), "state_current: foot_FL", "value")
    plot_xyz_subplot(axs_sc[2, 1], foot_fr.reshape(1, 3), "state_current: foot_FR", "value")
    plot_xyz_subplot(axs_sc[3, 0], foot_rl.reshape(1, 3), "state_current: foot_RL", "value")
    plot_xyz_subplot(axs_sc[3, 1], foot_rr.reshape(1, 3), "state_current: foot_RR", "value")
    fig_sc.suptitle("state_current grouped")
    fig_sc.tight_layout()
    plt.show()

    fig_sc2, axs_sc2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs_sc2[0].plot(np.arange(len(int_err)), int_err, marker="o", linewidth=1.2)
    axs_sc2[0].set_title("state_current: integral_errors")
    axs_sc2[0].set_ylabel("value")
    axs_sc2[0].grid(True)

    axs_sc2[1].plot(np.arange(len(arm_pos)), arm_pos, marker="o", linewidth=1.2, label="arm_joint_pos")
    axs_sc2[1].plot(np.arange(len(arm_vel)), arm_vel, marker="o", linewidth=1.2, label="arm_joint_vel")
    axs_sc2[1].set_title("state_current: arm joints")
    axs_sc2[1].set_xlabel("index")
    axs_sc2[1].set_ylabel("value")
    axs_sc2[1].grid(True)
    axs_sc2[1].legend()

    fig_sc2.suptitle("state_current additional groups")
    fig_sc2.tight_layout()
    plt.show()

# Helper: plot constraint bounds (last 6 constraints) in subplots
def plot_constraint_bounds(lower_bound, upper_bound, title):
    if lower_bound.size == 0 or upper_bound.size == 0:
        print("Skipping constraint plot: missing lower_bound/upper_bound in log.")
        return
    n = 6 if min(len(lower_bound), len(upper_bound)) >= 6 else min(len(lower_bound), len(upper_bound))
    if n == 0:
        print("Skipping constraint plot: empty bounds.")
        return
    lower_slice = lower_bound[-n:]
    upper_slice = upper_bound[-n:]

    idx = np.arange(n)
    fig_b, axs_b = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs_b[0].plot(idx, lower_slice, marker="o", linewidth=1.5)
    axs_b[0].set_title("Lower bound (last 6)")
    axs_b[0].set_ylabel("Value")
    axs_b[0].grid(True)

    axs_b[1].plot(idx, upper_slice, marker="o", linewidth=1.5)
    axs_b[1].set_title("Upper bound (last 6)")
    axs_b[1].set_xlabel("Constraint index (0-5)")
    axs_b[1].set_ylabel("Value")
    axs_b[1].grid(True)
    ##add legend
    axs_b[1].legend()
    axs_b[0].legend()

    fig_b.suptitle(title)
    fig_b.tight_layout()
    plt.show()




# Ins (u_pred)

foot_force_fl = u_pred[:, 12:15]
foot_force_fr = u_pred[:, 15:18]
foot_force_rl = u_pred[:, 18:21]
foot_force_rr = u_pred[:, 21:24]

# plot_xyz(foot_velocity_fl, "In u: Foot velocity FL (nodes marked)", "Foot velocity [m/s]")
# plot_xyz(foot_velocity_fr, "In u: Foot velocity FR (nodes marked)", "Foot velocity [m/s]")
# plot_xyz(foot_velocity_rl, "In u: Foot velocity RL (nodes marked)", "Foot velocity [m/s]")
# plot_xyz(foot_velocity_rr, "In u: Foot velocity RR (nodes marked)", "Foot velocity [m/s]")

fig_u, axs_u = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
plot_xyz_subplot(axs_u[0, 0], foot_force_fl, "In u: Foot force FL", "Foot force [N]")
plot_xyz_subplot(axs_u[0, 1], foot_force_fr, "In u: Foot force FR", "Foot force [N]")
plot_xyz_subplot(axs_u[1, 0], foot_force_rl, "In u: Foot force RL", "Foot force [N]")
plot_xyz_subplot(axs_u[1, 1], foot_force_rr, "In u: Foot force RR", "Foot force [N]")
fig_u.suptitle("Inputs u: Foot forces")
fig_u.tight_layout()
plt.show()

# Current state values (named)
plot_state_current(state_current)

# Current constraint values vs bounds (last 6 constraints)
plot_constraint_bounds(lower_bound, upper_bound,
                       "Constraint bounds (last 6)")

# States (x_pred)
com_position = x_pred[:, 0:3]
com_velocity = x_pred[:, 3:6]
foot_position_fl = x_pred[:, 12:15]
foot_position_fr = x_pred[:, 15:18]
foot_position_rl = x_pred[:, 18:21]
foot_position_rr = x_pred[:, 21:24]
q_arm = x_pred[:, 30:33]
q_dot_arm = x_pred[:, 33:36]

fig_x, axs_x = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
plot_xyz_subplot(axs_x[0, 0], com_position, "State x: CoM position", "CoM position [m]")
plot_xyz_subplot(axs_x[0, 1], com_velocity, "State x: CoM velocity", "CoM velocity [m/s]")
plot_xyz_subplot(axs_x[1, 0], foot_position_fl, "State x: Foot position FL", "Foot position [m]")
plot_xyz_subplot(axs_x[1, 1], foot_position_fr, "State x: Foot position FR", "Foot position [m]")
plot_xyz_subplot(axs_x[2, 0], foot_position_rl, "State x: Foot position RL", "Foot position [m]")
plot_xyz_subplot(axs_x[2, 1], foot_position_rr, "State x: Foot position RR", "Foot position [m]")
fig_x.suptitle("States x: Base and foot positions/velocities")
fig_x.tight_layout()
plt.show()

fig_arm, axs_arm = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs_arm[0].plot(q_arm[:, 0], marker="o", markersize=3, linewidth=1.2, label="arm joint 1")
axs_arm[0].plot(q_arm[:, 1], marker="o", markersize=3, linewidth=1.2, label="arm joint 2")
axs_arm[0].plot(q_arm[:, 2], marker="o", markersize=3, linewidth=1.2, label="arm joint 3")
axs_arm[0].set_ylabel("Arm joint position [rad]")
axs_arm[0].set_title("State x: Arm joint position")
axs_arm[0].grid(True)
axs_arm[0].legend()

axs_arm[1].plot(q_dot_arm[:, 0], marker="o", markersize=3, linewidth=1.2, label="arm joint 1 dot")
axs_arm[1].plot(q_dot_arm[:, 1], marker="o", markersize=3, linewidth=1.2, label="arm joint 2 dot")
axs_arm[1].plot(q_dot_arm[:, 2], marker="o", markersize=3, linewidth=1.2, label="arm joint 3 dot")
axs_arm[1].set_xlabel("Horizon index i")
axs_arm[1].set_ylabel("Arm joint velocity [rad/s]")
axs_arm[1].set_title("State x: Arm joint velocity")
axs_arm[1].grid(True)
axs_arm[1].legend()

fig_arm.suptitle("States x: Arm joints")
fig_arm.tight_layout()
plt.show()





def compute_zmp_constr(u_step, x_step, stance, m=25.523):
    FL, FR, RL, RR = stance #stance conditions

    # state
    base_w = x_step[0:3]
    yaw    = x_step[8]

    # feet positions in world frame
    FLp = x_step[12:15]
    FRp = x_step[15:18]
    RLp = x_step[18:21]
    RRp = x_step[21:24]

    # rotation
    h_R_w = np.array([[np.cos(yaw),  np.sin(yaw)],
                      [-np.sin(yaw), np.cos(yaw)]])

    # feet positions in body frame according to MPC convention
    FL_xy = h_R_w @ (FLp[0:2] - base_w[0:2])
    FR_xy = h_R_w @ (FRp[0:2] - base_w[0:2])
    RL_xy = h_R_w @ (RLp[0:2] - base_w[0:2])
    RR_xy = h_R_w @ (RRp[0:2] - base_w[0:2])

    # Ground reaction forces
    f = (
        u_step[12:15] * FL +
        u_step[15:18] * FR +
        u_step[18:21] * RL +
        u_step[21:24] * RR
    )

    gravity = np.array([0, 0, -9.81])
    a_com = f / m + gravity

    denom = m * (-gravity[2])

    zmp = np.array([
        (m * (-gravity[2]) * base_w[0] - base_w[2] * m * a_com[0]) / denom,
        (m * (-gravity[2]) * base_w[1] - base_w[2] * m * a_com[1]) / denom
    ])

    zmp_xy = h_R_w @ (zmp - base_w[0:2])
    x, y = zmp_xy

    # constraints
    c = np.array([
        x - (FR_xy[0] - FL_xy[0]) * (y - FL_xy[1]) / (FR_xy[1] - FL_xy[1] ) - FL_xy[0],
        y - (RR_xy[1] - FR_xy[1]) * (x - FR_xy[0]) / (RR_xy[0] - FR_xy[0] ) - FR_xy[1],
        x - (RL_xy[0] - RR_xy[0]) * (y - RR_xy[1]) / (RL_xy[1] - RR_xy[1] ) - RR_xy[0],
        y - (FL_xy[1] - RL_xy[1]) * (x - RL_xy[0]) / (FL_xy[0] - RL_xy[0] ) - RL_xy[1],
        y - (RR_xy[1] - FL_xy[1]) * (x - FL_xy[0]) / (RR_xy[0] - FL_xy[0] ) - FL_xy[1],
        y - (RL_xy[1] - FR_xy[1]) * (x - FR_xy[0]) / (RL_xy[0] - FR_xy[0] ) - FR_xy[1],
    ])

    return c

def build_zmp_bounds_from_contacts(stance_step, stability_margin=0.04):
    INF = 1000.0
    lb = -INF * np.ones(6)
    ub =  INF * np.ones(6)

    FL, FR, RL, RR = stance_step

    # CRAWL BACKDIAGONALCRAWL ONLY (exact copy of acados logic)
    if FL == 1:
        if FR == 1:
            ub[0] = -stability_margin      # FL_FR <= -margin
        else:
            lb[4] = +stability_margin      # FL_RR >= +margin

    if FR == 1:
        if RR == 1:
            lb[1] = +stability_margin      # FR_RR >= +margin
        else:
            lb[5] = +stability_margin      # FR_RL >= +margin

    if RR == 1:
        if RL == 1:
            lb[2] = +stability_margin      # RR_RL >= +margin
        else:
            ub[4] = -stability_margin      # FL_RR <= -margin

    if RL == 1:
        if FL == 1:
            ub[3] = -stability_margin      # RL_FL <= -margin
        else:
            ub[5] = -stability_margin      # FR_RL <= -margin

    return lb, ub


N = u_pred.shape[0]

zmp_constraints = np.zeros((N, 6))
zmp_lb = np.zeros((N, 6))
zmp_ub = np.zeros((N, 6))
zmp_min_margin = np.zeros(N)
zmp_worst_violation = np.zeros(N)

for j in range(N):
    stance_j = [
        contact_sequence[0][j],
        contact_sequence[1][j],
        contact_sequence[2][j],
        contact_sequence[3][j],
    ]

    c = compute_zmp_constr(
        u_pred[j],
        x_pred[j],
        stance_j,
    )

    lb, ub = build_zmp_bounds_from_contacts(
        stance_j,
        stability_margin=0.04
    )

    zmp_constraints[j] = c
    zmp_lb[j] = lb
    zmp_ub[j] = ub

    viol_upper = c - ub
    viol_lower = lb - c
    zmp_worst_violation[j] = max(viol_upper.max(), viol_lower.max())

    # signed margin (human-friendly)
    margins = np.zeros(6)
    for k in range(6):
        if ub[k] < 1000:
            margins[k] = ub[k] - c[k]
        elif lb[k] > -1000:
            margins[k] = c[k] - lb[k]
        else:
            margins[k] = np.inf

    zmp_min_margin[j] = margins.min()


plt.figure()
plt.plot(zmp_min_margin, marker="o")
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Horizon index j")
plt.ylabel("ZMP min margin [m]")
plt.title("ZMP stability margin along horizon")
plt.grid(True)
plt.show()