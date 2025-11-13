import rclpy 
from rclpy.node import Node 
from dls2_msgs.msg import BaseStateMsg, BlindStateMsg, ControlSignalMsg, TrajectoryGeneratorMsg
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading
import multiprocessing
from multiprocessing import shared_memory, Value
import queue
import copy
import os
import sys
import gc

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from gym_quadruped.utils.mujoco.visual import render_vector
from gym_quadruped.utils.mujoco.visual import render_sphere

# Config imports
from quadruped_pympc import config as cfg

dir_path = os.path.dirname(os.path.realpath(__file__))

# -------------------- Optional: process priority and affinity (no sudo assumed) ----------------
pid = os.getpid()
print("PID:", pid)
# Use os.system to elevate priority (will ask for password)
try:
    print(f"Setting high priority for main process (PID: {pid})...")
    os.system("sudo renice -n -21 -p " + str(pid))
    os.system("sudo echo -20 > /proc/" + str(pid) + "/autogroup")
except Exception as e:
    print(f"Warning: Could not set main process priority: {e}")

# Note: You are already setting affinity with 'taskset -c 4-5'
# but we can leave this as a fallback.
try:
    affinity_mask = {4, 5}
    os.sched_setaffinity(pid, affinity_mask)
except Exception:
    pass
# ----------------------------------------------------------------------------------------------

USE_DLS_CONVENTION = True

USE_THREADED_MPC = False
USE_PROCESS_MPC  = True
MPC_FREQ = 100 

USE_SCHEDULER = False # This enable a call to the run function every tot seconds, instead of as fast as possible
SCHEDULER_FREQ = 250 # this is only valid if USE_SCHEDULER is True

USE_FIXED_LOOP_TIME = False # This is used to fix the clock time of periodic gait gen to 1/SCHEDULER_FREQ
USE_SATURATED_LOOP_TIME = True # This is used to cap the clock time of periodic gait gen to max 250Hz

USE_SMOOTH_VELOCITY = False
USE_SMOOTH_HEIGHT = True

# Latest-only data policy for MPC output
MAX_MPC_DATA_AGE_S = 0.05  # 20 ms

# -------------------- Shared-memory layout for MPC → WBC --------------------------------------
# Payload layout (float64):
# 0..11   : GRF   (4 legs × 3)
# 12..23  : Footholds (4×3)
# 24..35  : Joints pos (4×3)
# 36..47  : Joints vel (4×3)
# 48..59  : Joints acc (4×3)
# 60..71  : Predicted state (12)
# 72      : best_sample_freq (1)
# 73      : last_mpc_loop_time (1)
# 74      : stamp_mono (1)
N_DBL = 75
IDX_GRF   = slice(0, 12)
IDX_FH    = slice(12, 24)
IDX_JP    = slice(24, 36)
IDX_JV    = slice(36, 48)
IDX_JA    = slice(48, 60)
IDX_PRED  = slice(60, 72)
IDX_BSF   = 72
IDX_LAST  = 73
IDX_STAMP = 74


def legsattr_to12(legs: LegsAttr) -> np.ndarray:
    return np.concatenate([np.asarray(legs.FL).reshape(-1),
                           np.asarray(legs.FR).reshape(-1),
                           np.asarray(legs.RL).reshape(-1),
                           np.asarray(legs.RR).reshape(-1)], axis=0)


def vec12_to_legsattr(vec12: np.ndarray) -> LegsAttr:
    v = np.asarray(vec12).reshape(4, 3)
    return LegsAttr(FL=v[0].copy(), FR=v[1].copy(), RL=v[2].copy(), RR=v[3].copy())


class Quadruped_PyMPC_Node(Node):
    def __init__(self):
        super().__init__('Quadruped_PyMPC_Node')

        # Subscribers and Publishers
        self.subscription_base_state = self.create_subscription(BaseStateMsg, "/dls2/base_state", self.get_base_state_callback, 1)
        self.subscription_blind_state = self.create_subscription(BlindStateMsg, "/dls2/blind_state", self.get_blind_state_callback, 1)
        self.subscription_joy = self.create_subscription(Joy, "joy", self.get_joy_callback, 1)

        self.publisher_control_signal = self.create_publisher(ControlSignalMsg, "dls2/quadruped_pympc_torques", 1)
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGeneratorMsg, "dls2/trajectory_generator", 1)

        if USE_SCHEDULER:
            self.timer = self.create_timer(1.0 / SCHEDULER_FREQ, self.compute_control_callback)

        # Flags
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False

        # Timing
        self.loop_time = 0.002
        self.last_start_time = None
        self.last_mpc_loop_time = 0.0
        self.last_mpc_update_mono = 0.0

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.stance_status = np.zeros(4)
        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.feet_contact = np.zeros(4)
        # Desired PD gain
        self.impedence_joint_position_gain = np.ones(12) * cfg.simulation_params['impedence_joint_position_gain']
        self.impedence_joint_velocity_gain = np.ones(12) * cfg.simulation_params['impedence_joint_velocity_gain']

        # Mujoco env
        self.env = QuadrupedEnv(
            robot=cfg.robot,
            scene=cfg.simulation_params['scene'],
            sim_dt=cfg.simulation_params['dt'],
            base_vel_command_type="human"
        )
        self.env.mjModel.opt.gravity[2] = -cfg.gravity_constant

        self.feet_traj_geom_ids, self.feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
        self.legs_order = ["FL", "FR", "RL", "RR"]
        self.env.reset(random=False)
        self.last_mpc_time = time.time()

        # Controllers
        from quadruped_pympc.interfaces.srbd_controller_interface import SRBDControllerInterface
        from quadruped_pympc.interfaces.srbd_batched_controller_interface import SRBDBatchedControllerInterface
        from quadruped_pympc.interfaces.wb_interface import WBInterface

        self.wb_interface = WBInterface(initial_feet_pos=self.env.feet_pos(frame='world'), legs_order=self.legs_order)
        self.srbd_controller_interface = SRBDControllerInterface()

        # Shared interface variables (threaded fallback)
        self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_footholds = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_pos = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_vel = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_acc = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_predicted_state = np.zeros(12)

        self.best_sample_freq = self.wb_interface.pgg.step_freq
        self.state_current = None
        self.ref_state = None
        self.contact_sequence = None
        self.inertia = None
        self.optimize_swing = None

        # Torques and limits
        self.tau = LegsAttr(*[np.zeros((self.env.mjModel.nv, 1)) for _ in range(4)])
        tau_soft_limits_scalar = 0.9
        self.tau_limits = LegsAttr(
            FL=self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.FL] * tau_soft_limits_scalar,
            FR=self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.FR] * tau_soft_limits_scalar,
            RL=self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.RL] * tau_soft_limits_scalar,
            RR=self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.RR] * tau_soft_limits_scalar,
        )

        # Start in FULL STANCE
        self.wb_interface.pgg.gait_type = 7
        self.tau_filtered = LegsAttr(FL=np.zeros((3,1)), FR=np.zeros((3,1)), RL=np.zeros((3,1)), RR=np.zeros((3,1)))


        # Threaded MPC
        if USE_THREADED_MPC:
            thread_mpc = threading.Thread(target=self.compute_mpc_thread_callback)
            thread_mpc.daemon = True
            thread_mpc.start()

        # Processes and IPC
        self._init_ipc()

        # Console thread
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

    # -------------------- IPC setup -------------------------------------------------------------
    def _init_ipc(self):
        self.input_data_process = None
        self.shm_out = None
        self.shm_out_name = None
        self.seq_out = None

        if USE_PROCESS_MPC:
            # WBC → MPC: keep a latest-only queue for complex inputs
            self.input_data_process = multiprocessing.Queue(maxsize=1)

            # MPC → WBC: shared-memory SPSC with seqlock
            self.shm_out = shared_memory.SharedMemory(create=True, size=N_DBL * 8)
            self.shm_out_name = self.shm_out.name
            np.ndarray((N_DBL,), dtype=np.float64, buffer=self.shm_out.buf)[:] = 0.0
            self.seq_out = Value('Q', 0, lock=False)  # 64-bit sequence, even=stable

            process_mpc = multiprocessing.Process(
                target=self.compute_mpc_process_callback,
                args=(self.input_data_process, self.shm_out_name, self.seq_out),
            )
            process_mpc.daemon = True
            process_mpc.start()

    # -------------------- Threaded MPC (unchanged) ---------------------------------------------
    def compute_mpc_thread_callback(self):
        last_mpc_thread_time = time.perf_counter()
        while True:
            if time.perf_counter() - last_mpc_thread_time > 1.0 / MPC_FREQ:
                if self.state_current is not None:
                    (
                        self.nmpc_GRFs,
                        self.nmpc_footholds,
                        self.nmpc_joints_pos,
                        self.nmpc_joints_vel,
                        self.nmpc_joints_acc,
                        self.best_sample_freq,
                        self.nmpc_predicted_state,
                    ) = self.srbd_controller_interface.compute_control(
                        self.state_current,
                        self.ref_state,
                        self.contact_sequence,
                        self.inertia,
                        self.wb_interface.pgg.phase_signal,
                        self.wb_interface.pgg.step_freq,
                        self.optimize_swing,
                    )
                    if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                        self.srbd_controller_interface.compute_RTI()
                    last_mpc_thread_time = time.perf_counter()

    # -------------------- MPC child process with SHM output ------------------------------------
    def compute_mpc_process_callback(self, input_data_process, shm_out_name: str, seq_out: Value):
        gc.disable()  # <-- Good!
        pid = os.getpid()

        # Set affinity for MPC process
        try:
            os.sched_setaffinity(pid, {6, 7})
        except Exception as e:
            print(f"MPC Process: Warning: Could not set CPU affinity. {e}")

        # --- This is your working method ---
        # Set high priority for MPC process
        try:
            print(f"Setting high priority for MPC process (PID: {pid})...")
            os.system("sudo renice -n -21 -p " + str(pid))
            os.system("sudo echo -20 > /proc/" + str(pid) + "/autogroup")
        except Exception as e:
            print(f"Warning: Could not set MPC process priority: {e}")
        # --- End of priority code ---

        shm = shared_memory.SharedMemory(name=shm_out_name)
        arr = np.ndarray((N_DBL,), dtype=np.float64, buffer=shm.buf)

        period = 1.0 / MPC_FREQ
        next_t = time.perf_counter()

        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            next_t += period

            # Non-blocking latest-only input
            try:
                data = input_data_process.get_nowait()
            except queue.Empty:
                continue

            (
                state_current,
                ref_state,
                contact_sequence,
                inertia,
                optimize_swing,
                phase_signal,
                step_freq,
            ) = data

            t0 = time.perf_counter()
            (
                nmpc_GRFs,
                nmpc_footholds,
                nmpc_joints_pos,
                nmpc_joints_vel,
                nmpc_joints_acc,
                best_sample_freq,
                nmpc_predicted_state,
            ) = self.srbd_controller_interface.compute_control(
                state_current,
                ref_state,
                contact_sequence,
                inertia,
                phase_signal,
                step_freq,
                optimize_swing,
            )
            comp_time = time.perf_counter() - t0

            # Publish to SHM with seqlock: odd=writing, even=stable
            s = seq_out.value
            if s % 2 == 0:
                seq_out.value = s + 1
            # pack payload
            arr[IDX_GRF]  = legsattr_to12(nmpc_GRFs)
            arr[IDX_FH]   = legsattr_to12(nmpc_footholds)
            arr[IDX_JP]   = (nmpc_joints_pos if nmpc_predicted_state is not None else np.zeros(12).reshape(-1)[:12])
            arr[IDX_JV]   = (nmpc_joints_pos if nmpc_predicted_state is not None else np.zeros(12).reshape(-1)[:12])
            arr[IDX_JA]   = (nmpc_joints_pos if nmpc_predicted_state is not None else np.zeros(12).reshape(-1)[:12])
            arr[IDX_PRED] = np.asarray(nmpc_predicted_state).reshape(-1)[:12]
            arr[IDX_BSF]  = float(best_sample_freq)
            arr[IDX_LAST] = float(comp_time)
            arr[IDX_STAMP]= float(time.monotonic())
            # mark stable
            seq_out.value = (s | 1) + 1

            if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                self.srbd_controller_interface.compute_RTI()

    # -------------------- ROS callbacks ---------------------------------------------------------
    def get_base_state_callback(self, msg):
        if USE_SMOOTH_HEIGHT:
            self.position[2] = 0.5 * self.position[2] + 0.5 * np.array(msg.position)[2]
        else:
            self.position[2] = np.array(msg.position)[2]
        self.position[0:2] = np.array(msg.position)[0:2]

        if USE_SMOOTH_VELOCITY:
            self.linear_velocity = 0.5 * self.linear_velocity + 0.5 * np.array(msg.linear_velocity)
        else:
            self.linear_velocity = np.array(msg.linear_velocity)

        self.orientation = np.roll(np.array(msg.orientation), 1)
        self.angular_velocity = np.array(msg.angular_velocity)
        self.stance_status = np.array(msg.stance_status)

        self.first_message_base_arrived = True

    def get_blind_state_callback(self, msg):
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)
        self.feet_contact = np.array(msg.feet_contact)

        if USE_DLS_CONVENTION:
            self.joint_positions[0] = -self.joint_positions[0]
            self.joint_positions[6] = -self.joint_positions[6]
            self.joint_velocities[0] = -self.joint_velocities[0]
            self.joint_velocities[6] = -self.joint_velocities[6]

        self.first_message_joints_arrived = True

        if not USE_SCHEDULER:
            self.compute_control_callback()

    def get_joy_callback(self, msg):
        self.env._ref_base_lin_vel_H[0] = msg.axes[1] / 3.5
        self.env._ref_base_lin_vel_H[1] = msg.axes[0] / 3.5
        self.env._ref_base_ang_yaw_dot = msg.axes[3] / 2.0

        if msg.buttons[8] == 1:
            self.get_logger().info("Joystick button pressed, shutting down the node.")
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            os.system("pkill -f play_ros2.py")
            exit(0)

    # -------------------- Main control loop -----------------------------------------------------
    def compute_control_callback(self):
        # Loop timing
        if USE_FIXED_LOOP_TIME:
            simulation_dt = 1.0 / SCHEDULER_FREQ
        else:
            start_time = time.perf_counter()
            if self.last_start_time is not None:
                self.loop_time = start_time - self.last_start_time
            self.last_start_time = start_time
            simulation_dt = self.loop_time
            if USE_SATURATED_LOOP_TIME and simulation_dt > 0.005:
                simulation_dt = 0.005

        # Safety
        if not self.first_message_base_arrived and not self.first_message_joints_arrived:
            return

        # Update Mujoco model
        self.env.mjData.qpos[0:3] = copy.deepcopy(self.position)
        self.env.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
        self.env.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)
        self.env.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
        self.env.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
        self.env.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
        self.env.mjModel.opt.timestep = simulation_dt
        self.env.mjModel.opt.disableflags = 16
        mujoco.mj_forward(self.env.mjModel, self.env.mjData)

        # Collect state
        legs_order = ["FL", "FR", "RL", "RR"]
        feet_pos = self.env.feet_pos(frame='world')
        feet_vel = self.env.feet_vel(frame='world')
        hip_pos = self.env.hip_positions(frame='world')
        base_lin_vel = self.env.base_lin_vel(frame='world')
        base_ang_vel = self.env.base_ang_vel(frame='base')
        base_ori_euler_xyz = self.env.base_ori_euler_xyz
        base_pos = self.env.base_pos
        com_pos = self.env.com

        # References
        ref_base_lin_vel, ref_base_ang_vel = self.env.target_base_vel()

        # Inertia
        if cfg.simulation_params['use_inertia_recomputation']:
            inertia = self.env.get_base_inertia().flatten()
        else:
            inertia = cfg.inertia.flatten()

        # qpos, qvel
        qpos, qvel = self.env.mjData.qpos, self.env.mjData.qvel
        joints_pos = LegsAttr(FL=qpos[7:10], FR=qpos[10:13], RL=qpos[13:16], RR=qpos[16:19])

        # Dynamics terms
        legs_mass_matrix = self.env.legs_mass_matrix
        legs_qfrc_bias = self.env.legs_qfrc_bias
        legs_qfrc_passive = self.env.legs_qfrc_passive

        # Jacobians
        feet_jac = self.env.feet_jacobians(frame='world', return_rot_jac=False)
        feet_jac_dot = self.env.feet_jacobians_dot(frame='world', return_rot_jac=False)

        # Indices
        legs_qvel_idx = self.env.legs_qvel_idx
        legs_qpos_idx = self.env.legs_qpos_idx

        # Heightmaps
        heightmaps = None

        # Update WBC state and reference
        (
            state_current,
            ref_state,
            contact_sequence,
            step_height,
            optimize_swing,
        ) = self.wb_interface.update_state_and_reference(
            com_pos,
            base_pos,
            base_lin_vel,
            base_ori_euler_xyz,
            base_ang_vel,
            feet_pos,
            hip_pos,
            joints_pos,
            heightmaps,
            legs_order,
            simulation_dt,
            ref_base_lin_vel,
            ref_base_ang_vel,
        )

        # Console tweaks
        ref_state["ref_position"][2] += self.console.height_delta
        ref_state["ref_orientation"][1] += self.console.pitch_delta

        # ---- MPC I/O ----
        if USE_THREADED_MPC:
            self.state_current = state_current
            self.ref_state = ref_state
            self.contact_sequence = contact_sequence
            self.inertia = inertia
            self.optimize_swing = optimize_swing

        elif USE_PROCESS_MPC:
            # Latest-only put into MPC input queue
            mpc_in = [state_current, ref_state, contact_sequence, inertia, optimize_swing, self.wb_interface.pgg.phase_signal, self.wb_interface.pgg.step_freq]
            # try:
            #     self.input_data_process.put_nowait(mpc_in)
            # except queue.Full:
            #     try:
            #         _ = self.input_data_process.get_nowait()
            #     except queue.Empty:
            #         pass
            #     self.input_data_process.put_nowait(mpc_in)

            q = self.input_data_process
            while True:
                try:
                    q.put_nowait(mpc_in)      # latest-only
                    break
                except queue.Full:
                    try:
                        _ = q.get_nowait()    # drop stale
                    except queue.Empty:
                        pass   
            # Read MPC output from shared memory with seqlock and stale-data guard
            if self.shm_out is not None and self.seq_out is not None:
                s1 = self.seq_out.value
                if s1 % 2 == 0:  # writer not in progress
                    buf = np.ndarray((N_DBL,), dtype=np.float64, buffer=self.shm_out.buf)
                    tmp = buf.copy()  # local copy
                    s2 = self.seq_out.value
                    if s1 == s2 and (s2 % 2 == 0):
                        # Validate age
                        if time.monotonic() - float(tmp[IDX_STAMP]) <= MAX_MPC_DATA_AGE_S:
                            self.nmpc_GRFs        = vec12_to_legsattr(tmp[IDX_GRF])
                            self.nmpc_footholds   = vec12_to_legsattr(tmp[IDX_FH])
                            self.nmpc_joints_pos  = vec12_to_legsattr(tmp[IDX_JP])
                            self.nmpc_joints_vel  = vec12_to_legsattr(tmp[IDX_JV])
                            self.nmpc_joints_acc  = vec12_to_legsattr(tmp[IDX_JA])
                            self.nmpc_predicted_state = tmp[IDX_PRED].copy()
                            self.best_sample_freq  = float(tmp[IDX_BSF])
                            self.last_mpc_loop_time = float(tmp[IDX_LAST])
                            self.last_mpc_update_mono = float(tmp[IDX_STAMP])
                        else:
                        # # --- THIS IS THE NEW, CRITICAL PART ---
                        # DATA IS STALE: Command a safe, zero-GRF fallback
                            self.get_logger().warn("MPC data STALE. Commanding zero GRFs.")
                            self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
                        # # You can also zero out other commands if needed
                        # # self.nmpc_footholds = ... (e.g., current footholds)

        else:
            if time.time() - self.last_mpc_time > 1.0 / MPC_FREQ:
                self.last_mpc_loop_time = time.time() - self.last_mpc_time
                (
                    self.nmpc_GRFs,
                    self.nmpc_footholds,
                    self.nmpc_joints_pos,
                    self.nmpc_joints_vel,
                    self.nmpc_joints_acc,
                    self.best_sample_freq,
                    self.nmpc_predicted_state,
                ) = self.srbd_controller_interface.compute_control(
                    state_current,
                    ref_state,
                    contact_sequence,
                    inertia,
                    self.wb_interface.pgg.phase_signal,
                    self.wb_interface.pgg.step_freq,
                    optimize_swing,
                )
                if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                    self.srbd_controller_interface.compute_RTI()
                self.last_mpc_time = time.time()

        # ---- WBC torque computation ----
        (
            self.tau,
            pd_target_joints_pos,
            pd_target_joints_vel,
        ) = self.wb_interface.compute_stance_and_swing_torque(
            simulation_dt,
            qpos,
            qvel,
            feet_jac,
            feet_jac_dot,
            feet_pos,
            feet_vel,
            legs_qfrc_passive,
            legs_qfrc_bias,
            legs_mass_matrix,
            self.nmpc_GRFs,
            self.nmpc_footholds,
            legs_qpos_idx,
            legs_qvel_idx,
            self.tau_filtered,
            optimize_swing,
            self.best_sample_freq,
            self.nmpc_joints_pos,
            self.nmpc_joints_vel,
            self.nmpc_joints_acc,
            self.nmpc_predicted_state,
        )

        # Torque limits
        # for leg in ["FL", "FR", "RL", "RR"]:
        #     tau_min, tau_max = self.tau_limits[leg][:, 0], self.tau_limits[leg][:, 1]
        #     self.tau[leg] = np.clip(self.tau[leg], tau_min, tau_max)

        # if USE_DLS_CONVENTION:
        #     self.tau.FL[0] = -self.tau.FL[0]
        #     self.tau.RL[0] = -self.tau.RL[0]

        # control_signal_msg = ControlSignalMsg()
        # control_signal_msg.torques = np.concatenate([self.tau.FL, self.tau.FR, self.tau.RL, self.tau.RR], axis=0).flatten()
        # control_signal_msg.timestamp = self.last_mpc_loop_time
        # self.publisher_control_signal.publish(control_signal_msg)

        # Torque limits
        for leg in ["FL", "FR", "RL", "RR"]:
            tau_min, tau_max = self.tau_limits[leg][:, 0], self.tau_limits[leg][:, 1]
            self.tau[leg] = np.clip(self.tau[leg], tau_min, tau_max)

        # --- THIS IS THE NEW FILTERING SECTION ---
        
        # Low-pass filter the output torques to smooth the "shakes"
        # alpha = 0.0: No filter (very jerky)
        # alpha = 0.9: Very smooth (but adds delay)
        # Start with a value like 0.7 or 0.8
        alpha = 0.8 
        self.tau_filtered.FL = alpha * self.tau_filtered.FL + (1.0 - alpha) * self.tau.FL
        self.tau_filtered.FR = alpha * self.tau_filtered.FR + (1.0 - alpha) * self.tau.FR
        self.tau_filtered.RL = alpha * self.tau_filtered.RL + (1.0 - alpha) * self.tau.RL
        self.tau_filtered.RR = alpha * self.tau_filtered.RR + (1.0 - alpha) * self.tau.RR
        
        # Apply convention AFTER filtering
        if USE_DLS_CONVENTION:
            self.tau_filtered.FL[0] = -self.tau_filtered.FL[0]
            self.tau_filtered.RL[0] = -self.tau_filtered.RL[0]
        # --- END NEW FILTERING SECTION ---

        control_signal_msg = ControlSignalMsg()
        
        # Publish the FILTERED torques
        control_signal_msg.torques = np.concatenate([self.tau_filtered.FL, self.tau_filtered.FR, self.tau_filtered.RL, self.tau_filtered.RR], axis=0).flatten()
        control_signal_msg.timestamp = self.last_mpc_loop_time
        self.publisher_control_signal.publish(control_signal_msg)


        if USE_DLS_CONVENTION:
            pd_target_joints_pos.FL[0] = -pd_target_joints_pos.FL[0]
            pd_target_joints_pos.RL[0] = -pd_target_joints_pos.RL[0]
            pd_target_joints_vel.FL[0] = -pd_target_joints_vel.FL[0]
            pd_target_joints_vel.RL[0] = -pd_target_joints_vel.RL[0]

        trajectory_generator_msg = TrajectoryGeneratorMsg()
        trajectory_generator_msg.joints_position = np.concatenate(
            [pd_target_joints_pos.FL, pd_target_joints_pos.FR, pd_target_joints_pos.RL, pd_target_joints_pos.RR], axis=0
        ).flatten()
        trajectory_generator_msg.joints_velocity = np.concatenate(
            [pd_target_joints_vel.FL, pd_target_joints_vel.FR, pd_target_joints_vel.RL, pd_target_joints_vel.RR], axis=0
        ).flatten()
        trajectory_generator_msg.stance_legs[0] = bool(contact_sequence[0, 0])
        trajectory_generator_msg.stance_legs[1] = bool(contact_sequence[0, 1])
        trajectory_generator_msg.stance_legs[2] = bool(contact_sequence[0, 2])
        trajectory_generator_msg.stance_legs[3] = bool(contact_sequence[0, 3])

        trajectory_generator_msg.timestamp = self.loop_time
        self.publisher_trajectory_generator.publish(trajectory_generator_msg)


def main():
    print('Hello from Quadruped-PyMPC ros interface.')
    gc.disable()
    rclpy.init()

    controller_node = Quadruped_PyMPC_Node()

    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
