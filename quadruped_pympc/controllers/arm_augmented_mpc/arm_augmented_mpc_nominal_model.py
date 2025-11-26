# Description: This file contains the class Centroidal_Model that defines the
# prediction model used by the MPC

# Authors: Giulio Turrisi -

import time
import unittest
import casadi as cs

import numpy as np
from acados_template import AcadosModel

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(dir_path)
sys.path.append(dir_path + '/../../')

from liecasadi import SO3

from quadruped_pympc import config

use_adam = False
use_fixed_inertia = True
use_centroidal_model = True
import adam

from adam.casadi import KinDynComputations
## Import of Pinocchio 
import pinocchio as pin
import pinocchio.casadi as cpin

 

# Class that defines the prediction model of the NMPC
class Arm_Augmented_Centroidal_Model:
    def __init__(self,) -> None: 




        self.joints_name_list_ = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
               'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
               'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
               'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
               'passive_arm_joint_0','passive_arm_joint_1','passive_arm_joint_2'
               ]

        # # Load the model with free-flyer base
        # if config.robot== "aliengo_follower":
        #     model_path_ = dir_path + '/../../gym-quadruped-coll/gym_quadruped/robot_model/aliengo_follower/new_urdf_export.xml'
        # elif config.robot== "aliengo_leader":
        #     model_path_ = dir_path + '/../../gym-quadruped-coll/gym_quadruped/robot_model/aliengo_leader/new_urdf_export.xml'

        model_path_ = dir_path + '/new_urdf_export.xml'
        
        model = pin.buildModelFromMJCF(model_path_, pin.JointModelFreeFlyer())
        data  = model.createData()
        cmodel = cpin.Model(model)
        cdata  = cmodel.createData()
        q = cs.SX.sym("q", model.nq, 1)   # config (7 + n_joints)
        v = cs.SX.sym("v", model.nv, 1)   # velocity (6 + n_joints)
        M = cpin.crba(cmodel, cdata, q)
        h = cpin.rnea(cmodel, cdata, q, v, cs.SX.zeros(model.nv, 1))
        # Wrap as CasADi functions
        self.M_fun = cs.Function("M_fun", [q], [M])
        self.h_fun = cs.Function("h_fun", [q, v], [h])

        frame_id = model.getFrameId("eef")   # link/frame name from URDF
        J = cpin.computeFrameJacobian(cmodel, cdata, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.jac_arm_fun = cs.Function("jac_arm_fun", [q], [J])

        # Compute end-effector position
        # end-effector position in BASE frame
        frame_id_eef = model.getFrameId("eef")
        self.fwd_kin=cpin.forwardKinematics(cmodel, cdata, q)
        self.frame_update=cpin.updateFramePlacements(cmodel, cdata)

        p_ee_sym = cdata.oMf[frame_id_eef].translation  # 3×1 SX
        # ## create a function that computes the forward kinematics of the arm end effector
        self.fk_arm_fun = cs.Function("fk_arm_fun", [q], [p_ee_sym])

        # frame_update=cpin.updateFramePlacements(cmodel, cdata)
        # p_ee_W = cdata.oMf[frame_id_eef].translation

        # Define state and its casadi variables
        com_position_x = cs.SX.sym("com_position_x")
        com_position_y = cs.SX.sym("com_position_y")
        com_position_z = cs.SX.sym("com_position_z")

        com_velocity_x = cs.SX.sym("com_velocity_x")
        com_velocity_y = cs.SX.sym("com_velocity_y")
        com_velocity_z = cs.SX.sym("com_velocity_z")
        
        roll = cs.SX.sym("roll", 1, 1)
        pitch = cs.SX.sym("pitch", 1, 1)
        yaw = cs.SX.sym("yaw", 1, 1)
        omega_x = cs.SX.sym("omega_x", 1, 1)
        omega_y = cs.SX.sym("omega_y", 1, 1)
        omega_z = cs.SX.sym("omega_z", 1, 1)
        
        foot_position_fl = cs.SX.sym("foot_position_fl", 3, 1)
        foot_position_fr = cs.SX.sym("foot_position_fr", 3, 1)
        foot_position_rl = cs.SX.sym("foot_position_rl", 3, 1)
        foot_position_rr = cs.SX.sym("foot_position_rr", 3, 1)


        com_position_z_integral = cs.SX.sym("com_position_z_integral")
        com_velocity_x_integral = cs.SX.sym("com_velocity_x_integral")
        com_velocity_y_integral = cs.SX.sym("com_velocity_y_integral")
        com_velocity_z_integral = cs.SX.sym("com_velocity_z_integral")
        roll_integral = cs.SX.sym("roll_integral")
        pitch_integral = cs.SX.sym("pitch_integral")

        ### ARM AUGMENTATION
        q_arm_1 = cs.SX.sym("q_arm_1")
        q_arm_2 = cs.SX.sym("q_arm_2")
        q_arm_3 = cs.SX.sym("q_arm_3")



        q_dot_arm_1 = cs.SX.sym("q_dot_arm_1")
        q_dot_arm_2 = cs.SX.sym("q_dot_arm_2")
        q_dot_arm_3 = cs.SX.sym("q_dot_arm_3")


        self.states = cs.vertcat(com_position_x,
                            com_position_y,
                            com_position_z,
                            com_velocity_x,
                            com_velocity_y,
                            com_velocity_z,
                            roll,
                            pitch,
                            yaw,
                            omega_x,
                            omega_y,
                            omega_z,
                            foot_position_fl,
                            foot_position_fr,
                            foot_position_rl,
                            foot_position_rr,
                            com_position_z_integral,
                            com_velocity_x_integral,
                            com_velocity_y_integral,
                            com_velocity_z_integral,
                            roll_integral,
                            pitch_integral,
                            q_arm_1,
                            q_arm_2,
                            q_arm_3,
                            q_dot_arm_1,
                            q_dot_arm_2,
                            q_dot_arm_3
                            )
        


        # Define state dot 
        ### ARM AUGMENTATION
        self.states_dot = cs.vertcat(cs.SX.sym("linear_com_vel", 3, 1), 
                                     cs.SX.sym("linear_com_acc", 3, 1), 
                                     cs.SX.sym("euler_rates_base", 3, 1), 
                                     cs.SX.sym("angular_acc_base", 3, 1),
                                     cs.SX.sym("linear_vel_foot_FL", 3, 1),
                                     cs.SX.sym("linear_vel_foot_FR", 3, 1),
                                     cs.SX.sym("linear_vel_foot_RL", 3, 1),
                                     cs.SX.sym("linear_vel_foot_RR", 3, 1),
                                     cs.SX.sym("linear_com_vel_z_integral", 1, 1),
                                     cs.SX.sym("linear_com_acc_integral", 3, 1),
                                     cs.SX.sym("euler_rates_roll_integral", 1, 1),
                                     cs.SX.sym("euler_rates_pitch_integral", 1, 1),
                                     cs.SX.sym("q_dot_arm", 3, 1),
                                     cs.SX.sym("q_ddot_arm", 3, 1)
                                     )
        


        # Define input and its casadi variables
        foot_velocity_fl = cs.SX.sym("foot_velocity_fl", 3, 1)
        foot_velocity_fr = cs.SX.sym("foot_velocity_fr", 3, 1)
        foot_velocity_rl = cs.SX.sym("foot_velocity_rl", 3, 1)
        foot_velocity_rr = cs.SX.sym("foot_velocity_rr", 3, 1)

        foot_force_fl = cs.SX.sym("foot_force_fl", 3, 1)
        foot_force_fr = cs.SX.sym("foot_force_fr", 3, 1)
        foot_force_rl = cs.SX.sym("foot_force_rl", 3, 1)
        foot_force_rr = cs.SX.sym("foot_force_rr", 3, 1)

        self.inputs = cs.vertcat(foot_velocity_fl, 
                            foot_velocity_fr, 
                            foot_velocity_rl, 
                            foot_velocity_rr, 
                            foot_force_fl, 
                            foot_force_fr, 
                            foot_force_rl, 
                            foot_force_rr)
        

        # Usefull for debug what things goes where in y_ref in the compute_control function,
        # because there are a lot of variables
        self.y_ref = cs.vertcat(self.states, self.inputs)
        

        # Define acados parameters that can be changed at runtine
        self.stanceFL = cs.SX.sym("stanceFL", 1, 1)
        self.stanceFR = cs.SX.sym("stanceFR", 1, 1)
        self.stanceRL = cs.SX.sym("stanceRL", 1, 1)
        self.stanceRR = cs.SX.sym("stanceRR", 1, 1)
        self.stance_param = cs.vertcat(self.stanceFL , self.stanceFR , self.stanceRL , self.stanceRR)


        self.mu_friction = cs.SX.sym("mu_friction", 1, 1)
        self.stance_proximity = cs.SX.sym("stanceProximity", 4, 1)
        self.base_position = cs.SX.sym("base_position", 3, 1)
        self.base_yaw = cs.SX.sym("base_yaw", 1, 1)

        self.external_wrench = cs.SX.sym("external_wrench", 6, 1)

        self.mass = cs.SX.sym("mass", 1, 1)

        ### ARM AUGMENTATION
        self.k = cs.SX.sym("k",3,1)
        self.d = cs.SX.sym("D",3,1)

        self.q_arm_rest_1 = cs.SX.sym("q_arm_rest_1")
        self.q_arm_rest_2 = cs.SX.sym("q_arm_rest_2")
        self.q_arm_rest_3 = cs.SX.sym("q_arm_rest_3")

        self.joint_position = np.array([
            0.0, 0.9, -1.8, 
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8, 
            0.0, 0.9, -1.8,
            0.0, 0.0, 0.0 #for now Ill fix somw random pos 
        ])

        self.joint_vel= np.zeros(len(self.joints_name_list_ ))




    def compute_b_R_w(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        #Z Y X rotations!
        #world to base
        Rx = cs.SX.eye(3)
        Rx[0,0] = 1   
        Rx[0,1] = 0
        Rx[0,2] = 0
        Rx[1,0] = 0
        Rx[1,1] = cs.cos(roll)
        Rx[1,2] = cs.sin(roll)
        Rx[2,0] = 0
        Rx[2,1] = -cs.sin(roll)
        Rx[2,2] = cs.cos(roll)
                
        Ry = cs.SX.eye(3)
        Ry[0,0] = cs.cos(pitch)
        Ry[0,1] = 0
        Ry[0,2] = -cs.sin(pitch)
        Ry[1,0] = 0
        Ry[1,1] = 1
        Ry[1,2] = 0
        Ry[2,0] = cs.sin(pitch)
        Ry[2,1] = 0
        Ry[2,2] = cs.cos(pitch)

        Rz = cs.SX.eye(3)
        Rz[0,0] = cs.cos(yaw)
        Rz[0,1] = cs.sin(yaw)
        Rz[0,2] = 0
        Rz[1,0] = -cs.sin(yaw)
        Rz[1,1] = cs.cos(yaw)
        Rz[1,2] = 0
        Rz[2,0] = 0
        Rz[2,1] = 0
        Rz[2,2] = 1
        
        b_R_w = Rx@Ry@Rz
        return b_R_w


    def forward_dynamics(self, states: np.ndarray, inputs: np.ndarray, param: np.ndarray) -> cs.SX:
        """
        This method computes the symbolic forward dynamics of the robot. It is used inside
        Acados to compute the prediction model. It fill the same variables as the one in
        self.states_dot.

        Args:
            states: A numpy array of shape (29,) representing the current state of the robot.
            inputs: A numpy array of shape (29,) representing the inputs to the robot.
            param: A numpy array of shape (4,) representing the parameters (contact status) of the robot.

        Returns:
            A CasADi SX object of shape (29,) representing the predicted state of the robot.
        """
        
        # Saving for clarity a bunch of variables
        foot_velocity_fl = inputs[0:3]
        foot_velocity_fr = inputs[3:6]
        foot_velocity_rl = inputs[6:9]
        foot_velocity_rr = inputs[9:12]
        foot_force_fl = inputs[12:15]
        foot_force_fr = inputs[15:18]
        foot_force_rl = inputs[18:21]
        foot_force_rr = inputs[21:24]

        com_position = states[0:3]
        com_velocity = states[3:6]
        foot_position_fl = states[12:15]
        foot_position_fr = states[15:18]
        foot_position_rl = states[18:21]
        foot_position_rr = states[21:24]

        q_arm= states[30:33] #arm joint position
        q_dot_arm = states[33:36] #arm joint velocity
        
        stanceFL = param[0]
        stanceFR = param[1]
        stanceRL = param[2]
        stanceRR = param[3]
        stance_proximity_FL = param[5]
        stance_proximity_FR = param[6]
        stance_proximity_RL = param[7]
        stance_proximity_RR = param[8]

        external_wrench_linear  = param[13:16]
        external_wrench_angular = param[16:19]

        k = param[19:22] #spring constant
        d = param[22:25] #damping constant


        roll = states[6]
        pitch = states[7]
        yaw = states[8]
        
        b_R_w = self.compute_b_R_w(roll, pitch, yaw)

        omega = states[9:12] #angular velocity in base frame
        r_dot = com_velocity #linear velocity in base frame

        # roll = states[6]
        # pitch = states[7]
        # yaw = states[8]
        
        # b_R_w = self.compute_b_R_w(roll, pitch, yaw)
        # omega_rotated = b_R_w @ omega



        ##=============================== JOINTS POSITION AND VELOCITY ===========================
        #update the joint position before passing the value to adam
        joint_position_update = cs.vertcat(
            self.joint_position[0:12], #legs joint position
            q_arm #arm joint position
        )

        self.joint_position_update = joint_position_update
        joint_vel = cs.vertcat(
            self.joint_vel[0:12], #legs joint velocity
            q_dot_arm #arm joint velocity
        )


        # pinocchio takes as input the base pos + the joint position
        # convert rpy to quaternion
        base_quat = cs.SX.zeros(4,1) # x y z w
        cy = cs.cos(yaw * 0.5)
        sy = cs.sin(yaw * 0.5)
        cp = cs.cos(pitch * 0.5)
        sp = cs.sin(pitch * 0.5)
        cr = cs.cos(roll * 0.5)
        sr = cs.sin(roll * 0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy          
        # Pinocchio wants [x,y,z,w]
        base_quat = cs.vertcat(qx, qy, qz, qw)

        self.full_joint_pos_update= cs.vertcat(com_position, base_quat, joint_position_update)
        M_arm=self.M_fun(self.full_joint_pos_update)[18:21, 18:21]
        ## velocity
        self.full_vel= cs.vertcat(r_dot, omega, joint_vel)

        M_base_arm=self.M_fun(self.full_joint_pos_update)[18:21,0:6]
        M_arm_base=self.M_fun(self.full_joint_pos_update)[0:6,18:21]
        h=self.h_fun(self.full_joint_pos_update,self.full_vel)
        B_arm = h[18:21]
        inertia = self.M_fun(self.full_joint_pos_update)[3:6,3:6] #inertia matrix of the base in base frame
        self.mass = self.M_fun(self.full_joint_pos_update)[0,0] # the mass is the first entry of the mass matrix
        J_eval = self.jac_arm_fun(self.full_joint_pos_update)   # 6 x nv
        jac_eef_arm = J_eval[:, -3:]           # 6 x 3 arm-only

        p_eef = self.fk_arm_fun(self.full_joint_pos_update)  # 3×1 SX

        
        
        # FINAL linear_com_vel STATE (1)
        linear_com_vel = states[3:6]
        

        # FINAL linear_com_acc STATE (2)
        temp =  foot_force_fl@stanceFL 
        temp += foot_force_fr@stanceFR
        temp += foot_force_rl@stanceRL
        temp += foot_force_rr@stanceRR
        # temp += external_wrench_linear
        gravity= cs.SX([0, 0, -9.81])

        # Start to write the component of euler_rates_base and angular_acc_base STATES
        w = states[9:12]
        roll = states[6]
        pitch = states[7]
        yaw = states[8]
    
        conj_euler_rates = cs.SX.eye(3)
        conj_euler_rates[1, 1] = cs.cos(roll)
        conj_euler_rates[2, 2] = cs.cos(pitch)*cs.cos(roll)
        conj_euler_rates[2, 1] = -cs.sin(roll)
        conj_euler_rates[0, 2] = -cs.sin(pitch)
        conj_euler_rates[1, 2] = cs.cos(pitch)*cs.sin(roll)


        temp2 =  cs.skew(foot_position_fl - com_position)@foot_force_fl@stanceFL 
        temp2 += cs.skew(foot_position_fr - com_position)@foot_force_fr@stanceFR
        temp2 += cs.skew(foot_position_rl - com_position)@foot_force_rl@stanceRL
        temp2 += cs.skew(foot_position_rr - com_position)@foot_force_rr@stanceRR

        

        

        # FINAL angular_acc_base STATE (4)

        

        ##################################################


        # fill the symbolic matrices
        ## Spring torques
        # tau_spring[0] = - k[0] * (q_arm[0]-self.q_arm_rest_1) 
        # tau_spring[1] = - k[1] * (q_arm[1]-self.q_arm_rest_2)
        # tau_spring[2] = - k[2] * (q_arm[2]-self.q_arm_rest_3)

        # ## Damping torques
        # tau_damping[0] = - d[0] * (q_dot_arm[0])
        # tau_damping[1] = - d[1] * (q_dot_arm[1])
        # tau_damping[2] = - d[2] * (q_dot_arm[2])

        # Cartesian-Joint space torque (J^T @ F_ext)
        # I can pass directly the torques from my momentum observer built on the arm analystical model
        

        wrench_estimate_lin_base = b_R_w@external_wrench_linear #linear part of the wrench estimated from the arm end effector in the world frame
        wrench_estimate_ang_base = b_R_w@external_wrench_angular #angular part of the wrench estimated from the arm end effector in the world fram must go in base frame 

        wrench_estimate_mixed = cs.vertcat(wrench_estimate_lin_base, wrench_estimate_ang_base) #wrench estimated from the arm end effector in the mixed representation

        tau_ext = jac_eef_arm.T @ wrench_estimate_mixed #this is the external torque from the arm end effector

        # # ## Spring-damping matrix

        D_M= cs.SX.zeros(3,1)
        D_M[0] = - k[0] * (q_arm[0]-self.q_arm_rest_1)   - d[0] * (q_dot_arm[0]) + tau_ext[0]
        D_M[1] = - k[1] * (q_arm[1]-self.q_arm_rest_2)   - d[1] * (q_dot_arm[1]) + tau_ext[1]
        D_M[2] = - k[2] * (q_arm[2]-self.q_arm_rest_3)   - d[2] * (q_dot_arm[2]) + tau_ext[2]

        q_ddot_arm = cs.inv(M_arm) @ ( -B_arm + D_M) 

        # coupling torques between the base and the arm
        tau_base_arm = M_base_arm.T @ q_ddot_arm

        # ###
        # # external wrench in base frame
        tau_arm_B = b_R_w @ wrench_estimate_ang_base + cs.skew(p_eef - com_position) @ wrench_estimate_lin_base

        temp  +=  - tau_base_arm[0:3] + wrench_estimate_lin_base
        temp2 +=  - tau_base_arm[3:6] + wrench_estimate_ang_base +tau_arm_B #this is the torque that goes in the base frame

        # # # set arm to zero
        ### setting this made the system at least stand up
        # q_ddot_arm = cs.SX.zeros(3,1) #arm acceleration is zero for now
        # q_dot_arm = cs.SX.zeros(3,1) #arm joint velocity

        linear_com_acc = (1/self.mass)@temp + gravity 
        angular_acc_base = cs.inv(inertia)@(b_R_w@temp2 - cs.skew(w)@inertia@w ) 

        # FINAL euler_rates_base STATE (3)
        euler_rates_base = cs.inv(conj_euler_rates)@w

        # FINAL linear_foot_vel STATES (5,6,7,8)
        if(not config.mpc_params["use_foothold_optimization"]):
            foot_velocity_fl = foot_velocity_fl@0.0
            foot_velocity_fr = foot_velocity_fr@0.0
            foot_velocity_rl = foot_velocity_rl@0.0
            foot_velocity_rr = foot_velocity_rr@0.0
        linear_foot_vel_FL = foot_velocity_fl@(1-stanceFL)@(1-stance_proximity_FL)
        linear_foot_vel_FR = foot_velocity_fr@(1-stanceFR)@(1-stance_proximity_FR)
        linear_foot_vel_RL = foot_velocity_rl@(1-stanceRL)@(1-stance_proximity_RL)
        linear_foot_vel_RR = foot_velocity_rr@(1-stanceRR)@(1-stance_proximity_RR)

        # Integral states
        integral_states = states[24:30]
        integral_states[0] += states[2]
        integral_states[1] += states[3]
        integral_states[2] += states[4]
        integral_states[3] += states[5]
        integral_states[4] += roll
        integral_states[5] += pitch


 

        
        # The order of the return should be equal to the order of the states_dot
        return cs.vertcat(linear_com_vel,     # 3
                          linear_com_acc,     # 3
                          euler_rates_base,   # 3
                          angular_acc_base,   # 3 x 1
                          linear_foot_vel_FL, # 3 x 1 
                          linear_foot_vel_FR, # 3 x 1 
                          linear_foot_vel_RL, # 3 x 1 
                          linear_foot_vel_RR, # 3 x 1  ]
                          integral_states, # 6 x 1
                          q_dot_arm, # 3 x 1
                          q_ddot_arm # 3 x 1
                          )
    

        
    def export_robot_model(self,) -> AcadosModel:
        """
        This method set some general properties of the NMPC, such as the params,
        prediction mode, etc...! It will be called in arm_augmented_centroidal_nmpc.py
        """
 
        # dynamics
        self.param = cs.vertcat(self.stance_param, self.mu_friction, self.stance_proximity, self.base_position, 
                                self.base_yaw, self.external_wrench,self.k,self.d,self.q_arm_rest_1,self.q_arm_rest_2,self.q_arm_rest_3)
        f_expl = self.forward_dynamics(self.states, self.inputs, self.param)
        f_impl = self.states_dot - f_expl

        acados_model = AcadosModel()
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl
        acados_model.x = self.states
        acados_model.xdot = self.states_dot
        acados_model.u = self.inputs
        acados_model.p = self.param
        acados_model.name = "arm_augmented_centroidal_model"





        return acados_model
    
