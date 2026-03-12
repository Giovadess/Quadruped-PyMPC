
from gym_quadruped.utils.mujoco.visual import render_line,render_sphere
import numpy as np
import copy
import matplotlib.pyplot as plt
from quadruped_pympc import config

from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator

from scipy.spatial.transform import Rotation as R
import math



def compute_zmp(base_position,linear_acc,base_or,ext_wrenches,Pee):
    del base_or

    if not config.mpc_params['use_zmp_stability']:
        return np.array([base_position[0], base_position[1], 0.0])

    gravity = np.array([0.0, 0.0, -9.81])
    mass = 25.523
    robot_height = 0.35

    total_vertical_force = mass * (-gravity[2]) + ext_wrenches[2]
    safe_vertical_force = max(total_vertical_force, 10.0)

    zmp_x = mass * (-gravity[2]) * base_position[0] - robot_height * mass * linear_acc[0]
    zmp_x += Pee[0] * ext_wrenches[2] - Pee[2] * ext_wrenches[0]
    zmp_x /= safe_vertical_force

    zmp_y = mass * (-gravity[2]) * base_position[1] - robot_height * mass * linear_acc[1]
    zmp_y += Pee[1] * ext_wrenches[2] - Pee[2] * ext_wrenches[1]
    zmp_y /= safe_vertical_force

    return np.array([zmp_x, zmp_y, 0.0])


def compute_com_acc(base_position,base_or, forces):

    base_w = base_position

    yaw = base_or[2]
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)


    h_R_w=R.from_euler('xyz', base_or).as_matrix()

    # print('rotation matrix',h_R_w)
    # h_R_w=R.from_euler('z', base_or[2]).as_matrix()
    # print('rotation matrix',h_R_w)

    foot_force_fl = forces['FL']  #@param[0]
    foot_force_fr = forces['FR']  #@param[1]
    foot_force_rl = forces['RL']  #@param[2]
    foot_force_rr = forces['RR']  #@param[3]
    temp = foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    gravity = np.array([0, 0, -9.81])
    mass=24  #robot mass only the total mass is given by the external force along x!
    linear_com_acc = (1 / mass) * temp + gravity
    # ext_wrenches=np.zeros(6)

    
    return linear_com_acc

def compute_center_of_pressure(base_position,base_or,feet_pos,forces):
    foot_force_fl = forces['FL'][2]
    foot_force_fr = forces['FR'][2]
    foot_force_rl = forces['RL'][2]
    foot_force_rr = forces['RR'][2]
    base_w = copy.deepcopy(base_position)

    FL = copy.deepcopy(feet_pos['FL'])
    FR = copy.deepcopy(feet_pos['FR'])
    RL = copy.deepcopy(feet_pos['RL'])
    RR = copy.deepcopy(feet_pos['RR'])
    yaw = copy.deepcopy(base_or[2])

    yaw = base_or[2]
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)
    # breakpoint()
    FL[0:2] = h_R_w @ (FL[0:2] - base_w[0:2])
    FR[0:2] = h_R_w @ (FR[0:2] - base_w[0:2])
    RL[0:2] = h_R_w @ (RL[0:2] - base_w[0:2])
    RR[0:2] = h_R_w @ (RR[0:2] - base_w[0:2])

    temp=foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    numerator=(FL*foot_force_fl+FR*foot_force_fr+RL*foot_force_rl+RR*foot_force_rr)


    # print('temp',temp)

    

    if np.all(temp) ==0:
        cop=[0,0,0]
    else:
        cop=numerator/temp
    cop=[cop[0],cop[1]]

    cop=cop+base_w[0:2]
    return cop


def signed_distance(point,p1,p2):
    """
    Compute the signed distance from a point to the line defined by points p1 and p2.
    
    The sign of the distance indicates which side of the line the point lies on.
    """
    x0, y0 = point[0:2]
    x1, y1 = p1[0:2]
    x2, y2 = p2[0:2]
    numerator =(x2 - x1) * (y1-y0) - (y2 - y1) * (x1-x0)
    denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if denominator == 0:
        # raise ValueError("p1 and p2 cannot be the same point")
        margin=0
    margin=numerator / denominator
    return margin
    # if margin < 0.07:
    #     margin=0
    

def compute_zmp_margin(zmp,feet_pos,contact_state):
        # zmp_giulio=compute_zmp_giulio(base_pos[i],linear_com_acc,wrench_estimate[i],end_effector_pos[i],h_R_w)
        # zmp=zmp+base_pos[i][0:2] NO!

        # zmp=[zmp_exp[i][0],zmp_exp[i][1]]
        # print('zmp',zmp)
        # print('base pos',base_pos[i][0:2])
        # then I compute the zmp margin with respect to each side of the polygon
        list_margins=[]
        # zmp=zmp_exp[i][0:2]
        temp_marg_1=signed_distance(zmp,  feet_pos['FL'],feet_pos['FR'])#*contact_state[i]['FL']*contact_state[i]['FR']
        temp_marg_2=signed_distance(zmp,  feet_pos['FR'],feet_pos['RR'])#*contact_state[i]['FR']*contact_state[i]['RR']
        temp_marg_3=signed_distance(zmp,  feet_pos['RR'],feet_pos['RL'])#*contact_state[i]['RR']*contact_state[i]['RL']
        temp_marg_4=signed_distance(zmp,  feet_pos['RL'],feet_pos['FL'])#*contact_state[i]['RL']*contact_state[i]['FL']
        temp_marg_5_1=signed_distance(zmp,feet_pos['FL'],feet_pos['RR'])
        temp_marg_5_2=-temp_marg_5_1
        temp_marg_6_1=signed_distance(zmp,feet_pos['FR'],feet_pos['RL'])
        temp_marg_6_2=-temp_marg_6_1

        '''
        In the dls2 msg the order is not FL FR RL RR but 0 1 2 3
        '''

        # if contact_state['FL']*contact_state['FR']*contact_state['RL']*contact_state['RR']:
        if contact_state[0]*contact_state[1]*contact_state[2]*contact_state[3]:
            list_margins=(temp_marg_1,temp_marg_2,temp_marg_3,temp_marg_4)
        # elif contact_state['FL']*contact_state['FR']*contact_state['RL']:
        elif contact_state[0]*contact_state[1]*contact_state[2]:
            list_margins=(temp_marg_1,temp_marg_4,temp_marg_6_1)
        elif contact_state[0]*contact_state[1]*contact_state[3]:
            list_margins=(temp_marg_1,temp_marg_2,temp_marg_5_2)
            # print("Diagonal 5.2")
            # print("temp 5.2",temp_marg_5_2)
        elif contact_state[0]*contact_state[2]*contact_state[3]:
            list_margins=(temp_marg_4,temp_marg_3,temp_marg_5_1)
            # print("Diagonal 5.1 ")
            # print("temp 5.1",temp_marg_5_1)

        elif contact_state[1]*contact_state[2]*contact_state[3]:
            list_margins=(temp_marg_2,temp_marg_3,temp_marg_6_2)
        else:
            list_margins=[10.0,10.0,10.0,10.0]
        # zmp_margin_M=max(list_margins)
        zmp_margin_m=min(list_margins)
        # if zmp_margin_m<0:
        #     zmp_margin_m=0
        return zmp_margin_m



def compute_capture_point(robot_com, robot_com_vel):
    '''
    A Capture Point is a point on the ground where the robot can step to in order to bring itself to a complete stop.
    The formula is given according to "Capture Point: A Step Toward Humanoid Push Recovery" by Pratt et al.

    '''

    k= np.sqrt(9.81/robot_com[2])

    capture_point = robot_com[0:2] + robot_com_vel[0:2] / k

    return capture_point


