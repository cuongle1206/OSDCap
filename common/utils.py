### Cuong Le - CVL, Linköping University, Sweden
### Collecting data from TRACE and H3.6M dataset

# Import nessesary libraries
import sys, os
repo_path = os.getcwd()
sys.path.append(repo_path + "/rbdl/build/python")
import time, rbdl
import numpy as np
import torch
from pytorch3d import transforms
import argparse

def get_parse():
    parser = argparse.ArgumentParser(description='OSDCap - Optimal-State Dynamics Capture')
    parser.add_argument("-dst", "--dataset", type=str, default="h36m")
    parser.add_argument("-abl", "--ablation", action="store_true")
    parser.add_argument("-osd", "--use_osd", action="store_false")
    parser.add_argument("-see", "--seed", type=int, default=0)
    parser.add_argument("-epo", "--num_epochs", type=int, default=15)
    parser.add_argument("-wup", "--warm_ups", type=int, default=5)
    parser.add_argument("-bsz", "--batch_size", type=int, default=64)
    parser.add_argument("-lnr", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-lrs", "--lr_steps", nargs='+', type=int, default=[10, 13])
    parser.add_argument("-lmd", "--loss_sc", nargs='+', type=float, default=[0.5, 5, 0.5, 5, 0.5, 1.0, 1.6])
    parser.add_argument("-gcl", "--grad_clip", type=float, default=0.8)
    parser.add_argument("-trn", "--train_models", action="store_true")
    parser.add_argument("-sav", "--save_model", action="store_false")
    parser.add_argument("-uwb", "--use_wandb", action="store_true")
    parser.add_argument("-var", "--abl_var", type=float, default=0.0)
    return parser

parser          = get_parse()
args            = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using GPU: " + torch.cuda.get_device_name(device=device))
humanoid            = rbdl.loadModel("urdf/smpl_human.urdf", floating_base=True)
humanoid.gravity    = [0, 0, -9.81]
q_size              = humanoid.q_size
qd_size             = humanoid.qdot_size
dt                  = 1/30 # Simulation rate 30Hz
# for j in range(len(humanoid.mBodies)):
#     print(j+3, humanoid.GetBodyName(j))

h36m_jids           = [0, 6,7,8,9,10,  1,2,3,4,5, 11,12, 16,17,18,19,22, 24,25,26,27,30, 13,14,15]
fit3d_jids          = [0, 1,2,3,17,18, 4,5,6,19,20, 0,7,  8,11,12,13,22,  8,14,15,16,24,   8,9,10]

if args.dataset != "sport":
    j17_ids_gt          = [0, 1,2,3, 6,7,8, 12, 23,24,25, 14,15,16, 19,20,21]
    j17_ids_pr          = [0, 1,2,3, 6,7,8, 12, 23,24,25, 14,15,16, 19,20,21]
else:
    j17_ids_gt          = [0, 12,14,16, 13,15,17, 6,8,10, 7,9,11]
    j17_ids_pr          = [0, 1,2,3, 6,7,8, 14,15,16, 19,20,21]

aist_jids           = range(0,17)
sport_jids          = range(0,18)


bone_length_h36m    = torch.Tensor( [[0.0, 0.0, 0.0],        # Root
                        [0.12719393604653, 0.0, 0.0],    # Left leg
                        [0.0, -0.43429100912261, 0.0],
                        [0.0, -0.448767017425826, 0.0],
                        [0.0, 0.0, 0.15150702088756],
                        [0.0, 0.0, 0.0749999965609858],
                        [-0.12719393604653, 0.0, 0.0],   # Right leg
                        [0.0, -0.43429100912261, 0.0],
                        [0.0, -0.448767017425826, 0.0],
                        [0.0, 0.0, 0.15150702088756],
                        [0.0, 0.0, 0.0749999965609858],
                        [0.0, 0.001, 0.0],     # Spine
                        [0.0, 0.245913012437421, 0.0],
                        [0.0, 0.248462964506057, 0.0],    # Left arm
                        [0.124881979654096, 0.0, 0.0], 
                        [0.259758046908119, 0.0, 0.0], 
                        [0.245542024364365, 0.0, 0.0],
                        [0.0750, 0.0, 0.0],
                        [0.0, 0.248462964506057, 0.0],    # Right arm
                        [-0.124881979654096, 0.0, 0.0], 
                        [-0.259758046908119, 0.0, 0.0], 
                        [-0.245542024364365, 0.0, 0.0],
                        [-0.0750, 0.0, 0.0],
                        [0.0, 0.248462964506057, 0.0],     # Head
                        [0.0, 0.0927524780924562, 0.0],
                        [0.0, 0.114999962169752, 0.0]]).to(device)

def print_log(str, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)

def get_dynamics(bsize, q, qd):
    # Detaching state variables q, qd
    q_np            = q.detach().clone().cpu().numpy().astype(float)
    qd_np           = qd.detach().clone().cpu().numpy().astype(float)
    M, gcc, com     = [], [], []
    for b in range(bsize):
        # Computing joint-state inertia matrix.
        M_np            = np.zeros((qd_size, qd_size))
        rbdl.CompositeRigidBodyAlgorithm(humanoid, q_np[b,:], M_np)
        M.append(torch.FloatTensor(M_np))
        
        # Computing graviational, Coriolis and centrifugal forces.
        gcc_np          = np.zeros(qd_size)
        rbdl.NonlinearEffects(humanoid, q_np[b,:], 0.1*qd_np[b,:], gcc_np)
        gcc.append(torch.FloatTensor(gcc_np))

        # Compute CoM
        com_np          = np.zeros(3)
        mass            = rbdl.CalcCenterOfMass(humanoid, q_np[0,:], np.zeros(66), com_np)
        com.append(torch.FloatTensor(com_np))
    
    M               = torch.stack(M,        dim=0).to(device)
    gcc             = torch.stack(gcc,      dim=0).to(device)
    com             = torch.stack(com,      dim=0).to(device)
    return mass, M, gcc, com

def get_rot_quat(q):
    return torch.cat((q[...,-1, None], q[...,3:6]), dim=-1)

def normalize_q_quat(q):
    q_rot_quat      = get_rot_quat(q)
    q_rot_quat_norm = q_rot_quat / torch.norm(q_rot_quat, dim=-1, keepdim=True)
    return torch.cat((q[...,:3], q_rot_quat_norm[...,1:], q[...,6:-1], q_rot_quat_norm[...,0:1]), dim=-1)

def get_pose_error(q_tar, q_pre):
    error_trans     = q_tar[...,:3]     - q_pre[...,:3]
    error_joint     = q_tar[...,6:-1]   - q_pre[...,6:-1]
    q_tar_quat      = get_rot_quat(q_tar)
    q_pre_quat      = get_rot_quat(q_pre)
    q_pre_quat_inv  = transforms.quaternion_invert(q_pre_quat)
    error_rot       = transforms.quaternion_multiply(q_tar_quat, q_pre_quat_inv)[...,1:]
    error           = torch.cat((error_trans, error_rot, error_joint), dim=-1)
    return error

def low_pass_filter(qh_pre, qh, alpha):
    # qh_new_trans    = (1-alpha) * qh_pre[...,:3] + alpha * qh[...,:3]
    qh_new_trans    = qh[...,:3]
    qh_new_rot      = get_rot_quat(qh)
    qh_new_jnt      = (1-alpha) * qh_pre[...,6:-1] + alpha * qh[...,6:-1]
    return torch.cat((qh_new_trans, qh_new_rot[...,1:], qh_new_jnt, qh_new_rot[...,0:1]), dim=-1)

def State_transition(bsize, q, qd, dt):
    """
    Non-linear state-transitioning (F)
    
    """
    
    # Root translation
    q_trans         = q[...,:3]     + qd[...,:3] * dt
    
    # Root rotation
    q_rot_quat      = get_rot_quat(q)
    qd_rot_quat     = torch.cat((torch.zeros(bsize, 1).to(device), qd[...,3:6]), dim=-1)
    q_rot           = q_rot_quat + 0.5 * transforms.quaternion_raw_multiply(qd_rot_quat, q_rot_quat) * dt
    q_rot_norm      = q_rot / torch.norm(q_rot, dim=-1, keepdim=True)
    
    # Joint rotations
    q_joint         = q[...,6:-1]   + qd[...,6:] * dt
    
    # Positional state vector from {t-1} to {t}
    q_new           = torch.cat((q_trans, q_rot_norm[...,1:], q_joint, q_rot_norm[...,0,None]), dim=-1)
    
    return q_new

def get_pose_diff(qh, q):
    """
    Similar to get pose error, but keeping the imaginary part of the quaternion
    """
    innova_trans    = qh[...,:3]   - q[...,:3]
    innova_joint    = qh[...,6:-1] - q[...,6:-1]
    q_rot_quat      = get_rot_quat(q)
    qh_rot_quat     = get_rot_quat(qh)
    innova_rot      = transforms.quaternion_multiply(qh_rot_quat, transforms.quaternion_invert(q_rot_quat))
    return torch.cat((innova_trans, innova_rot[...,1:], innova_joint, innova_rot[...,0:1]), dim=-1)


def Kalman_update(q, kalman, innovation):
    """
    Kalman update function (almost similar to state transistioning, but without velocity)
    
    x{t|t} = x{t|t-1} + K @ (yh{t} - H @ F @ x{t|t-1})
    
    """
    
    # Optimal-state update term
    delta           = (kalman @ innovation.unsqueeze(-1)).squeeze(-1)
    
    # Root translation
    q_trans         = q[...,:3]     + delta[...,:3]
    
    # Root rotation
    q_rot_quat      = get_rot_quat(q)
    delta_rot_quat  = get_rot_quat(delta)
    q_rot           = q_rot_quat + 0.5 * transforms.quaternion_raw_multiply(delta_rot_quat, q_rot_quat)
    q_rot_norm      = q_rot / torch.norm(q_rot, dim=-1, keepdim=True)
    
    # Joint rotations
    q_joint         = q[...,6:-1]   + delta[...,6:-1]
    
    # Optimal-state
    q_opt           = torch.cat((q_trans, q_rot_norm[...,1:], q_joint, q_rot_norm[...,0,None]), dim=-1)
    
    return q_opt

def fix_inverse_M(Minv):
    Minv_fix        = Minv.clone()
    unrot_id        = [9, 11, 15, 17, 21, 23, 27, 29, 42, 43, 54, 55]
    Minv_fix[:, unrot_id, :] = 0.
    return Minv_fix

def limit_torque(tau, lim_trans, lim_rot, lim_joints):
    tau_trans       = torch.clamp(tau[:,:3],  -lim_trans,  lim_trans)
    tau_rot         = torch.clamp(tau[:,3:6], -lim_rot,    lim_rot)
    tau_joint       = torch.clamp(tau[:,6:],  -lim_joints, lim_joints)
    return torch.cat((tau_trans, tau_rot, tau_joint), dim=1)

def forward_kinematics(bsize, q, bone_length):
    
    q_trans, q_rot, q_joint  = q[...,:3], get_rot_quat(q), q[...,6:-1]
    mat_rot         = transforms.quaternion_to_matrix(q_rot)
    mat_joint       = transforms.euler_angles_to_matrix(q_joint.reshape(bsize,-1,3), "ZXY")
    mat_all         = torch.cat((mat_rot.unsqueeze(1), mat_joint), dim=1)
    
    joints          = [0,1,2,3,4, 0,5,6,7, 8,  0, 9, 10,11,12,13,14, 10,15,16,17,18, 10,19,20]
    parents         = [0,1,2,3,4, 0,6,7,8, 9,  0,11, 12,13,14,15,16, 12,18,19,20,21, 12,23,24]
    bones           = [1,2,3,4,5, 6,7,8,9,10, 11,12, 13,14,15,16,17, 18,19,20,21,22, 23,24,25]
    bone_length_opt = bone_length.unsqueeze(0).repeat(bsize, 1, 1).to(device)
    R_list, P_list  = [], []
    P_list.append(q_trans)
    R_list.append(torch.eye(3).reshape((1, 3, 3)).repeat(bsize, 1, 1).to(device))
    for v in range(len(joints)):
        R               = torch.bmm(R_list[parents[v]], mat_all[:,joints[v],...])
        P_list.append(P_list[parents[v]] + torch.bmm(R, bone_length_opt[:,bones[v],:].unsqueeze(-1)).squeeze(-1))
        R_list.append(R)
    p_recon         = torch.stack(P_list, dim=1)
    return p_recon
    
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles