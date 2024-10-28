### Cuong Le - CVL, Linköping University, Sweden
### Optimal-state Dynamics Estimation for Physics-informed Human Motion Capture from Videos

import torch.nn as nn
import torch
from pytorch3d import transforms
from common.utils import get_rot_quat, get_pose_error

device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn         = nn.L1Loss(reduction='sum')    # L1 dist
loss_fn_c       = nn.BCELoss(reduction='sum')   # BCE loss for contacts

def compute_RA_p_loss(p_ra, pgt_ra):
    return loss_fn(p_ra, pgt_ra)

def compute_trans_loss(p_trans, pgt_trans):
    return loss_fn(p_trans, pgt_trans)

def compute_c_loss(c, cgt):
    return loss_fn_c(c, cgt)

def compute_v_loss(p_prime, p, pgt, pgt_pre):
    p_ra_moved      = (torch.sum(torch.abs((p_prime - p)), dim=-1, keepdim=True)).squeeze(-1)
    pgt_ra_moved    = (torch.sum(torch.abs((pgt - pgt_pre)), dim=-1, keepdim=True)).squeeze(-1)
    return loss_fn(p_ra_moved, pgt_ra_moved)

def compute_v_loss2(p_moved, pgt_t, pgt_p):
    pgt_moved       = (torch.sum(torch.abs((pgt_t - pgt_p)), dim=-1, keepdim=True)).squeeze(-1)
    return loss_fn(p_moved, pgt_moved)


# def compute_q_loss(q, qgt):
#     loss_trans      = loss_fn(q[...,:3], qgt[...,:3])
#     mat1            = transforms.quaternion_to_matrix(get_rot_quat(q))
#     mat2            = transforms.quaternion_to_matrix(get_rot_quat(qgt))
#     loss_rot        = loss_fn(mat1, mat2)
#     loss_joint      = torch.sum(torch.abs(torch.remainder(q[...,6:-1] - qgt[...,6:-1] + torch.pi, 2*torch.pi) - torch.pi))
#     return (0*loss_trans + loss_rot + loss_joint)

def compute_fric_loss(c, p_moved, mask_left, mask_right):
    fric_loss       = (((c[:,0:1]>0.5).float() * p_moved[:,[3,4]] * mask_left).sum() + 
                       ((c[:,1:2]>0.5).float() * p_moved[:,[8,9]] * mask_right).sum())
    # fric_loss       = ((c[:,0:1] * p_moved[:,[3,4]]).sum() + (c[:,1:2] * p_moved[:,[8,9]]).sum())
    return fric_loss