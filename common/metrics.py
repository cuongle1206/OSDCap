### Cuong Le - CVL, Linköping University, Sweden
### Optimal-state Dynamics Estimation for Physics-informed Human Motion Capture from Videos

import argparse
import torch
from utils import *
interest_joints_gt = j17_ids_gt
interest_joints_pr = j17_ids_pr

def procrustes(X, Y, scaling=True):
    # Reimplementation of MATLAB's `procrustes` function to Numpy.
    
    bx, nx, mx = X.shape
    by, ny, my = Y.shape
    muX     = torch.mean(X, dim=1, keepdim=True)
    muY     = torch.mean(Y, dim=1, keepdim=True)
    X0      = X - muX
    Y0      = Y - muY
    ssX     = torch.sum(torch.sum(X0**2., dim=-1, keepdim=True), dim=-2, keepdim=True)
    ssY     = torch.sum(torch.sum(Y0**2., dim=-1, keepdim=True), dim=-2, keepdim=True)

    # centred Frobenius norm
    normX   = torch.sqrt(ssX)
    normY   = torch.sqrt(ssY)

    # scale to equal (unit) norm
    X0      /= normX
    Y0      /= normY

    # optimum rotation matrix of Y
    A       = torch.bmm(X0.permute(0,2,1), Y0)
    U,s,Vt  = torch.linalg.svd(A,full_matrices=False)
    V       = Vt.permute(0,2,1)
    T       = torch.bmm(V, U.permute(0,2,1))

    V[:,:,-1] *= torch.sign(torch.linalg.det(T)).unsqueeze(1)
    s[:,-1]   *= torch.sign(torch.linalg.det(T))
    T       = torch.bmm(V, U.permute(0,2,1))
    traceTA = torch.sum(s, dim=-1, keepdim=True).unsqueeze(1)

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*torch.bmm(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*torch.bmm(Y0, T) + muX
    c = muX - b*torch.bmm(muY, T)

    return d, Z, T, b, c


def calculate_mpjpe(pgt, p):
    pgt_ra          = (pgt - pgt[:, 0:1, :])        # Root-aligned
    pgt_17_ra       = pgt_ra[:, interest_joints_gt, :] * 1e3   # from [m] to [mm]
    p_ra            = (p - p[:, 0:1, :])            # Root-aligned
    p17_ra          = p_ra[:, interest_joints_pr, :] * 1e3     # from [m] to [mm]
    # mpjpe           = torch.sum(torch.mean(torch.sqrt(torch.sum((pgt_17_ra - p17_ra)**2, dim=2)), dim=1))
    mpjpe           = torch.mean(torch.sqrt(torch.sum((pgt_17_ra - p17_ra)**2, dim=2)), dim=1)
    return mpjpe
    
def calculate_mpjpe_g(pgt, p):
    pgt_17          = pgt[:, interest_joints_gt, :] * 1e3      # from [m] to [mm]
    p17             = p[:, interest_joints_pr, :] * 1e3        # from [m] to [mm]
    mpjpe_g         = torch.mean(torch.sqrt(torch.sum((pgt_17 - p17)**2, dim=2)), dim=1)
    return mpjpe_g
    
def calculate_mpjpe_pa(pgt, p):
    pgt_17          = pgt[:, interest_joints_gt, :] * 1e3      # from [m] to [mm]
    p17             = p[:, interest_joints_pr, :] * 1e3        # from [m] to [mm]
    _, _, T, b, c   = procrustes(pgt_17, p17, scaling=True)
    frame_pred      = (b * torch.bmm(p17, T)) + c
    # mpjpe_pa        = torch.sum(torch.mean(torch.sqrt(torch.sum((pgt_17 - frame_pred)**2, dim=2)), dim=1))
    mpjpe_pa        = torch.mean(torch.sqrt(torch.sum((pgt_17 - frame_pred)**2, dim=2)), dim=1)
    return mpjpe_pa
    
def calculate_pck(pgt, p, threshold=150):
    pgt_ra          = pgt - pgt[:, 0:1, :]          # Root-aligned
    pgt_17_ra       = pgt_ra[:, interest_joints_gt, :] * 1e3   # from [m] to [mm]
    p_ra            = p - p[:, 0:1, :]              # Root-aligned
    p17_ra          = p_ra[:, interest_joints_pr, :] * 1e3     # from [m] to [mm]
    # pck             = torch.sum(torch.count_nonzero((torch.sqrt(torch.sum((pgt_17_ra[:,1:,:] - p17_ra[:,1:,:])**2, dim=2))<threshold), dim=1) / 16)
    pck             = torch.count_nonzero((torch.sqrt(torch.sum((pgt_17_ra[:,1:,:] - p17_ra[:,1:,:])**2, dim=2))<threshold), dim=1) / 16
    return pck

def calculate_accel(seq_pgt, seq_p):
    bsize, seq_len  = seq_p.shape[0], seq_p.shape[1]
    
    seq_pgt_ra      = seq_pgt - seq_pgt[:,:,0:1,:]
    seq_pgt_ra17    = seq_pgt_ra[:,:,interest_joints_gt,:] * 1e3
    seq_p_ra        = seq_p - seq_p[:,:,0:1,:]
    seq_p_ra17      = seq_p_ra[:,:,interest_joints_pr,:] * 1e3
    
    # dpgt            = (seq_pgt_ra17[:,1:,...] - seq_pgt_ra17[:,:-1,...])
    # ddpgt           = (dpgt[:,1:,...] - dpgt[:,:-1,...])
    # ddpgt_norm      = torch.norm(ddpgt, dim=-1)
    # dp              = (seq_p_ra17[:,1:,...] - seq_p_ra17[:,:-1,...])
    # ddp             = (dp[:,1:,...] - dp[:,:-1,...])
    # ddp_norm        = torch.norm(ddp, dim=-1)
    # accel           = torch.sum(torch.abs(ddpgt_norm-ddp_norm))
    # return accel
    
    accel_gt        = seq_pgt_ra17[:,:-2,...] - 2 * seq_pgt_ra17[:,1:-1,...] + seq_pgt_ra17[:,2:,...]
    accel_pred      = seq_p_ra17[:,:-2,...] - 2 * seq_p_ra17[:,1:-1,...] + seq_p_ra17[:,2:,...]
    normed          = torch.mean(torch.norm(accel_pred - accel_gt, dim=-1), dim=-1)
    
    # return torch.sum(torch.mean(normed, dim=-1))
    return torch.mean(normed, dim=-1)

def calculate_grp(pgt, p):
    pgt_17          = pgt[:, interest_joints_gt, :] * 1e3      # from [m] to [mm]
    p17             = p[:, interest_joints_pr, :] * 1e3        # from [m] to [mm]
    grp             = torch.mean(torch.sqrt(torch.sum((pgt_17[:,0:1,:] - p17[:,0:1,:])**2, dim=2)), dim=1)
    return grp

def calculate_cp(pgt, p, th):
    pgt_17          = pgt[:, interest_joints_gt, :] * 1e3      # from [m] to [mm]
    p17             = p[:, interest_joints_pr, :] * 1e3        # from [m] to [mm]
    _, _, T, b, c   = procrustes(pgt_17, p17, scaling=True)
    frame_pred      = (b * torch.bmm(p17, T)) + c
    cp              = torch.all(torch.sqrt(torch.sum((pgt_17 - frame_pred)**2, dim=2))<th, dim=1, keepdim=True).bool().int()
    return cp