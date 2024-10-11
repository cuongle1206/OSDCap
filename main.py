### Cuong Le - CVL, Linköping University, Sweden
### Optimal-state Dynamics Estimation for Physics-based Human Motion Capture from Videos

# Import libraries
import sys, os
repo_path = os.getcwd()
sys.path.append(repo_path + "/rbdl/build/python")
import argparse, rbdl, math
import numpy as np
from tqdm import tqdm

# Logging and visualization
import wandb
import shutil, imageio
import matplotlib, cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

# Utilities and Net architecture
from common.networks import OSDNet
from common.metrics import *
from common.losses import *
from common.utils import *

# Training nessesities
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3d import transforms

from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import time
from PIL import Image

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

def Processor_OSDNet(args, osdnet, model_path, mode):
    
    if mode == 'train':
        # Load training data
        print('--- Loading train data. ---')
        train_set       = torch.load("datasets/" + args.dataset + "/" + experiment + "/train_set.pt")
        loader          = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        
        # Setting up optimizer
        print('--- Setting up optimizer. ---')
        if args.dataset == "sport":
            print(model_path + "_h36m.pth")
            osdnet.load_state_dict(torch.load(model_path + "_h36m.pth"))
        osdnet.train()
        optimizer       = torch.optim.Adamax(osdnet.parameters(), lr=args.learning_rate)
        scheduler       = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
        num_batches     = math.ceil(len(train_set)/args.batch_size)
        lr_steps        = args.lr_steps
        Lambda          = args.loss_sc
        num_epochs      = args.num_epochs
        print('--- Training started! ---')
        print()
        
    if mode == 'test':
        # Load testing data
        print('--- Loading test data. ---')
        test_set        = torch.load("datasets/" + args.dataset + "/" + experiment + "/test_set.pt")
        loader          = DataLoader(dataset=test_set,  batch_size=args.batch_size, shuffle=False)
        num_batches     = math.ceil(len(test_set)/args.batch_size)
        num_epochs      = 1
        
        # Load trained model
        print('--- Loading model. ---')
        print(model_path)
        osdnet.load_state_dict(torch.load(model_path + ".pth"))
        osdnet.eval()
        total_mpjpe_trace, total_mpjpe_osd = [], []
        total_mpjpe_g_trace, total_mpjpe_g_osd = [], []
        total_mpjpe_pa_trace, total_mpjpe_pa_osd = [], []
        total_pck_trace, total_pck_osd = [], []
        total_grp_trace, total_grp_osd = [], []
        total_accel_trace, total_accel_osd = [], []
        total_cp_trace, total_cp_osd = [], []
        
        # Rebuttal additions
        total_gp_trace, total_gp_osd = [], []
        total_v_trace, total_v_osd = [], []
        total_fric_trace, total_fric_osd = [], []
        total_skate_trace, total_skate_osd = [], []
        
        result_qh_all, result_q_all, result_qsim_all = [], [], []
        result_ph_all, result_p_all, result_pgt_all = [], [], []
        result_kalman_all, result_minv_all, result_c_all = [], [], []
        result_extf_all = []
        
        print('--- Testing started! ---')
        print()
        
    ########## Main processing loop ##########
    for epoch in range(num_epochs):
        if mode == 'train':
            # Warmup strategy and learing rate scheduler.
            if epoch < args.warm_ups:
                lr  = args.learning_rate * (epoch + 1) / args.warm_ups
                for i, param_group in enumerate(optimizer.param_groups): param_group['lr'] = lr
            # print_log('\tEpoch: {:d}, learning rate: {:.2e}.'.format(epoch+1, optimizer.param_groups[0]['lr']))
            print_log('\t---------- Epoch {:d} ----------'.format(epoch+1))
            print_log('\tTotal epochs: {:d}, LR: {:.2e}'.format(num_epochs, optimizer.param_groups[0]['lr']))
            print_log('\tLearning rate steps at: {:d}, {:d}'.format(lr_steps[0], lr_steps[1]))
            print_log('\tLoss scaling: {:.2f}'.format(Lambda[-1]))
            print_log('\tLoss weighting: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(
                Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4], Lambda[5]))
        
        # Processing loop
        loss_values     = []
        loop            = tqdm(enumerate(loader),
                               total=num_batches,
                               bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                               position=0)
        for batch_id, batch in loop:
            
            # Only train and evaluate on subset of interested keypoints
            if args.dataset == "h36m":  interest_joints = h36m_jids
            if args.dataset == "fit3d": interest_joints = fit3d_jids
            if args.dataset == "aist":  interest_joints = aist_jids
            if args.dataset == "sport":  interest_joints = sport_jids

            # Load TRACE inputs and ground-truth from pre-processes data
            seq_qh          = batch[0].to(device, dtype=torch.float)    # Measurements
            seq_qgt         = batch[1].to(device, dtype=torch.float)    # GT poses
            seq_pgt         = batch[2].to(device, dtype=torch.float)[..., interest_joints, :] # GT 3d key-points
            seq_cgt         = batch[3].to(device, dtype=torch.float)    # Pseudo-GT contacts
            
            """Using offline temporal filtering: Median or Gaussian"""
            # temp_seq_qh     = seq_qh.detach().cpu().numpy()
            # for b in range(seq_qh.shape[0]):
            #     for c in range(seq_qh.shape[2]):
            #         # temp_seq_qh[b,:,c] = medfilt(seq_qh[b,:,c].detach().cpu())
            #         temp_seq_qh[b,:,c] = gaussian_filter1d(seq_qh[b,:,c].detach().cpu(), 1)
            # seq_qh          = torch.from_numpy(temp_seq_qh).to(device)
            
            bsize, seq_len  = seq_qh.shape[0], seq_qh.shape[1] - 1      # Skiping first frame during simulation

            # Initital states.
            h_gru           = osdnet.init_hidden(bsize)                     # init GRUs hidden state
            qh_pre          = seq_qh[:,0,:]                                 # qh_{t-1}
            qdd             = torch.zeros((bsize, qd_size)).to(device)      # qdd_{t-1}
            input_offset    = torch.cat((osdnet.offset, torch.zeros(q_size-3).to(device))).unsqueeze(0)
            qd              = (torch.bmm(osdnet.qd_mapp.repeat((bsize,1,1)),
                                         get_pose_error(seq_qh[:,1,:], seq_qh[:,0,:]).unsqueeze(-1)).squeeze() + 
                                osdnet.qd_bias.repeat((bsize,1)).to(device)) # qd_{t-1|t-1}
            q               = normalize_q_quat(torch.bmm(osdnet.q_mapp.repeat((bsize,1,1)),
                                        (seq_qh[:,0,:] + input_offset).unsqueeze(-1)).squeeze()) # q_{t-1|t-1}
            p               = forward_kinematics(bsize, q, osdnet.bone_length)  # [B, V, 3]
            # p_trace_pre     = forward_kinematics(bsize, q, bone_length_h36m)    # [B, V, 3]
            p_moved         = torch.zeros(bsize, p.shape[1]).to(device)         # [B, V]
            c               = seq_cgt[:,0,:]                                    # [B, 2]
            
            H_obs           = osdnet.obs_model.unsqueeze(0).repeat(bsize, 1, 1) # Observation model H (Optimized)
            C_adp           = osdnet.adp_model.unsqueeze(0).repeat(bsize, 1, 1) # Adaptation model C (Optimized)
            
            observation_diff    = torch.zeros((bsize, q_size)).to(device)
            innovation_diff     = torch.zeros((bsize, q_size)).to(device)
            fw_evolution_diff   = torch.zeros((bsize, q_size)).to(device)
            fw_update_diff      = torch.zeros((bsize, q_size)).to(device)
            
            # Initialize the accumulated losses (training).
            loss_p, loss_t, loss_psim, loss_qsim, loss_c, loss_v = 0, 0, 0, 0, 0, 0
            if mode == 'train':
                p_ra            = (p - p[:,0:1,:])[:,1:,:]
                pgt_ra          = (seq_pgt[:,0,...] - seq_pgt[:,0,0:1,:])[:,1:,:]
                # if args.dataset != "sport":
                #     loss_p          += compute_RA_p_loss(p_ra, pgt_ra) / bsize
                #     loss_t          += compute_trans_loss(q, seq_qgt[:,0,:]) / bsize
                # else:
                #     loss_p          += compute_RA_p_loss(p_ra[:,j17_ids_pr], pgt_ra[:,j17_ids_gt]) / bsize
                #     loss_t          += compute_trans_loss(q, seq_qgt[:,0,:]) / bsize
            
            # Initialize the accumulated evaluation metrics (testing).
            mpjpe_trace, mpjpe_g_trace, mpjpe_pa_trace, grp_trace, pck_trace    = 0, 0, 0, 0, 0
            mpjpe_osd, mpjpe_g_osd, mpjpe_pa_osd, grp_osd, pck_osd              = 0, 0, 0, 0, 0
            cp_trace, cp_osd = torch.zeros(300, bsize, 1).to(device), torch.zeros(300, bsize, 1).to(device)
            
            result_qh, result_qsim, result_q        = [], [], []
            result_ph, result_p, result_pgt         = [], [], []
            result_kalman, result_minv, result_c    = [], [], []
            result_extf                             = []
            
            # Rebuttal additions
            gp_dist, v_diff, fric_dist = 0, 0, 0
            fk_count = torch.zeros(bsize).to(device)
            
            ########## Loop thru all sequences frame-by-frame (batch-wise) ##########
            for f in range(seq_len):
                # start_time = time.time()
                # Observation and ground-truths.
                qh              = seq_qh[:,f+1,:]       # qh_{t}
                qgt             = seq_qgt[:,f+1,:]      # GT pose (t+1)
                pgt             = seq_pgt[:,f+1,...]    # GT kinematics (t+1)
                cgt             = seq_cgt[:,f+1,...]    # Pseudo-GT contacts (t+1)
                
                # Get dynamics variables from RBDL proxy character.
                mass, M, gcc, com = get_dynamics(bsize, q, qd)      # [1], [B,66,66], [B,66], [B,3]
                
                # Using NN to predict optimal error and Kp, Kd
                error_h         = get_pose_error(qh, q) # Current error
                kalman_in       = torch.cat((observation_diff, innovation_diff, fw_evolution_diff, fw_update_diff), dim=-1).unsqueeze(1) # [B, 1, 268]
                state_vector    = torch.cat((p.view(bsize, -1), q, qd, qh, error_h, c, com, M.view(bsize,-1)), dim=-1).unsqueeze(1) # [B, 1, 412]
                contact_in      = torch.cat((p[:,:11,:].view(bsize,-1), p_moved[:,:11], q[:,6:30], c, com), dim=-1).unsqueeze(1) # [B, 1, 73]
                
                ###### Using OSD #####
                if args.use_osd:
                    # OSDNet does its things!
                    kalman, h_gru, PDgains, M_bias, c_prime, fext, jacs = osdnet(state_vector, kalman_in, contact_in, h_gru)
                    # print(torch.diag(kalman.mean(0)).mean())
                    
                    ######################## Optimal-state filtering ########################
                    """State-transitioning and system observation""" # mid-point method
                    q_predict       = State_transition(bsize, q, qd, dt)     # x_{t|t-1} = [F] x x_{t-1|t-1}
                    p_predict       = forward_kinematics(bsize, q_predict, osdnet.bone_length)
                    system          = torch.bmm(H_obs, q_predict.unsqueeze(-1)).squeeze(-1)     # y^hat_{t|t-1}
                    
                    """Get observation from TRACE"""
                    qh_fixed        = qh.clone()                                                # y_{t}
                    error_inno      = quaternion_to_axis_angle(get_rot_quat(get_pose_diff(qh, q))) # 
                    e_id            = torch.where(abs(error_inno[:,2]) > 1.6)
                    if len(e_id[0]):
                        # Set global rotation to simulation stream if the input is too noisy
                        qh_fixed[e_id[0],3:]    = system.clone()[e_id[0],3:]
                        qd                      = qd.clone() * 0.
                    qh_pre          = low_pass_filter(qh_pre, qh_fixed, osdnet.comp_filter)           # complementary
                    random_noise    = (torch.cat((5e-3* torch.randn(qh_fixed.shape[0], qh_fixed.shape[1]-1).to(device),
                                                  torch.zeros(bsize,1).to(device)), dim=-1) if mode == 'train' else 0)
                    qh_fixed_obs    = torch.bmm(C_adp, qh_pre.unsqueeze(-1)).squeeze(-1) + random_noise
                    observation     = normalize_q_quat(qh_fixed_obs.clone())        # y_{t}
                    
                    """Optimal-state filtering"""
                    observation_diff    = get_pose_diff(observation, q)             # y_{t} - sum(y_{:t-1})
                    innovation_diff     = get_pose_diff(observation, system)        # y_{t} - y^hat_{t|t-1}
                    
                    # Kalman filtering -- updating positional state.
                    q_prime         = Kalman_update(q_predict, kalman, innovation_diff) # x_{t|t} = x_{t|t-1} + K @ inno
                    p_prime         = forward_kinematics(bsize, q_prime, osdnet.bone_length)
                    p_moved         = (torch.sum(torch.abs((p_prime - p)), dim=-1, keepdim=True)).squeeze(-1)

                    fw_evolution_diff   = get_pose_diff(q_prime, q)                  # x_{t|t} - x_{t-1|t-1}
                    fw_update_diff      = get_pose_diff(q_prime, q_predict)          # x_{t|t} - x_{t|t-1}

                    """Physics simulation"""
                    Minv            = torch.inverse(M) + M_bias                         # Inverse of JSIM with additional bias
                    grf             = torch.sum(torch.einsum('bcie,bcet->bcit', jacs, (c_prime[:,:,None] * fext).unsqueeze(-1)), dim=1)
                    grf_prime       = grf.clone() - ((c_prime[:,0:1,None]>0.5).float() * (c_prime[:,1:2,None]>0.5).float() * 
                                                    c_prime.sum(1).view(-1,1,1) * osdnet.grf_comp * grf)
                    torque_sim, sum_force_sim = [], []
                    qdd_sim, qd_sim, q_sim = [], [], []
                    qdd_sim.append(qdd.clone())
                    qd_sim.append(qd.clone())
                    q_sim.append(q_predict.clone())
                    for step in range(2): # Runge-Kutta 2nd order integrator
                        torque_sim.append(PDgains[0] @ get_pose_error(q_prime, q_sim[step]).unsqueeze(-1) - PDgains[1] @ qd_sim[step].unsqueeze(-1))
                        sum_force_sim.append(limit_torque(mass * (torque_sim[step] + grf_prime) - gcc.unsqueeze(-1), 1500, 500, 100))
                        qdd_sim.append(torch.bmm(Minv, sum_force_sim[step]).squeeze(-1))
                        qd_sim.append(qd_sim[step] + qdd_sim[step+1] * dt/2)
                        q_sim.append(State_transition(bsize, q_sim[step], qd_sim[step+1], dt/2))
                    qdd_prime       = qdd_sim[-1].clone()
                    qd_prime        = qd_sim[-1].clone()
                    
                    if mode == 'train':
                        p_ra            = (p - p[:,0:1,:])[:,1:,:]
                        p_prime_ra      = (p_prime - p_prime[:,0:1,:])[:,1:,:]
                        p_predict_ra    = (p_predict - p_predict[:,0:1,:])[:,1:,:]
                        pgt_ra          = (pgt - pgt[:,0:1,:])[:,1:,:]
                        pgt_ra_pre      = (seq_pgt[:,f,...] - seq_pgt[:,f,0:1,:])[:,1:,:]
                        if args.dataset != "sport":
                            loss_p          += compute_RA_p_loss(p_prime_ra, pgt_ra) / bsize
                            loss_t          += compute_trans_loss(q_prime, qgt) / bsize
                            loss_psim       += compute_RA_p_loss(p_predict_ra, pgt_ra) / bsize
                            loss_qsim       += compute_trans_loss(q_predict, qgt) / bsize
                            loss_c          += compute_c_loss(c_prime, cgt) / bsize
                            loss_v          += compute_v_loss(p_prime_ra, p_ra, pgt_ra, pgt_ra_pre) / bsize
                        else:
                            loss_p          += compute_RA_p_loss(p_prime_ra[:,j17_ids_pr], pgt_ra[:,j17_ids_gt]) / bsize
                            loss_t          += compute_trans_loss(q_prime, qgt) / bsize
                            loss_psim       += compute_RA_p_loss(p_predict_ra[:,j17_ids_pr], pgt_ra[:,j17_ids_gt]) / bsize
                            loss_qsim       += compute_trans_loss(q_predict, qgt) / bsize
                            loss_c          += compute_c_loss(c_prime, cgt) / bsize
                            loss_v          += compute_v_loss(p_prime_ra, p_ra, pgt_ra, pgt_ra_pre) / bsize
                   
                ###### Using PD controller #####
                else:
                    # PD gains prediction
                    PDgains, M_bias, c_prime, fext, jacs = osdnet(state_vector, kalman_in, contact_in, h_gru) # kalman_in not in use.
                    
                    Minv            = torch.inverse(M) + M_bias                         # Inverse of JSIM with additional bias
                    grf             = torch.sum(torch.einsum('bcie,bcet->bcit', jacs, (c_prime[:,:,None] * fext).unsqueeze(-1)), dim=1)
                    grf_prime       = grf.clone() - ((c_prime[:,0:1,None]>0.5).float() * (c_prime[:,1:2,None]>0.5).float() * 
                                                    c_prime.sum(1).view(-1,1,1) * osdnet.grf_comp * grf)
                    qh_fixed        = qh.clone()                                                # y_{t}
                    error_inno      = quaternion_to_axis_angle(get_rot_quat(get_pose_diff(qh, q))) # 
                    e_id            = torch.where(abs(error_inno[:,2]) > 1.6)
                    if len(e_id[0]):
                        # Set global rotation to simulation stream if the kinematics fails
                        qh_fixed[e_id[0],3:]    = q.clone()[e_id[0],3:]
                        qd                      = qd.clone() * 0.
                    qh_pre          = low_pass_filter(qh_pre, qh_fixed, osdnet.comp_filter)     # complementary
                    
                    torque_sim, sum_force_sim = [], []
                    qdd_sim, qd_sim, q_sim = [], [], []
                    qdd_sim.append(qdd.clone())
                    qd_sim.append(qd.clone())
                    q_sim.append(q.clone())
                    for step in range(2):
                        torque_sim.append(PDgains[0] @ get_pose_error(qh_pre, q_sim[step]).unsqueeze(-1) - PDgains[1] @ qd_sim[step].unsqueeze(-1))
                        sum_force_sim.append(limit_torque(mass * (torque_sim[step] + grf_prime) - gcc.unsqueeze(-1), 1500, 500, 100))
                        qdd_sim.append(torch.bmm(Minv, sum_force_sim[step]).squeeze(-1))
                        qd_sim.append(qd_sim[step] + qdd_sim[step+1] * dt/2)
                        q_sim.append(State_transition(bsize, q_sim[step], qd_sim[step+1], dt/2)) # Heun's method
                    qdd_prime       = qdd_sim[-1].clone()
                    qd_prime        = qd_sim[-1].clone()
                    q_prime         = q_sim[-1].clone()
                    p_prime         = forward_kinematics(bsize, q_prime, osdnet.bone_length)
                    p_moved         = (torch.sum(torch.abs((p_prime - p)), dim=-1, keepdim=True)).squeeze(-1)
                    
                    if mode == 'train':
                        p_ra            = (p - p[:,0:1,:])[:,1:,:]
                        p_prime_ra      = (p_prime - p_prime[:,0:1,:])[:,1:,:]
                        pgt_ra          = (pgt - pgt[:,0:1,:])[:,1:,:]
                        pgt_ra_pre      = (seq_pgt[:,f,...] - seq_pgt[:,f,0:1,:])[:,1:,:]
                        loss_p          += compute_RA_p_loss(p_prime_ra, pgt_ra) / bsize
                        loss_t          += compute_trans_loss(q_prime, qgt) / bsize
                        loss_c          += compute_c_loss(c_prime, cgt) / bsize
                        loss_v          += compute_v_loss(p_prime_ra, p_ra, pgt_ra, pgt_ra_pre) / bsize
                
                if mode == 'test':
                    # p_trace         = forward_kinematics(bsize, qh, osdnet.bone_length)
                    p_trace         = forward_kinematics(bsize, qh,  bone_length_h36m)
                    
                    mpjpe_trace     += calculate_mpjpe(pgt, p_trace)
                    mpjpe_g_trace   += calculate_mpjpe_g(pgt, p_trace)
                    mpjpe_pa_trace  += calculate_mpjpe_pa(pgt, p_trace)
                    grp_trace       += calculate_grp(pgt, p_trace)
                    pck_trace       += calculate_pck(pgt, p_trace)
                    
                    mpjpe_osd       += calculate_mpjpe(pgt, p_prime)
                    mpjpe_g_osd     += calculate_mpjpe_g(pgt, p_prime)
                    mpjpe_pa_osd    += calculate_mpjpe_pa(pgt, p_prime)
                    grp_osd         += calculate_grp(pgt, p_prime)
                    pck_osd         += calculate_pck(pgt, p_prime)
                    
                    for th in range(1, 301):
                        cp_trace[th-1, ...] += calculate_cp(pgt, p_trace, th)
                        cp_osd[th-1, ...] += calculate_cp(pgt, p_prime, th)
                    
                     # Saving the poses for testing
                    result_qh.append(qh)
                    result_q.append(q_prime)
                    
                    result_p.append(p_prime)
                    result_ph.append(p_trace)
                    result_pgt.append(pgt)
                    
                    if args.use_osd:
                        result_qsim.append(qgt)
                        result_kalman.append(kalman)
                        
                    result_minv.append(Minv)
                    result_c.append(c_prime)
                    result_extf.append(c_prime[:,:,None] * fext)

                    # c_mask_left              = torch.zeros(bsize,3).to(device)
                    # c_mask_left[pgt[:,3,2] < 0.12,0] = 1
                    # c_mask_left[pgt[:,4,2] < 0.072,1] = 1
                    # c_mask_left[pgt[:,5,2] < 0.072,2] = 1
                    
                    # c_mask_right             = torch.zeros(bsize,3).to(device)
                    # c_mask_right[pgt[:,8,2] < 0.12,0] = 1
                    # c_mask_right[pgt[:,9,2] < 0.072,1] = 1
                    # c_mask_right[pgt[:,10,2] < 0.072,2] = 1
                    
                    # p_interest          = p_trace
                    # # p_interest          = p_prime
                    
                    # # gp_dist_left        = (c_mask_left * nn.functional.relu(pgt[:,[3,4,5],2] - p_interest[:,[3,4,5],2])).mean(1)  * cgt[:,0]
                    # # gp_dist_right       = (c_mask_right * nn.functional.relu(pgt[:,[8,9,10],2] - p_interest[:,[8,9,10],2])).mean(1) * cgt[:,1]
                    # # gp_dist             += ((gp_dist_left + gp_dist_right)/2).mean()
                    
                    # gp_dist_left        = (c_mask_left * torch.abs(pgt[:,[3,4,5],2] - p_interest[:,[3,4,5],2])).mean(1)  * cgt[:,0]
                    # gp_dist_right       = (c_mask_right * torch.abs(pgt[:,[8,9,10],2] - p_interest[:,[8,9,10],2])).mean(1) * cgt[:,1]
                    # gp_dist             += ((gp_dist_left + gp_dist_right)/2).mean()
                    
                    # p_moved1             = torch.sqrt(torch.sum((p_prime - p)**2, dim=-1))
                    # p_moved2            = torch.sqrt(torch.sum((p_trace - p_trace_pre)**2, dim=-1))
                    # fric_dist           += compute_fric_loss(cgt, p_moved2, c_mask_left[...,:2], c_mask_right[...,:2]) / bsize
                    # v_diff              += compute_v_loss(p_moved2, pgt, seq_pgt[:,f,:]) / bsize
                    
                    # mov_left            = cgt[:,0] * p_moved2[:,[3,4,5]].mean(-1) * c_mask_left[:,1]
                    # mov_right           = cgt[:,1] * p_moved2[:,[8,9,10]].mean(-1) * c_mask_right[:,1]
                    
                    # skate               = (mov_left>0.02) + (mov_right>0.02)
                    # fk_count            += skate
                    
                # Updating the states
                qd, q, p, c = qd_prime, q_prime, p_prime, c_prime
                # print("--- %s seconds ---" % (time.time() - start_time))
                
            if mode == 'train':
                # Scheduling the smooth losses
                v_scale     = (Lambda[5] if epoch > args.warm_ups else 0.)
                if args.use_osd:
                    sample_loss     = Lambda[-1] * (Lambda[0] * loss_p + Lambda[1] * loss_t +
                                                    Lambda[2] * loss_psim + Lambda[3] * loss_qsim + 
                                                    Lambda[4] * loss_c + v_scale * loss_v)
                    loss_values.append([loss_p.data.item(), loss_t.data.item(), loss_psim.data.item(), loss_qsim.data.item(),
                                        loss_c.data.item(), loss_v.data.item(), sample_loss.data.item()])
                else:
                    sample_loss     = Lambda[-1] * (Lambda[0] * loss_p + Lambda[1] * loss_t +
                                                    Lambda[4] * loss_c + Lambda[5] * loss_v)
                    loss_values.append([loss_p.data.item(), loss_t.data.item(), loss_c.data.item(), loss_v.data.item(),
                                        sample_loss.data.item()])
                # Optimization, clipping the gradient to prevent explosion at the very beginning.
                optimizer.zero_grad()
                sample_loss.backward()
                nn.utils.clip_grad_norm_(osdnet.parameters(), args.grad_clip)
                optimizer.step()
                loop.set_postfix(loss = sample_loss.item())
            
            if mode == 'test':
                # Save all poses
                result_qh       = torch.stack(result_qh, dim=1)
                result_qh_all.append(result_qh)
                result_q        = torch.stack(result_q, dim=1)
                result_q_all.append(result_q)
                if args.use_osd: 
                    result_qsim      = torch.stack(result_qsim, dim=1)
                    result_qsim_all.append(result_qsim)
                
                # Save all kinematics
                result_ph       = torch.stack(result_ph, dim=1)
                result_ph_all.append(result_ph)
                result_p        = torch.stack(result_p, dim=1)
                result_p_all.append(result_p)
                result_pgt      = torch.stack(result_pgt, dim=1)
                result_pgt_all.append(result_pgt)
                
                # Save the kalman gains and Minv
                if args.use_osd: 
                    result_kalman       = torch.stack(result_kalman, dim=1)
                    result_kalman_all.append(result_kalman)
                result_minv     = torch.stack(result_minv, dim=1)
                result_minv_all.append(result_minv)
                result_c        = torch.stack(result_c, dim=1)
                result_c_all.append(result_c)
                result_extf     = torch.stack(result_extf, dim=1)
                result_extf_all.append(result_extf)
                
                # Appending total evaluation metrics (for TRACE)
                total_mpjpe_trace.append(mpjpe_trace/seq_len)       
                total_mpjpe_g_trace.append(mpjpe_g_trace/seq_len)
                total_mpjpe_pa_trace.append(mpjpe_pa_trace/seq_len)
                total_pck_trace.append(pck_trace * 1e2/seq_len)
                total_grp_trace.append(grp_trace/seq_len)
                total_accel_trace.append(calculate_accel(result_pgt, result_ph))
                total_cp_trace.append(cp_trace * 1e2/seq_len)
                
                # Appending total evaluation metrics (for OSD)
                total_mpjpe_osd.append(mpjpe_osd/seq_len)
                total_mpjpe_g_osd.append(mpjpe_g_osd/seq_len)
                total_mpjpe_pa_osd.append(mpjpe_pa_osd/seq_len)
                total_pck_osd.append(pck_osd * 1e2/seq_len)
                total_grp_osd.append(grp_osd/seq_len)
                total_accel_osd.append(calculate_accel(result_pgt, result_p))
                total_cp_osd.append(cp_osd * 1e2/seq_len)
                
                # total_gp_osd.append(gp_dist.detach().cpu()/seq_len)
                # total_fric_osd.append(fric_dist.detach().cpu()/seq_len)
                # total_v_osd.append(v_diff.detach().cpu()/seq_len)
                # total_fk_osd.append((fk_count.detach().cpu()/seq_len).mean())
        
        if mode == 'train':
            scheduler.step() # LR scheduler counts
            mean_loss   = np.mean(loss_values, axis=0) # Averaging the losses across all batches.
            if args.use_wandb: wandb.log({'epoch': epoch+1})
            if args.use_osd:    names = ['p_loss', 't_loss', 'ps_loss', 'ts_loss', 'c_loss', 'v_loss', 'sample']              
            else:               names = ['p_loss', 't_loss', 'c_loss', 'v_loss', 'sample']
            for i in range(mean_loss.shape[-1]):
                print_log('\tMean training {}: \t{:.2f}.'.format(names[i], mean_loss[i]))
                if args.use_wandb: wandb.log({names[i]: mean_loss[i]})
            print('------------------------------')

    if (mode == 'train') and (args.save_model):
        torch.save(osdnet.state_dict(), model_path + ".pth")
        print('--- Training finished! ---')
        print("Model is saved at: " + model_path)
        print('------------------------------')
    
    if (mode == 'test'):
        total_mpjpe_trace       = torch.cat(total_mpjpe_trace, dim=0)
        total_mpjpe_g_trace     = torch.cat(total_mpjpe_g_trace, dim=0)
        total_mpjpe_pa_trace    = torch.cat(total_mpjpe_pa_trace, dim=0)
        total_pck_trace         = torch.cat(total_pck_trace, dim=0)
        total_grp_trace         = torch.cat(total_grp_trace, dim=0)
        total_accel_trace       = torch.cat(total_accel_trace, dim=0)
        total_cp_trace          = torch.mean(torch.cat(total_cp_trace, dim=1), dim=1).squeeze()
        auc_cps_trace           = torch.sum(total_cp_trace.detach().cpu()*1e-2).item()
        
        total_mpjpe_osd         = torch.cat(total_mpjpe_osd, dim=0)
        total_mpjpe_g_osd       = torch.cat(total_mpjpe_g_osd, dim=0)
        total_mpjpe_pa_osd      = torch.cat(total_mpjpe_pa_osd, dim=0)
        total_pck_osd           = torch.cat(total_pck_osd, dim=0)
        total_grp_osd           = torch.cat(total_grp_osd, dim=0)
        total_accel_osd         = torch.cat(total_accel_osd, dim=0)
        total_cp_osd            = torch.mean(torch.cat(total_cp_osd, dim=1), dim=1).squeeze()
        auc_cps_osd             = torch.sum(total_cp_osd.detach().cpu()*1e-2).item()
        
        # for i in range(total_mpjpe_osd.shape[0]):
        #     mpjpe_gain              = total_mpjpe_trace[i, ...].item() - total_mpjpe_osd[i,...].item()
        #     # mpjpe_gain              = total_mpjpe_g_trace[i, ...].item() - total_mpjpe_g_osd[i,...].item()
        #     # if mpjpe_gain >= 35:
        #         # print(i, mpjpe_gain, str(test_set[i][-1]))
        #     # if total_mpjpe_osd[i,...].item() < 58.7:
        #     print(i, round(total_mpjpe_trace[i, ...].item(),1),
        #             round(total_mpjpe_osd[i,...].item(),1),
        #             round(mpjpe_gain,1),
        #             str(test_set[i][-1]))
        
        # total_gp_osd            = torch.cat(total_gp_osd, dim=0)
        # print(np.array(total_gp_osd).mean())
        # print(np.array(total_fric_osd).mean()/2)
        # print(np.array(total_v_osd).mean()/26)
        # print(np.array(total_fk_osd).mean())
    
        print()
        print("  Methods | MPJPE | MPJPE-G | MPJPE-PA | PCK  | CPS | GRP   | Accel |")
        # print("  Methods | MPJPE | MPJPE-G | MPJPE-PA | PCK  | GRP   | Accel |")
        print(("  TRACE   | " 
            + str(round((total_mpjpe_trace.mean(0)).item(), 1)) + "  | " 
            + str(round((total_mpjpe_g_trace.mean(0)).item(), 1)) + "   | " 
            + str(round((total_mpjpe_pa_trace.mean(0)).item(),1)) + "     | "
            + str(round((total_pck_trace.mean(0)).item(),1))+ " | "
            + str(round((auc_cps_trace),1))+ " | "
            + str(round((total_grp_trace.mean(0)).item(),1))+ " | "
            + str(round((total_accel_trace.mean(0)).item(),1))+ "  | "))
        print(("  OSDNet  | " 
            + str(round((total_mpjpe_osd.mean(0)).item(), 1)) + "  | " 
            + str(round((total_mpjpe_g_osd.mean(0)).item(), 1)) + "   | " 
            + str(round((total_mpjpe_pa_osd.mean(0)).item(),1)) + "     | "
            + str(round((total_pck_osd.mean(0)).item(),1)) + " | "
            + str(round((auc_cps_osd),1)) + " | "
            + str(round((total_grp_osd.mean(0)).item(),1)) + "  | "
            + str(round((total_accel_osd.mean(0)).item(),1))+ "   | "))
        print()
    
        if args.use_wandb:
            wandb.log({
                "OSDNet_MPJPE": round((total_mpjpe_osd.mean(0)).item(), 1),
                "OSDNet_MPJPE_G": round((total_mpjpe_g_osd.mean(0)).item(), 1),
                "OSDNet_MPJPE_PA": round((total_mpjpe_pa_osd.mean(0)).item(),1),
                "OSDNet_PCK": round((total_pck_osd.mean(0)).item(),1),
                "OSDNet_GRP": round((total_grp_osd.mean(0)).item(),1),
                "OSDNet_accel": round((total_accel_osd.mean(0)).item(),1),
            })
            wandb.finish()
        
        result_qh_all       = torch.cat(result_qh_all, dim=0)
        result_q_all        = torch.cat(result_q_all, dim=0)
        
        result_ph_all       = torch.cat(result_ph_all, dim=0)
        result_p_all        = torch.cat(result_p_all, dim=0)
        result_pgt_all      = torch.cat(result_pgt_all, dim=0)
        
        if args.use_osd: 
            result_qsim_all     = torch.cat(result_qsim_all, dim=0)
            result_kalman_all = torch.cat(result_kalman_all, dim = 0)
            
        result_minv_all     = torch.cat(result_minv_all, dim=0)
        result_c_all        = torch.cat(result_c_all, dim=0)
        result_extf_all     = torch.cat(result_extf_all, dim=0)
        
        
        return [result_qh_all, result_q_all, result_qsim_all,
                result_ph_all, result_p_all, result_pgt_all,
                result_kalman_all, result_minv_all, result_c_all,
                result_extf_all]


def rendering_results_plt(args, results, sample_id, res=10, animation=True, interactive=False, show_filters=False):
    
    print('--- Loading test data. ---')
    test_set        = torch.load("datasets/" + args.dataset + "/" + experiment + "/test_set.pt")
    # data_path = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/h36m/processed/"
    data_path = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/fit3d/train/extracted/"
    # data_path       = "/cephyr/users/lecu/Alvis/TRACE_results/"
    if animation or interactive:
        fig             = plt.figure(figsize=(10,10))
        ax1             = fig.add_subplot(111, projection='3d', computed_zorder=False)
        ax1.view_init(elev=10., azim=-90, vertical_axis='z')
        # ax1.view_init(elev=5., azim=0, vertical_axis='z')
        if show_filters:
            # ax2_1           = fig.add_subplot(344)
            # ax2_2           = fig.add_subplot(348)
            # ax2_3           = fig.add_subplot(224)
            ax2_1           = fig.add_axes([0.6, 0.5, 0.1, 0.2], anchor='NW')
            ax2_2           = fig.add_axes([0.6, 0.35, 0.1, 0.2], anchor='NW')
            # ax2_3           = fig.add_axes([0.3, 0.35, 0.1, 0.2], anchor='NW')
        if interactive:
            plt.ion()
            plt.show()
        # parent_old      = [0,1,2,3,4, 0,6,7,8,9,   0,11,12,12, 13,14,15,16, 18,19,20,21, 23,24]
        # child_old       = [1,2,3,4,5, 6,7,8,9,10, 11,12,14,19, 14,15,16,17, 19,20,21,22, 24,25]
        # parent          = [0,1,2,3,4,5,  0,7,8,9,10,11,  0,13,14,14, 15,16,17,18, 20,21,22,23, 25,26]
        # child           = [1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,16,21, 16,17,18,19, 21,22,23,24, 26,27]
        # # parent      = [0,1,2,3,4, 0,6,7,8,9,   0,11,12,12, 13,14,15,16, 18,19,20,21, 23,24]
        # # child       = [1,2,3,4,5, 6,7,8,9,10, 11,12,14,19, 14,15,16,17, 19,20,21,22, 24,25]
        # par_gt          = [0,1,2,3,4, 0,6,7,8,9,   0,11,12,12, 13,14,15,16, 18,19,20,21, 23,24]
        # chi_gt          = [1,2,3,4,5, 6,7,8,9,10, 11,12,14,19, 14,15,16,17, 19,20,21,22, 24,25]
        
        parent_old      = [0,1,2,3,4, 0,6,7,8,9,   1,6, 13,14,15,16, 18,19,20,21, 23,24]
        child_old       = [1,2,3,4,5, 6,7,8,9,10, 14,19, 14,15,16,17, 19,20,21,22, 24,25]
        parent          = [0,1,2,3,4,5,  0,7,8,9,10,11, 1,7, 15,16,17,18, 20,21,22,23, 25,26]
        child           = [1,2,3,4,5,6, 7,8,9,10,11,12, 16,21, 16,17,18,19, 21,22,23,24, 26,27]
        par_gt          = [7,9,  6,8,   7,6,  13,15, 12,14, 13,7,]
        chi_gt          = [9,11, 8,10, 13,12, 15,17, 14,16, 12,6,]
        
        frames          = []
    
    
    result_qh_np, result_q_np, result_qsim_np = [], [], []
    result_ph_np, result_p_np, result_pgt_np, = [], [], []
    result_kalman_np, result_minv_np, result_c_np = [], [], []
    for s in range(len(sample_id)):
        
        scale               = 400
        result_qh_np.append(results[0][sample_id[s],...].detach().cpu())
        result_q_np.append(results[1][sample_id[s],...].detach().cpu())
        result_qsim_np.append(results[2][sample_id[s],...].detach().cpu())
        result_ph_np.append(results[3][sample_id[s],...].detach().cpu() * scale)
        result_p_np.append(results[4][sample_id[s],...].detach().cpu() * scale)
        result_pgt_np.append(results[5][sample_id[s],...].detach().cpu() * scale)
        result_kalman_np.append(results[6][sample_id[s],...].detach().cpu())
        result_minv_np.append(results[7][sample_id[s],...].detach().cpu())
        result_c_np.append(results[8][sample_id[s],...].detach().cpu())
        info                = test_set[sample_id[s]][-1]
        print("Test sample: " + str(info))
    
    result_qh_np        = torch.cat(result_qh_np)
    result_q_np         = torch.cat(result_q_np)
    result_qsim_np      = torch.cat(result_qsim_np)
    result_ph_np        = torch.cat(result_ph_np)
    result_p_np         = torch.cat(result_p_np)
    result_pgt_np       = torch.cat(result_pgt_np)
    result_kalman_np    = torch.cat(result_kalman_np)
    result_minv_np      = torch.cat(result_minv_np)
    result_c_np         = torch.cat(result_c_np)
    
    heel_l_n    = result_p_np[:,3,:] - result_p_np[:,2,:]
    heel_r_n    = result_p_np[:,8,:] - result_p_np[:,7,:]
    heel_l      = result_p_np[:,3,:] + heel_l_n / np.linalg.norm(heel_l_n, axis=-1, keepdims=True) * scale * 0.05
    heel_r      = result_p_np[:,8,:] + heel_r_n / np.linalg.norm(heel_r_n, axis=-1, keepdims=True) * scale * 0.05
    result_p_np     = np.insert(result_p_np, 3, heel_l, axis=1)
    result_p_np     = np.insert(result_p_np, 9, heel_r, axis=1)
    
    # past_colors     = ['gainsboro', 'lightgray', 'silver', 'darkgray', 'gray']
    
    # for f in range(result_ph_np.shape[0]):
    # result_ph_np[:,:,0] += 300
    # result_pgt_np[:,:,0] -= 300
    
    # result_ph_np[:,:,1] *= 0
    # result_p_np[:,:,1] *= 0
    # result_pgt_np[:,:,1] *= 0
    
    # start, stop = 49, 50
    # start, stop = 89, 90
    start,stop = 0, result_ph_np.shape[0]
    for f in range(start, stop, 10):
        
        # Start rendering
        ax1.cla()
        # ax1.title.set_text('Optimal-state Dynamics')
        ax1.grid(visible=False)
        ax1.set_xlim([-800,800])
        ax1.set_ylim([-800,800])
        ax1.set_zlim([ 10,1300])
        # ax1.set_xlim([-700,700])
        # ax1.set_ylim([-600,600])
        # ax1.set_zlim([ 10,1300])
        ax1.set_box_aspect([1.0, 1.0, 1.0])
        ax1.xaxis.set_pane_color((0.1, 0.1, 0.1, 0.0))
        ax1.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.0))
        ax1.zaxis.set_pane_color((0.3, 0.1, 0.1, 0.8))
        ax1.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        # ax1.set_xlabel('x')
        # ax1.set_ylabel('y')
        # ax1.set_zlabel('z')
        
        # rgb_img     = plt.imread(data_path + info[0] + "/" + info[1] + "/imageSequence/60457274/img_" + f"{(info[2]*2+2*f+1):06d}" + ".jpg")
        rgb_img     = plt.imread(data_path + info[1] + "_frames" + f"/{(info[2]*2+2*f+1):08d}" + ".jpg")
        # rgb_img     = plt.imread(data_path + "CAM3_rotated_video_5_frames/" + f"{(info[2]*2+2*f+1):08d}" + ".jpg")
        
        # print(rgb_img.shape)
        # img         = cv2.resize(rgb_img, (1600, 1000), interpolation = cv2.INTER_LINEAR)
        img         = cv2.resize(rgb_img, (800, 1000), interpolation = cv2.INTER_LINEAR)
        img         = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) / 255
        x, z        = np.ogrid[-img.shape[0]/2:img.shape[0]/2, -5:img.shape[1]-5]
        ax1.plot_surface(x, np.atleast_2d(img.shape[0]/2), z, rstride=res, cstride=res, facecolors=img, shade=False, zorder=0)
        # ax1.plot_surface(x, np.atleast_2d(600), z, rstride=res, cstride=res, facecolors=img, shade=False, zorder=0)
        
        # Contact states
        cL      = Circle(((result_p_np[f,3,0]+result_p_np[f,4,0])/2,
                          (result_p_np[f,3,1]+result_p_np[f,4,1])/2),
                         radius = 50, facecolor = 'tab:green',  edgecolor = 'dimgray', zorder = 1)
        cR      = Circle(((result_p_np[f,8,0]+result_p_np[f,9,0])/2,
                          (result_p_np[f,8,1]+result_p_np[f,9,1])/2),
                         radius = 50, facecolor = 'tab:orange', edgecolor = 'dimgray', zorder = 1)
        
        # if result_c_np[f,0] >= 0.5:
        #     ax1.add_patch(cL)
        #     art3d.pathpatch_2d_to_3d(cL, z=20, zdir="z")
        #     # ax1.text((result_p_np[f,3,0]+result_p_np[f,4,0])/2 -100, (result_p_np[f,3,1]+result_p_np[f,4,1])/2 -250, z=0.02, s='L', zorder = 2)
        # if result_c_np[f,1] >= 0.5:
        #     ax1.add_patch(cR)
        #     art3d.pathpatch_2d_to_3d(cR, z=20, zdir="z")
        #     # ax1.text((result_p_np[f,8,0]+result_p_np[f,9,0])/2 -100, (result_p_np[f,8,1]+result_p_np[f,9,1])/2 -250, z=0.02, s='R', zorder = 2)
        
        # ax1.plot(   [result_pgt_np[f,par_gt[0],0], result_pgt_np[f,chi_gt[0],0]],
        #             [result_pgt_np[f,par_gt[0],1], result_pgt_np[f,chi_gt[0],1]], 
        #             [result_pgt_np[f,par_gt[0],2], result_pgt_np[f,chi_gt[0],2]],
        #             linestyle='-',
        #             linewidth=1,
        #             color='tab:red', label='ground truth')
        # ax1.plot(   [result_ph_np[f,parent_old[0],0], result_ph_np[f,child_old[0],0]],
        #             [result_ph_np[f,parent_old[0],1], result_ph_np[f,child_old[0],1]], 
        #             [result_ph_np[f,parent_old[0],2], result_ph_np[f,child_old[0],2]],
        #             linestyle='-',
        #             linewidth=1,
        #             color='tab:purple', zorder=1, label='kinematics')
        # ax1.plot(   [result_p_np[f,parent[0],0], result_p_np[f,child[0],0]],
        #             [result_p_np[f,parent[0],1], result_p_np[f,child[0],1]], 
        #             [result_p_np[f,parent[0],2], result_p_np[f,child[0],2]],
        #             color = 'aqua', linestyle = '-', linewidth = 1, label='OSDCap')
        
        
        # Ground-truth
        ax1.scatter(result_pgt_np[f,:,0], result_pgt_np[f,:,1], result_pgt_np[f,:,2], s=25, c='red', marker='*', zorder=2)
        for k in range(len(par_gt)):
            ax1.plot(   [result_pgt_np[f,par_gt[k],0], result_pgt_np[f,chi_gt[k],0]],
                        [result_pgt_np[f,par_gt[k],1], result_pgt_np[f,chi_gt[k],1]], 
                        [result_pgt_np[f,par_gt[k],2], result_pgt_np[f,chi_gt[k],2]],
                        linestyle='solid',
                        linewidth=2,
                        color='tab:red', zorder=1)
        
        # TRACE Input
        # ax1.scatter(result_ph_np[f,:,0], result_ph_np[f,:,1], result_ph_np[f,:,2], s=25, c='k', marker='*', zorder=2)
        for k in range(len(parent_old)):
            ax1.plot(   [result_ph_np[f,parent_old[k],0], result_ph_np[f,child_old[k],0]],
                        [result_ph_np[f,parent_old[k],1], result_ph_np[f,child_old[k],1]], 
                        [result_ph_np[f,parent_old[k],2], result_ph_np[f,child_old[k],2]],
                        linestyle='solid',
                        linewidth=2,
                        color='grey',
                        zorder=1)
        # for s in range(0, f, 10):
        #     for k in range(len(parent_old)):
        #         ax1.plot(   [result_ph_np[s,parent_old[k],0], result_ph_np[s,child_old[k],0]],
        #                     [result_ph_np[s,parent_old[k],1], result_ph_np[s,child_old[k],1]], 
        #                     [result_ph_np[s,parent_old[k],2], result_ph_np[s,child_old[k],2]],
        #                     linestyle='dashed',
        #                     linewidth=2,
        #                     alpha=f*0.003,
        #                     color='tab:purple', zorder=1)
        
        # mid = (result_p_np[f,3,:]+result_p_np[f,4,:])/2
        # ax1.scatter(mid[0], mid[1], mid[2], s=500)
        # Optimal-state
        for k in range(len(parent)):
        #     # dist_p_to_cam     = np.linalg.norm(result_p_np[f,parent[k],:] - np.array([1000, 1000, 1000])) / scale
        #     # dist_c_to_cam     = np.linalg.norm(result_p_np[f,child[k],:] - np.array([1000, 1000, 1000])) / scale
        #     # dist_p_to_cam     = np.linalg.norm(result_p_np[f,parent[k],:] - np.array([-1000, 1000, 1000])) / scale
        #     # dist_c_to_cam     = np.linalg.norm(result_p_np[f,child[k],:] - np.array([-1000, 1000, 1000])) / scale
            dist_p_to_cam     = np.linalg.norm(result_p_np[f,parent[k],:] - np.array([0, 600, 600])) / scale
            dist_c_to_cam     = np.linalg.norm(result_p_np[f,child[k],:] - np.array([0, 600, 600])) / scale
            
        #     # for i in range(1, 5):
        #     #     if f - i < 0: continue
        #     #     ax1.scatter(result_p_np[f-i,parent[k],0], result_p_np[f-i,parent[k],1], result_p_np[f-i,parent[k],2], 
        #     #                 s=80, color=past_colors[i], marker='.', zorder = dist_p_to_cam-0.1*i)
        #     #     ax1.scatter(result_p_np[f-i,child[k],0],  result_p_np[f-i,child[k],1],  result_p_np[f-i,child[k],2], 
        #     #                 s=80, color=past_colors[i], marker='.', zorder = dist_c_to_cam-0.5*i)
            
            
        #     ax1.scatter(result_p_np[f,parent[k],0], result_p_np[f,parent[k],1], result_p_np[f,parent[k],2], 
        #                 s=160, color='mediumblue', marker='.', edgecolors='aqua', zorder = dist_p_to_cam+0.2)
        #     ax1.scatter(result_p_np[f,child[k],0],  result_p_np[f,child[k],1],  result_p_np[f,child[k],2], 
        #                 s=160, color='mediumblue', marker='.', edgecolors='aqua', zorder = dist_c_to_cam+0.1)
            
            ax1.plot(   [result_p_np[f,parent[k],0], result_p_np[f,child[k],0]],
                        [result_p_np[f,parent[k],1], result_p_np[f,child[k],1]], 
                        [result_p_np[f,parent[k],2], result_p_np[f,child[k],2]],
                        color = 'aqua', linestyle = '-', linewidth = 4, zorder = dist_p_to_cam,
                        path_effects=[pe.Stroke(linewidth=5, foreground='darkcyan'), pe.Normal()])
            
        # # Error lines between prediction and GT
        # result_pgt_np_n     = result_pgt_np.clone()[f,j17_ids_gt,:]
        # ax1.scatter(result_pgt_np_n[:,0], result_pgt_np_n[:,1], result_pgt_np_n[:,2], s=25, c='tab:blue', marker='*', zorder=5)
        # result_p_np_n       = result_p_np.clone()[f,j17_ids_pr,:]
        # ax1.scatter(result_p_np_n[:,0], result_p_np_n[:,1], result_p_np_n[:,2], s=25, c='tab:orange', marker='*', zorder=5)
        # for j in range(result_pgt_np_n.shape[0]):
        #     ax1.plot(   [result_p_np_n[j,0], result_pgt_np_n[j,0]],
        #                 [result_p_np_n[j,1], result_pgt_np_n[j,1]], 
        #                 [result_p_np_n[j,2], result_pgt_np_n[j,2]],
        #                 linewidth=1,
        #                 color='tab:gray',
        #                 zorder=2)
        
        # ax1.quiver(800, 0, 10, -1, 0, 0, length=200, color='white')
        # ax1.text(800, 400, 10, "optical axis", color='white', fontsize='small')
        
        # ax1.quiver(0, -600, 10, 0, 1, 0, length=200, color='white')
        # ax1.text(300, -600, 0, "optical axis", color='white', fontsize='small')
        
        # ax1.plot([result_ph_np[f,0,0], result_ph_np[f,0,0]],
        #          [result_ph_np[f,0,1], result_ph_np[f,0,1]],
        #          [0, 750], color = 'green',linewidth = 1, linestyle = 'dashed')
        # ax1.plot([result_p_np[f,0,0], result_p_np[f,0,0]],
        #          [result_p_np[f,0,1], result_p_np[f,0,1]],
        #          [0, 750], color = 'green',linewidth = 1, linestyle = 'dashed')
        # ax1.plot([result_pgt_np[f,0,0], result_pgt_np[f,0,0]],
        #          [result_pgt_np[f,0,1], result_pgt_np[f,0,1]],
        #          [0, 750], color = 'green',linewidth = 1, linestyle = 'dashed')
        
        # ax1.legend(loc='center', bbox_to_anchor=(0.0, 0.40, 0.5, 0.5), fontsize='small')
        
        if show_filters:
            # Kalman filter 
            ax2_1.cla()
            ax2_1.title.set_text('Kalman gains')
            ax2_1.set_xticks([])
            ax2_1.set_yticks([])
            im1     = ax2_1.imshow(result_kalman_np[f,...], cmap='inferno', vmin=result_kalman_np.min(), vmax=result_kalman_np.max())
            bar1    = fig.colorbar(im1, ax=ax2_1, fraction=0.046, pad=0.04)
            
            # Inverse Joint-space Inertia Matrix
            ax2_2.cla()
            ax2_2.title.set_text('Inverse JSIM')
            ax2_2.set_xticks([])
            ax2_2.set_yticks([])
            im2     = ax2_2.imshow(result_minv_np[f,...], cmap='summer', vmin=result_minv_np.min(), vmax=result_minv_np.max())
            bar2    = fig.colorbar(im2, ax=ax2_2, fraction=0.046, pad=0.04)
            
            # # Kalman bias diagonal
            # ax2_3.cla()
            # # ax2_3.set_xticks([])
            # kalman_diag = np.diag(result_kalman_np[f,...])[0:6]
            # colors = []
            # cmap = matplotlib.colormaps.get_cmap('inferno')
            # for decimal in kalman_diag:
            #     colors.append(cmap(decimal))
            # im3     = ax2_3.barh(range(kalman_diag.shape[0]), kalman_diag, color=colors)
            # # ax2_3.set_xlim(0., 0.8)
            # LABELS = ["x", "y", "z", r"$\theta_x$", r"$\theta_y$", r"$\theta_z$"]
            # ax2_3.set_yticks([0,1,2,3,4,5], LABELS)
            # ax2_3.invert_yaxis()
            # ax2_3.set_xlabel('K gains')
            # ax2_3.set_ylabel('dof')
            # ax2_3.xaxis.set_label_position("top")
        
        # Draw and save frames
        # fig.canvas.draw()
        # fig.tight_layout()
        # plt.savefig(f'tmp_imgs/frame_{f}.png', transparent=True, dpi=300, format='png',
        #             bbox_inches=matplotlib.transforms.Bbox(np.array([[1.5,2],[9,7]])))
        plt.savefig(f'tmp_imgs/frame_{f}.png', transparent=True, dpi=100, format='png', facecolor='white',
                    bbox_inches='tight')
        # image = imageio.v2.imread(f'./tmp_imgs/frame_{f}.png', format='PNG')
        image   = Image.open(f'./tmp_imgs/frame_{f}.png')
        frames.append(image)
        # fig.canvas.flush_events()
        if show_filters:
            bar1.remove()
            bar2.remove()
    
    if animation:
        name = "animations/test_sample_" + str(sample_id) + ".gif"
        # imageio.mimsave(name, frames, format='GIF', fps=15)
        frames[0].save("out1.gif", save_all=True, append_images=frames[1:], duration=60, disposal=2, loop=0)
        plt.close()


def force_plots(args, results, sample_id):
    print('--- Loading test data. ---')
    test_set        = torch.load("datasets/" + args.dataset + "/" + experiment + "/test_set.pt")
    # print(len(results[9]))
    # result_extf_np.append(results[9].detach().cpu())
    
    result_extf_np  = []
    for s in range(len(sample_id)):
    #     scale               = 400
    #     # print(results[9].shape)
    #     # result_qh_np.append(results[0][sample_id[s],...].detach().cpu())
    #     # result_q_np.append(results[1][sample_id[s],...].detach().cpu())
    #     # result_qsim_np.append(results[2][sample_id[s],...].detach().cpu())
    #     # result_ph_np.append(results[3][sample_id[s],...].detach().cpu() * scale)
    #     # result_p_np.append(results[4][sample_id[s],...].detach().cpu() * scale)
    #     # result_pgt_np.append(results[5][sample_id[s],...].detach().cpu() * scale)
    #     # result_kalman_np.append(results[6][sample_id[s],...].detach().cpu())
    #     # result_minv_np.append(results[7][sample_id[s],...].detach().cpu())
    #     # result_c_np.append(results[8][sample_id[s],...].detach().cpu())
        result_extf_np.append(results[9][sample_id[s],...].detach().cpu())
        info                = test_set[sample_id[s]][-1]
        print("Test sample: " + str(info))
    result_extf_np         = torch.cat(result_extf_np)
    print(result_extf_np.shape)
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(result_extf_np[50:,0,0])
    # ax1.plot(result_extf_np[50:,1,0])
    # plt.savefig('forces/x.png')
    
    # fig = plt.figure()
    # ax2 = fig.add_subplot(111)
    # ax2.plot(result_extf_np[50:,0,1])
    # ax2.plot(result_extf_np[50:,1,1])
    # plt.savefig('forces/y.png')
    
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.set_ylim([0, 10.0])
    ax3.set_ylabel('Nm/kg')
    ax3.set_xlabel('ms')
    
    ax3.plot(result_extf_np[60:,0,2], label='Left', linewidth=2)
    ax3.plot(result_extf_np[60:,1,2], label='Right', linewidth=2)
    ax3.legend(loc='best')
    ax3.set_title('Vertical-axis reaction forces')
    
    plt.savefig('forces/z.png', dpi=400, bbox_inches='tight')
    
    plt.close()
    
    
if __name__ == "__main__":
    
    parser          = get_parse()
    args            = parser.parse_args()
    torch.manual_seed(args.seed)
    
    if args.ablation:   experiment = "ablation"
    else:               experiment = "full"
    osdnet          = OSDNet(in_dim         = 349,
                             gru_dim        = 128,
                             hid_dim        = 512,
                             nlayers        = 1,
                             out_dim        = 128,
                             pd_temp        = 0.5,
                             c_temp         = 0.1,
                             use_osd        = args.use_osd).to(device)
    print("OSDNet num. params: ", sum(p.numel() for p in osdnet.parameters()))

    if args.use_wandb:
        config      = { "Kp_trs": 20.0, "Kp_rot": 10.0, "Kp_jnt": 1.3,
                        "Kd_trs": 1.0, "Kd_rot": 0.07, "Kd_jnt": 0.03}
        if args.ablation:
            wandb.init(project = "hmdcap", name = "Ablation, variable: " + str(args.abl_var), config = config)
        else:
            if args.dataset == 'h36m':
                wandb.init(project = "hmdcap", name = "Full training Human3.6m" + "_seed_" + str(args.seed), config = config)
            elif args.dataset == 'fit3d':
                wandb.init(project = "hmdcap", name = "Full training Fit3d" + "_seed_" + str(args.seed), config = config)
    
    if args.use_osd:
        print("Using Optimal-state Dynamics Estimation")
        print("Ablation variable: " + str(args.abl_var))
        if args.ablation:
            model_path      = "trained_models/" + args.dataset + "/" + experiment + "/OSDNet_var_" + str(args.abl_var)
        else:
            model_path      = "trained_models/" + args.dataset + "/" + experiment + "/OSDNet"
    else:
        print("Using Normal PD")
        model_path      = "trained_models/" + args.dataset + "/" + experiment + "/PDNet_only2"
    
    if args.train_models:
        Processor_OSDNet(args, osdnet, model_path, mode='train')
    results = Processor_OSDNet(args, osdnet, model_path, mode='test')
    torch.save(results, "saved_results/" + args.dataset + "/" + args.dataset + "_results.pth")
    
    # shutil.rmtree('tmp_imgs')
    # os.mkdir('tmp_imgs')
    # results = torch.load("saved_results/" + args.dataset + "/" + args.dataset + "_results.pth")
    # # # # samples         = [129, 130, 208, 324, 327, 330, 335, 336, 337, 340, 341, 342, 388, 390, 394]
    # # samples  = [110]
    # # # for s in range(len(samples)):
    # # # samples  = [324]
    # samples     = [97]
    # rendering_results_plt(args, results, samples, res=10,
    #                     animation=True, interactive=False, show_filters=False)
    # # # force_plots(args, results, samples)
