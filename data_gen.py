### Cuong Le - CVL, Linköping University, Sweden
### Collecting data from H36M, Fit3D, and SportPose dataset

# Import nessesary libraries
import sys, os
sys.path.append(os.getcwd() + "/rbdl/build/python")
import rbdl
import argparse, h5py, json, pickle
import numpy as np
import torch
from common.utils import *
from pytorch3d import transforms

# base_path           = "/proj/cvl/datasets"
base_path           = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets"

def get_parse():
    parser = argparse.ArgumentParser(description='Database generation for OSDCap experiments') 
    parser.add_argument("-pth", "--base-path", type=str,
                        default="/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets", 
                        help="Absolute path to your dataset folder") # "/proj/cvl/datasets"
    parser.add_argument("-dst", "--database", type=str, default="h36m",
                        help="Specify the dataset to be used")
    parser.add_argument("-abl", "--ablation", action="store_true", 
                        help="Generate smaller samples for the ablation study") 
    parser.add_argument("-ltr", "--train-len", type=int, default=103, 
                        help="The samples' length to be extracted (training)")
    parser.add_argument("-lte", "--test-len", type=int, default=103, 
                        help="The samples' length to be extracted (testing)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser

def data_mapping_smpl(root_trans, root_rot, joint_angle, length, qsize):
    # Root translation and joint angular mapping
    q_joint         = torch.zeros((length, 20, 3))
    simu_id         = [0,1,2,3, 4,5,6, 7, 8,9, 10,11,12,13, 14,15,16,17, 18,19]
    smpl_id         = [0,3,6,9, 1,4,7,10, 2,8, 12,15,17,19, 13,16,18,20, 11,14]
    for j in range(len(simu_id)):
        q_joint[:,simu_id[j],:] = joint_angle[:,smpl_id[j],:]
    return torch.cat((root_trans, root_rot[:,1:], q_joint.reshape(length,-1), root_rot[:,0:1]), dim=1)

def data_mapping_h36m(root_trans, root_rot, joint_angle, length, qsize):
    # Root translation and joint angular mapping
    q_joint         = torch.zeros((length, 20, 3))
    simu_id         = [0,1,2,3, 4,5,6,7, 8,9, 10,11,12,13, 14,15,16,17, 18,19]
    h36m_id         = [4,5,6,7, 0,1,2,3, 8,9, 12,13,14,15, 18,19,20,21, 10,11]
    for j in range(len(simu_id)):
        q_joint[:,simu_id[j],:] = joint_angle[:,h36m_id[j],:]
    return torch.cat((root_trans, root_rot[:,1:], q_joint.reshape(length,-1), root_rot[:,0:1]), dim=1)

def get_contact_labels(sample, sample_ang, idx=[8,9,3,4]):
    lhw0, ltw0 ,rhw0, rtw0 = sample[0,idx[0],:], sample[0,idx[1],:], sample[0,idx[2],:], sample[0,idx[3],:]
    contacts        = torch.zeros(sample.shape[0], 2)
    quat            = get_rot_quat(sample_ang)
    for f in range(sample.shape[0]):
        base            = sample[f,0,:]
        lhw, ltw        = sample[f,idx[0],:], sample[f,idx[1],:]
        rhw, rtw        = sample[f,idx[2],:], sample[f,idx[3],:] 
        
        Lhmove          = (lhw - lhw0).pow(2).sum().sqrt()
        Ltmove          = (ltw - ltw0).pow(2).sum().sqrt()
        Rhmove          = (rhw - rhw0).pow(2).sum().sqrt()
        Rtmove          = (rtw - rtw0).pow(2).sum().sqrt()

        if (lhw[2] < 0.16) or (ltw[2] < 0.12):
            if (Lhmove < 0.022) or (Ltmove < 0.022):
                contacts[f,0] = 1.0
        if (rhw[2] < 0.16) or (rtw[2] < 0.12):
            if (Rhmove < 0.022) or (Rtmove < 0.022):
                contacts[f,1] = 1.0
        
        lhw0, ltw0            = lhw, ltw
        rhw0, rtw0            = rhw, rtw
    return contacts
    
if __name__ == "__main__":
    
    # Get arguments
    args                = get_parse().parse_args() 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Humanoid model
    humanoid_path       = 'common/smpl_human.urdf'
    humanoid            = rbdl.loadModel(humanoid_path.encode(), floating_base=True)
    qd_size             = humanoid.qdot_size
    
    if args.ablation: experiment = "ablation"
    else: experiment = "full"
    
    if args.database == "h36m":
        # Camera parameters
        raw_cam_params      = np.load('datasets/h36m/camera_h36m.npy')
        subject_names       = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
        train_subs, test_subs = ["S1", "S5", "S6", "S7", "S8"], ["S9", "S11"]
        view_names          = ["54138969", "55011271", "58860488", "60457274"] # 0, 1, 2, 3
        if args.ablation:
            action_names        = [ "Directions-1", "Discussion-1", "Greeting-1", "Posing-1", "TakingPhoto-1", 
                                    "Purchases-1", "Waiting-1", "Walking-1", "WalkingDog-1", "WalkingTogether-1"]
        else:
            action_names        = [ "Directions-1",     "Directions-2",     "Discussion-1",     "Discussion-2",
                                    "Greeting-1",       "Greeting-2",       "Posing-1",         "Posing-2",
                                    "TakingPhoto-1",    "TakingPhoto-2",    "Purchases-1",      "Purchases-2",
                                    "Waiting-1",        "Waiting-2",        "Walking-1",        "Walking-2",
                                    "WalkingDog-1",     "WalkingDog-2",     "WalkingTogether-1","WalkingTogether-2"]
            
            
    if args.database == "fit3d":
        subject_names       = ["s03", "s04", "s05", "s07", "s08", "s10", "s09", "s11"]
        train_subs, test_subs = ["s03", "s04", "s05", "s07", "s08", "s10"], ["s09", "s11"]
        view_names          = ["60457274"]
        action_names        = [ "band_pull_apart", "barbell_dead_row", "barbell_row", "barbell_shrug",
                                "clean_and_press", "deadlift", "drag_curl", "dumbbell_biceps_curls", 
                                "dumbbell_curl_trifecta" ,"dumbbell_hammer_curls", "dumbbell_high_pulls", 
                                "dumbbell_overhead_shoulder_press", "dumbbell_reverse_lunge", "dumbbell_scaptions", 
                                "neutral_overhead_shoulder_press", "one_arm_row", "overhead_extension_thruster", 
                                "overhead_trap_raises", "side_lateral_raise", "squat", "standing_ab_twists", 
                                "w_raise", "walk_the_box", "warmup_2", "warmup_3", "warmup_4", "warmup_5",
                                "warmup_6", "warmup_7", "warmup_8", "warmup_9", "warmup_10", "warmup_11", 
                                "warmup_12", "warmup_13", "warmup_14", "warmup_15", "warmup_16", "warmup_17", "warmup_18", "warmup_19"]
        
    if args.database == "aist":
        subject_names       =  ["None"]
        action_names        =  ["gBR_sBM_c06_d06_mBR4_ch06",
                                "gBR_sBM_c07_d06_mBR4_ch02",
                                "gBR_sBM_c08_d05_mBR1_ch01",
                                "gBR_sFM_c03_d04_mBR0_ch01",
                                "gJB_sBM_c02_d09_mJB3_ch10",
                                "gKR_sBM_c09_d30_mKR5_ch05",
                                "gLH_sBM_c04_d18_mLH5_ch07",
                                "gLH_sBM_c07_d18_mLH4_ch03",
                                "gLH_sBM_c09_d17_mLH1_ch02",
                                "gLH_sFM_c03_d18_mLH0_ch15",
                                "gLO_sBM_c05_d14_mLO4_ch07",
                                "gLO_sBM_c07_d15_mLO4_ch09",
                                "gLO_sFM_c02_d15_mLO4_ch21",
                                "gMH_sBM_c01_d24_mMH3_ch02",
                                "gMH_sBM_c05_d24_mMH4_ch07"]
    
        label_names         =  ["gBR_sBM_cAll_d06_mBR4_ch06", # setting 1
                                "gBR_sBM_cAll_d06_mBR4_ch02", # setting 1
                                "gBR_sBM_cAll_d05_mBR1_ch01", # setting 1
                                "gBR_sFM_cAll_d04_mBR0_ch01", # setting 1
                                "gJB_sBM_cAll_d09_mJB3_ch10", # setting 3
                                "gKR_sBM_cAll_d30_mKR5_ch05", # setting 5
                                "gLH_sBM_cAll_d18_mLH5_ch07", # setting 6
                                "gLH_sBM_cAll_d18_mLH4_ch03", # setting 6
                                "gLH_sBM_cAll_d17_mLH1_ch02", # setting 6
                                "gLH_sFM_cAll_d18_mLH0_ch15", # setting 6
                                "gLO_sBM_cAll_d14_mLO4_ch07", # setting 7_1
                                "gLO_sBM_cAll_d15_mLO4_ch09", # setting 7_1
                                "gLO_sFM_cAll_d15_mLO4_ch21", # setting 7_2
                                "gMH_sBM_cAll_d24_mMH3_ch02", # setting 8_1
                                "gMH_sBM_cAll_d24_mMH4_ch07"] # setting 8_1
        
        smpl_names          =  ["gBR_sBM_cAll_d06_mBR4_ch06", # setting 1
                                "gBR_sBM_cAll_d06_mBR4_ch02", # setting 1
                                "gBR_sBM_cAll_d05_mBR1_ch01", # setting 1
                                "gBR_sFM_cAll_d04_mBR0_ch01", # setting 1
                                "gJB_sBM_cAll_d09_mJB3_ch10", # setting 3
                                "gKR_sBM_cAll_d30_mKR5_ch05", # setting 5
                                "gLH_sBM_cAll_d18_mLH5_ch07", # setting 6
                                "gLH_sBM_cAll_d18_mLH4_ch03", # setting 6
                                "gLH_sBM_cAll_d17_mLH1_ch02", # setting 6
                                "gLH_sFM_cAll_d18_mLH0_ch15", # setting 6
                                "gLO_sBM_cAll_d14_mLO4_ch07", # setting 7_1
                                "gLO_sBM_cAll_d15_mLO4_ch09", # setting 7_1
                                "gLO_sFM_cAll_d15_mLO4_ch21", # setting 7_2
                                "gMH_sBM_cAll_d24_mMH3_ch02", # setting 8_1
                                "gMH_sBM_cAll_d24_mMH4_ch07"] # setting 8_1
        
        setting_names       =  ["setting1", "setting1", "setting1", "setting1",
                                "setting3", "setting5",
                                "setting6", "setting6", "setting6", "setting6",
                                "setting7_1", "setting7_1", "setting7_2",
                                "setting7_2", "setting8_1", "setting8_1"]
        
        camera_ids          =  [6, 7, 8, 3, 2, 9, 4, 7, 9, 3, 5, 7, 2, 1, 5]
    
    if args.database == "sport":
        subject_names       = ["S02", "S03", "S05", "S06", "S07", "S08", "S09", "S12", "S13", "S14"]
        train_subs, test_subs = ["S02", "S03", "S05", "S06", "S07", "S08", "S09"], ["S12", "S13", "S14"]
        action_names        = ["jump0000", "jump0001", "jump0002", "jump0003", "jump0004",
                               "throw_baseball0005", "throw_baseball0006", "throw_baseball0007", "throw_baseball0008", "throw_baseball0009",
                               "soccer0010", "soccer0011", "soccer0012", "soccer0013", "soccer0014",
                               "volley0015", "volley0016", "volley0017", "volley0018", "volley0019",
                               "tennis0020", "tennis0021", "tennis0022", "tennis0023", "tennis0024"]
        
    train_set, test_set = [], []
    
    for subject_id in range(len(subject_names)):
        for action_id in range(len(action_names)):
            
            if args.database == "h36m":
                if subject_names[subject_id] in train_subs:
                    if args.ablation:   cam_list    = [3]
                    else:               cam_list    = [0, 1, 2, 3]
                if subject_names[subject_id] in test_subs:
                    cam_list    = [3]
                num_view    = len(cam_list)
            if args.database == "fit3d":    num_view = 1
            if args.database == "aist":     num_view = 1
            if args.database == "sport":    num_view = 1
            
            for idx in range(num_view):
                
                ########## Load the pose estimation results from TRACE ##########
                if args.database == "h36m":
                    view_id     = cam_list[idx]
                    data_path   = (base_path + "/h36m/TRACE_results/" 
                                   + subject_names[subject_id] + "/" 
                                   + action_names[action_id] + "/" 
                                   + view_names[view_id] + ".npz")
                    print(("Processing H36m "   + subject_names[subject_id] 
                                                + ", Camera: "   + view_names[view_id] 
                                                + ", Action: "   + action_names[action_id]))
                    
                if args.database == "fit3d": 
                    view_id     = 0
                    data_path   = (base_path + "/fit3d/TRACE_results/" 
                                   + subject_names[subject_id] + "/" 
                                   + view_names[0] + "/" 
                                   + action_names[action_id] + "/" 
                                   + action_names[action_id] +".mp4.npz")
                    print(("Processing Fit3d "  + subject_names[subject_id] 
                                                + ", Camera: "   + view_names[view_id] 
                                                + ", Action: "   + action_names[action_id]))
                    
                if args.database == "aist":
                    data_path   = (base_path + "/aist/TRACE_results/"
                                   + action_names[action_id] + "/"
                                   + action_names[action_id] + ".mp4.npz")
                    print(("Processing AISTpp" + ", Action: "   + action_names[action_id]))
                
                if args.database == "sport":
                    data_path   = (base_path + "/sport/TRACE_results/"
                                   + "indoors/"
                                   + subject_names[subject_id] + "/"
                                   + action_names[action_id] + "/" 
                                   + "CAM3_rotated_video_{:d}.avi.npz").format(action_id)
                    print(("Processing SportsPose "  + subject_names[subject_id] 
                                                + ", Action: "   + action_names[action_id]))
                
                data             = np.load(data_path, allow_pickle=True)['outputs'][()]
                root_trans_cc    = torch.from_numpy(np.array(data.get('cam_trans')))       # in Meters
                theta_in_cc      = torch.from_numpy(np.array(data.get('smpl_thetas')))     # in Axis-angle format
                theta_in_cc      = theta_in_cc.reshape(-1, 24, 3)
                seq_len          = theta_in_cc.shape[0]
                
                ########## Ground truth data from Human 3.6m ##########
                if args.database == "h36m":
                    
                    # GT from dataset
                    annot_path      = ("datasets/h36m/processed/" 
                                        + subject_names[subject_id] + "/" 
                                        + action_names[action_id] + "/annot.h5")
                    annot           = h5py.File(annot_path, 'r')
                    j3d_gt          = torch.from_numpy(np.array(annot.get('pose')['3d-univ'])[seq_len*view_id:seq_len*view_id+seq_len,...] * 1e-3)
                    theta_gt_cc     = torch.from_numpy(np.array(annot.get('angles'))[seq_len*view_id:seq_len*view_id+seq_len,...]).float()
                    
                    # Process the GT data (only for H36m)
                    theta_gt_cc[:,14,0]          += 90
                    theta_gt_cc[:,20,0]          -= 90
                    theta_gt_cc[:,14:18,[1, 2]]  = theta_gt_cc[:,14:18,[2, 1]]
                    theta_gt_cc[:,20:24,[1, 2]]  = theta_gt_cc[:,20:24,[2, 1]]
                    theta_gt_cc[:,14:18,2]       = -theta_gt_cc[:,14:18,2]
                    theta_gt_cc[:,20:24,1]       = -theta_gt_cc[:,20:24,1]
                    
                    # Camera parameters depending on views and subjects
                    param               = torch.Tensor(raw_cam_params[subject_id][view_id])
                    cam_R               = transforms.euler_angles_to_matrix(param[:3], "XYZ")
                    cam_T               = param[3:6] * 1e-3
                    cam_G               = torch.eye(4)
                    cam_G[:3,:3]        = cam_R
                    cam_G[:3,3]         = cam_T
                    
                    # Transform TRACE inputs from camera space to world space
                    theta_root_cc       = theta_in_cc[:, 0, :]
                    root_rot_mat_cc     = transforms.axis_angle_to_matrix(theta_root_cc)
                    
                    theta_gt_root_cc    = torch.deg2rad(theta_gt_cc[:, 1, :])
                    root_gt_rot_mat_cc  = transforms.euler_angles_to_matrix(theta_gt_root_cc, "ZXY")
                    
                    # Pre-allocate the variables
                    root_trans_wc       = torch.zeros(seq_len, 3)
                    root_quat_wc        = torch.zeros(seq_len, 4)
                    root_gt_trans_wc    = theta_gt_cc[:, 0, :] * 1e-3
                    root_gt_quat_wc     = torch.zeros(seq_len, 4)
                    j3d_gt_wc_raw       = torch.zeros_like(j3d_gt)          # ground-truth 3d joints
                    
                    for f in range(seq_len):
                        root_trans_homog        = torch.Tensor([0,0,0,1])
                        root_trans_homog[:3]    = root_trans_cc[f,:]
                        root_trans_wc[f, :]     = (torch.linalg.inv(cam_G) @ root_trans_homog)[:3]
                        qr_homog                = torch.eye(4)
                        qr_homog[:3, :3]        = root_rot_mat_cc[f, :]
                        root_quat_wc[f, :]      = transforms.matrix_to_quaternion((torch.linalg.inv(cam_G) @ qr_homog)[:3, :3])
                        root_gt_quat_wc[f, :]   = transforms.matrix_to_quaternion(root_gt_rot_mat_cc[f, :])
                        j3d_gt_wc_raw[f,...]    = (cam_R.T @ j3d_gt[f,...].float().T - cam_R.T @ cam_T[:, None]).transpose(1,0)
                    
                    theta_joints    = theta_in_cc[:, 1:, :]
                    joints_rot_mat  = transforms.axis_angle_to_matrix(theta_joints.reshape(-1, 3))
                    joints_rot      = transforms.matrix_to_euler_angles(joints_rot_mat, "ZXY").reshape(-1, 23, 3)
                    
                    joints_gt_rot   = torch.deg2rad(theta_gt_cc[:, 2:, :])
                    joints_gt_rot[abs(joints_gt_rot)>2*torch.pi] %= (2*torch.pi)
                    
                    # Mapping joint configuration from H36m to the simulated character
                    gt_state_raw    = data_mapping_h36m(root_gt_trans_wc, root_gt_quat_wc, joints_gt_rot, seq_len, qd_size)

                ########## Ground truth data from Fit3D ##########
                if args.database == "fit3d":

                    j3d_path            = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/fit3d/train/" 
                                           + subject_names[subject_id] + "/joints3d_25/" 
                                           + action_names[action_id] + ".json")
                    ang_path            = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/fit3d/train/" 
                                           + subject_names[subject_id] + "/smplx/" 
                                           + action_names[action_id] + ".json")
                    param_path          = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/fit3d/train/" 
                                           + subject_names[subject_id] + "/camera_parameters/60457274/" 
                                           + action_names[action_id] + ".json")
                    
                    with open(j3d_path) as json_file:
                        j3d_data = json.load(json_file)
                    with open(ang_path) as json_file:
                        ang_data = json.load(json_file)
                    with open(param_path) as json_file:
                        cam_param = json.load(json_file)

                    j3d_gt              = torch.from_numpy(np.array(j3d_data['joints3d_25'])).float()
                    theta_gt_cc_trs     = torch.from_numpy(np.array(ang_data['transl']))
                    theta_gt_cc_rot     = torch.from_numpy(np.array(ang_data['global_orient']))
                    theta_gt_cc_jnt     = torch.from_numpy(np.array(ang_data['body_pose']))
                    cam_ex              = np.array(cam_param['extrinsics'])
                    seq_len             = j3d_gt.shape[0]
                    
                    # Camera parameters depending on views and subjects
                    cam_R               = torch.from_numpy(np.array(cam_ex.item()['R'])).float()
                    cam_T               = torch.from_numpy(np.array(cam_ex.item()['T'])).float()
                    cam_G               = torch.eye(4)                        # Transformation matrix
                    cam_G[:3,:3]        = cam_R
                    cam_G[0,3]          = cam_T[0,0]
                    cam_G[1,3]          = cam_T[0,2]
                    cam_G[2,3]          = cam_T[0,1]
                    
                    # Transform TRACE inputs from camera space to world space
                    root_trans_cc       = root_trans_cc[:seq_len, :]
                    theta_root_cc       = theta_in_cc[:seq_len, 0, :]
                    root_rot_mat_cc     = transforms.axis_angle_to_matrix(theta_root_cc)
                    
                    # Pre-allocate the variables
                    root_trans_wc       = torch.zeros(seq_len, 3)
                    root_quat_wc        = torch.zeros(seq_len, 4)
                    j3d_gt_wc_raw       = torch.zeros_like(j3d_gt)          # ground-truth 3d joints
                    
                    for f in range(seq_len):
                        root_trans_homog        = torch.Tensor([0,0,0,1])
                        root_trans_homog[:3]    = root_trans_cc[f,:]
                        root_trans_wc[f, :]     = (torch.linalg.inv(cam_G) @ root_trans_homog)[:3]
                        qr_homog                = torch.eye(4)
                        qr_homog[:3, :3]        = root_rot_mat_cc[f, :]
                        root_quat_wc[f, :]      = transforms.matrix_to_quaternion((torch.linalg.inv(cam_G) @ qr_homog)[:3, :3])
                    
                    theta_joints        = theta_in_cc[:seq_len, 1:, :]
                    joints_rot_mat      = transforms.axis_angle_to_matrix(theta_joints.reshape(-1, 3))
                    joints_rot          = transforms.matrix_to_euler_angles(joints_rot_mat, "ZXY").reshape(-1, 23, 3)
                    
                    # Converting the root Euler's representation to world coordinate
                    root_gt_trans_wc    = theta_gt_cc_trs
                    root_gt_quat_wc     = transforms.matrix_to_quaternion(theta_gt_cc_rot).squeeze(1)
                    joints_gt_rot       = transforms.matrix_to_euler_angles(theta_gt_cc_jnt, "ZXY")
                    j3d_gt_wc_raw       = j3d_gt
                    
                    # Mapping joint configuration from Fit3D to the simulated character
                    gt_state_raw        = data_mapping_smpl(root_gt_trans_wc, root_gt_quat_wc, joints_gt_rot, seq_len, qd_size)
                
                ########## Ground truth data from AIST ##########
                if args.database == "aist":
                    label_path          = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/aist/subset/labels/"  + label_names[action_id]   + ".pkl")
                    smpl_path           = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/aist/subset/motions/" + smpl_names[action_id]  + ".pkl")
                    setting_path        = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/aist/subset/cameras/" + setting_names[action_id] + ".json")
                    with open(label_path, 'rb') as pkl_file:
                        j3d_data = pickle.load(pkl_file)
                    with open(smpl_path, 'rb') as pkl_file:
                        smpl = pickle.load(pkl_file)
                    with open(setting_path) as json_file:
                        setting = json.load(json_file)

                    j3d_gt              = torch.from_numpy(np.array(j3d_data['keypoints3d'])*1e-2).float()
                    smpl_scaling        = torch.from_numpy(np.array(smpl['smpl_scaling'])).float()      # alpha
                    smpl_poses          = torch.from_numpy(np.array(smpl['smpl_poses'])).float()        # theta
                    smpl_trans          = torch.from_numpy(np.array(smpl['smpl_trans'])).float()*1e-2   # translation [m]
                    
                    theta_gt_cc         = transforms.axis_angle_to_matrix(smpl_poses.reshape(-1, 3)).reshape(-1, 24, 3, 3)
                    theta_gt_cc_trs     = smpl_trans
                    theta_gt_cc_rot     = torch.bmm(torch.Tensor([[1,0,0],[0,0,-1],[0,1,0]]).repeat(seq_len,1,1), theta_gt_cc[:seq_len, 0, ...])
                    theta_gt_cc_jnt     = theta_gt_cc[:, 1:, ...]
                    
                    #### Camera parameters
                    camera              = setting[camera_ids[action_id]-1]
                    cam_R               = transforms.axis_angle_to_matrix(torch.Tensor(camera['rotation'])).squeeze().float()
                    # cam_R               = torch.from_numpy(cv2.Rodrigues(np.array(camera['rotation']))[0]).float()
                    cam_T               = (torch.from_numpy(np.array(camera['translation']))*1e-2).float()
                    cam_G               = torch.eye(4)
                    cam_G[:3,:3]        = cam_R
                    cam_G[:3,3]         = cam_T
                    
                    # Transform TRACE inputs from camera space to world space
                    theta_root_cc       = theta_in_cc[:, 0, :]
                    root_rot_mat_cc     = transforms.axis_angle_to_matrix(theta_root_cc)
                    
                    # Pre-allocate the variables
                    root_trans_wc       = torch.zeros(seq_len, 3)
                    root_quat_wc        = torch.zeros(seq_len, 4)
                    j3d_gt_wc_raw       = torch.zeros_like(j3d_gt)          # ground-truth 3d joints
                    
                    for f in range(seq_len):
                        root_trans_homog        = torch.Tensor([0,0,0,1])
                        root_trans_homog[:3]    = root_trans_cc[f,:]
                        root_trans_wc[f, :]     = (torch.linalg.inv(cam_G) @ root_trans_homog)[:3]
                        qr_homog                = torch.eye(4)
                        qr_homog[:3, :3]        = root_rot_mat_cc[f, :]
                        root_rot_wc             = torch.mm(torch.Tensor([[1,0,0],[0,0,-1],[0,1,0]]), (torch.linalg.inv(cam_G) @ qr_homog)[:3, :3])
                        root_quat_wc[f, :]      = transforms.matrix_to_quaternion(root_rot_wc)
                    
                    theta_joints        = theta_in_cc[:seq_len, 1:, :]
                    joints_rot_mat      = transforms.axis_angle_to_matrix(theta_joints.reshape(-1, 3))
                    joints_rot          = transforms.matrix_to_euler_angles(joints_rot_mat, "ZXY").reshape(-1, 23, 3)
                    
                    # Converting the root Euler's representation to world coordinate
                    root_gt_trans_wc    = theta_gt_cc_trs[:seq_len,...]
                    root_gt_quat_wc     = transforms.matrix_to_quaternion(theta_gt_cc_rot).squeeze(1)[:seq_len,...]
                    joints_gt_rot       = transforms.matrix_to_euler_angles(theta_gt_cc_jnt, "ZXY")[:seq_len,...]
                    j3d_gt_wc_raw       = j3d_gt[:seq_len,...]
                    
                    # Mapping joint configuration from Fit3D to the simulated character
                    gt_state_raw        = data_mapping_smpl(root_gt_trans_wc, root_gt_quat_wc, joints_gt_rot, seq_len, qd_size)
                
                ########## Ground truth data from SportPose ##########
                if args.database == "sport":
                    
                    label_path          = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/SportsPose/data/indoors/" 
                                           + subject_names[subject_id] + "/" 
                                           + action_names[action_id] + ".npy")
                    timing_path         = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/SportsPose/data/indoors/" 
                                           + subject_names[subject_id] + "/" 
                                           + action_names[action_id] + "_timing.pkl")
                    calib_path          = ("/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/SportsPose/data/indoors/" 
                                           + subject_names[subject_id] + "/calib.pkl")
                    
                    j3d_data = np.load(label_path)
                    with open(calib_path, 'rb') as pkl_file:
                        cam_param = pickle.load(pkl_file)
                    with open(timing_path, 'rb') as pkl_file:
                        timing = pickle.load(pkl_file)
                    
                    j3d_gt              = torch.from_numpy(j3d_data).float()
                    cam_ex              = np.array(cam_param['calibration'])
                    seq_len             = root_trans_cc.shape[0]
                    
                    ang                 = np.pi/2
                    Rz                  = torch.Tensor([[np.cos(ang), -np.sin(ang), 0, 0],
                                                       [np.sin(ang), np.cos(ang), 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]])
                    
                    # Camera parameters depending on views and subjects
                    cam_R               = torch.from_numpy(cam_ex[3]['R']).float()
                    cam_T               = torch.from_numpy(cam_ex[3]['T']).float()
                    cam_G               = torch.eye(4)                        # Transformation matrix
                    cam_G[:3,:3]        = cam_R
                    cam_G[:3,3]         = cam_T
                    
                    # Transform TRACE inputs from camera space to world space
                    root_trans_cc       = root_trans_cc[:seq_len, :]
                    theta_root_cc       = theta_in_cc[:seq_len, 0, :]
                    root_rot_mat_cc     = transforms.axis_angle_to_matrix(theta_root_cc)
                    
                    # Pre-allocate the variables
                    root_trans_wc       = torch.zeros(seq_len, 3)
                    root_quat_wc        = torch.zeros(seq_len, 4)
                    j3d_gt_wc_raw       = torch.zeros_like(j3d_gt)          # ground-truth 3d joints
                    
                    for f in range(seq_len):
                        root_trans_homog        = torch.Tensor([0,0,0,1])
                        root_trans_homog[:3]    = root_trans_cc[f,:]
                        root_trans_wc[f, :]     = (torch.inverse(cam_G) @ -torch.inverse(Rz) @ root_trans_homog)[:3]
                        qr_homog                = torch.eye(4)
                        qr_homog[:3, :3]        = root_rot_mat_cc[f, :]
                        root_quat_wc[f, :]      = transforms.matrix_to_quaternion((torch.inverse(cam_G) @ Rz @ qr_homog)[:3, :3])
                    
                    theta_joints        = theta_in_cc[:seq_len, 1:, :]
                    joints_rot_mat      = transforms.axis_angle_to_matrix(theta_joints.reshape(-1, 3))
                    joints_rot          = transforms.matrix_to_euler_angles(joints_rot_mat, "ZXY").reshape(-1, 23, 3)
                    
                    # Converting the root Euler's representation to world coordinate
                    j3d_gt_wc_raw[...,0] = -j3d_gt[...,0]
                    j3d_gt_wc_raw[...,1] = j3d_gt[...,1]
                    j3d_gt_wc_raw[...,2] = -j3d_gt[...,2]
                    
                    # root_trans_wc[...,2] = -root_trans_wc[...,2]
                    
                # Mapping joint configuration from TRACE to the simulated character
                input_state_raw = data_mapping_smpl(root_trans_wc,
                                                    root_quat_wc,
                                                    joints_rot,
                                                    seq_len,
                                                    qd_size)

                if args.database == "h36m" or args.database == "fit3d":
                    # Align the signals to start at [0,0,z]
                    vert_offset             = gt_state_raw[0,2] - input_state_raw[0,2]
                    input_state             = input_state_raw.clone()
                    input_state[:,:2]       = input_state_raw[:,:2] - input_state_raw[0,:2]
                    input_state[:,2]        = input_state[:,2] + vert_offset
                    gt_state                = gt_state_raw.clone()
                    gt_state[:,:2]          = gt_state_raw[:,:2] - gt_state_raw[0,:2]
                    j3d_gt_wc               = j3d_gt_wc_raw.clone()
                    j3d_gt_wc[:,:,:2]       = j3d_gt_wc_raw[:,:,:2] - j3d_gt_wc_raw[0,0,:2]
                    
                    # Converting from 50hz to 25 Hz
                    inx            = np.array([i for i in range(seq_len) if i%2!=0])
                    input_state    = input_state[inx,...]
                    gt_state       = gt_state[inx,...]
                    j3d_gt_wc      = j3d_gt_wc[inx,...]
                    seq_len_n      = inx.shape[0]

                    if subject_names[subject_id] in train_subs:
                        train_len       = args.train_len
                        num_samples     = int(seq_len_n/train_len)
                        for k in range(num_samples):
                            sample_q    = input_state[train_len*k:train_len*k+train_len, :]
                            sample_qgt  = gt_state[train_len*k:train_len*k+train_len, :]
                            sample_j3d  = j3d_gt_wc[train_len*k:train_len*k+train_len, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [3,17,6,19])
                            train_set.append([sample_q, sample_qgt, sample_j3d, sample_c])
                        if (seq_len_n%train_len) > 0:
                            sample_q    = input_state[-train_len:, :]
                            sample_qgt  = gt_state[-train_len:, :]
                            sample_j3d  = j3d_gt_wc[-train_len:, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [3,17,6,19])
                            train_set.append([sample_q, sample_qgt, sample_j3d, sample_c])
                    
                    if subject_names[subject_id] in test_subs:
                        test_len       = args.test_len
                        num_samples     = int(seq_len_n/test_len)
                        # num_samples     = 4 # Ablation
                        for k in range(num_samples):
                            sample_q    = input_state[test_len*k:test_len*k+test_len, :]
                            sample_qgt  = gt_state[test_len*k:test_len*k+test_len, :]
                            sample_j3d  = j3d_gt_wc[test_len*k:test_len*k+test_len, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [3,17,6,19])
                            test_set.append([sample_q, sample_qgt, sample_j3d, sample_c, 
                                             [subject_names[subject_id], action_names[action_id], test_len*k]])
                        if (seq_len_n%test_len) > 0:
                            sample_q    = input_state[-test_len:, :]
                            sample_qgt  = gt_state[-test_len:, :]
                            sample_j3d  = j3d_gt_wc[-test_len:, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [3,17,6,19])
                            test_set.append([sample_q, sample_qgt, sample_j3d, sample_c, 
                                             [subject_names[subject_id], action_names[action_id], seq_len_n-test_len]])
                
                elif args.database == "aist":
                    # [xyz] => [zxy]
                    j3d_gt_wc_n             = j3d_gt_wc_raw.clone()
                    j3d_root                = (j3d_gt_wc_raw[:,11,:] + j3d_gt_wc_raw[:,12,:])/2
                    j3d_gt_wc_n             = (j3d_gt_wc_raw - j3d_root.unsqueeze(1)[0,...]) + torch.Tensor([0,0.89,0]).reshape(1,1,-1)
                    j3d_gt_wc               = j3d_gt_wc_n.clone()
                    j3d_gt_wc[...,0]        = j3d_gt_wc_n[...,2]
                    j3d_gt_wc[...,1]        = j3d_gt_wc_n[...,0]
                    j3d_gt_wc[...,2]        = j3d_gt_wc_n[...,1]
                    
                    gt_state_n              = gt_state_raw.clone()
                    gt_state_n[:,:3]        = (gt_state_raw[:,:3] - gt_state_raw[0,:3]) + torch.Tensor([0,0.89,0]).reshape(1,-1)
                    gt_state                = gt_state_n.clone()
                    gt_state[...,0]         = gt_state_n[...,2]
                    gt_state[...,1]         = gt_state_n[...,0]
                    gt_state[...,2]         = gt_state_n[...,1]
                    
                    input_state_n           = input_state_raw.clone()
                    input_state_n[:,:3]     = (input_state_raw[:,:3] - input_state_raw[0,:3]) + torch.Tensor([0,0.89,0]).reshape(1,-1)
                    input_state             = input_state_n.clone()
                    input_state[...,0]      = input_state_n[...,2]
                    input_state[...,1]      = input_state_n[...,0]
                    input_state[...,2]      = input_state_n[...,1]
                    
                    # Converting from 50hz to 25 Hz
                    inx            = np.array([i for i in range(seq_len) if i%2!=0])
                    input_state    = input_state[inx,...]
                    gt_state       = gt_state[inx,...]
                    j3d_gt_wc      = j3d_gt_wc[inx,...]
                    seq_len_n      = inx.shape[0]
                    
                    test_len       = args.test_len
                    num_samples     = int(seq_len_n/test_len)
                    for k in range(num_samples):
                        sample_q    = input_state[test_len*k:test_len*k+test_len, :]
                        sample_qgt  = gt_state[test_len*k:test_len*k+test_len, :]
                        sample_j3d  = j3d_gt_wc[test_len*k:test_len*k+test_len, ...]
                        sample_c    = get_contact_labels(sample_j3d, sample_qgt)
                        test_set.append([sample_q, sample_qgt, sample_j3d, sample_c,
                                         [action_names[action_id], test_len*k]])
                    if (seq_len_n%test_len) > 0:
                        sample_q    = input_state[-test_len:, :]
                        sample_qgt  = gt_state[-test_len:, :]
                        sample_j3d  = j3d_gt_wc[-test_len:, ...]
                        sample_c    = get_contact_labels(sample_j3d, sample_qgt)
                        test_set.append([sample_q, sample_qgt, sample_j3d, sample_c,
                                         [action_names[action_id], seq_len_n-test_len]])
                
                elif args.database == "sport":
                    
                    root                    = (j3d_gt_wc_raw[:,11,:] + j3d_gt_wc_raw[:,12,:])/2
                    vert_offset             = root[0,2] - input_state_raw[0,2]
                    input_state             = input_state_raw.clone()
                    input_state[:,:2]       = input_state_raw[:,:2] - input_state_raw[0,:2]
                    input_state[:,2]        = input_state[:,2] + vert_offset
                    gt_state                = input_state_raw.clone()
                    gt_state[:,:2]          = input_state_raw[:,:2] - input_state_raw[0,:2]
                    j3d_gt_wc               = torch.cat((root.unsqueeze(1), j3d_gt_wc_raw.clone()), dim=1)
                    j3d_gt_wc[:,:,:2]       = j3d_gt_wc[:,:,:2] - root[0,:2]
                    
                    # Converting from 50hz to 25 Hz
                    inx            = np.array([i for i in range(seq_len) if i%2!=0])
                    input_state    = input_state[inx,...]
                    gt_state       = gt_state[inx,...]
                    j3d_gt_wc      = j3d_gt_wc[inx,...]
                    seq_len_n      = inx.shape[0]
                    
                    if subject_names[subject_id] in train_subs:
                        train_len       = args.train_len
                        num_samples     = int(seq_len_n/train_len)
                        for k in range(num_samples):
                            sample_q    = input_state[train_len*k:train_len*k+train_len, :]
                            sample_qgt  = gt_state[train_len*k:train_len*k+train_len, :]
                            sample_j3d  = j3d_gt_wc[train_len*k:train_len*k+train_len, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [16,16,17,17])
                            train_set.append([sample_q, sample_qgt, sample_j3d, sample_c])
                        if (seq_len_n%train_len) > 0:
                            sample_q    = input_state[-train_len:, :]
                            sample_qgt  = gt_state[-train_len:, :]
                            sample_j3d  = j3d_gt_wc[-train_len:, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [16,16,17,17])
                            train_set.append([sample_q, sample_qgt, sample_j3d, sample_c])
                    
                    if subject_names[subject_id] in test_subs:
                        test_len       = args.test_len
                        num_samples     = int(seq_len_n/test_len)
                        # num_samples     = 4 # Ablation
                        for k in range(num_samples):
                            sample_q    = input_state[test_len*k:test_len*k+test_len, :]
                            sample_qgt  = gt_state[test_len*k:test_len*k+test_len, :]
                            sample_j3d  = j3d_gt_wc[test_len*k:test_len*k+test_len, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [16,16,17,17])
                            test_set.append([sample_q, sample_qgt, sample_j3d, sample_c, 
                                             [subject_names[subject_id], action_names[action_id], test_len*k]])
                        if (seq_len_n%test_len) > 0:
                            sample_q    = input_state[-test_len:, :]
                            sample_qgt  = gt_state[-test_len:, :]
                            sample_j3d  = j3d_gt_wc[-test_len:, ...]
                            sample_c    = get_contact_labels(sample_j3d, sample_qgt, [16,16,17,17])
                            test_set.append([sample_q, sample_qgt, sample_j3d, sample_c, 
                                             [subject_names[subject_id], action_names[action_id], seq_len_n-test_len]])
                    
    print("Total training samples extracted with length " + str(args.train_len) + ": " + str(len(train_set)))
    print("Total testing samples extracted with length " + str(args.test_len) + ":  " + str(len(test_set)))
    torch.save(train_set, "datasets/" + args.database + "/" + experiment + "/train_set.pt")
    torch.save(test_set,  "datasets/" + args.database + "/" + experiment + "/test_set.pt")