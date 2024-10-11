"""
Architecture of the OSDNet
--- Cuong Le - CVL ---
"""

import torch
import torch.nn as nn
from utils import *
torch.manual_seed(0)
    
class OSDNet(nn.Module):
    def __init__(self, in_dim, gru_dim, hid_dim, nlayers, out_dim, pd_temp, c_temp, use_osd):
        super().__init__()
        
        self.q_size         = q_size
        self.qd_size        = qd_size
        self.gru_dim        = gru_dim
        self.nlayers        = nlayers
        self.pd_temp        = pd_temp
        self.c_temp         = c_temp
        self.use_osd        = use_osd
        
        limit_p         = torch.empty((66,)).to(device)
        limit_p[:3], limit_p[3:6], limit_p[6:] = 25.0, 12.0, 1.6
        limit_d         = torch.empty((66,)).to(device)
        limit_d[:3], limit_d[3:6], limit_d[6:] = 1.5, 0.1, 0.05

        # Hidden layers.
        self.layer1         = nn.Linear(in_dim + qd_size**2, hid_dim)
        self.layer2         = nn.Linear(hid_dim, hid_dim)
        self.layer3         = nn.Linear(hid_dim, hid_dim)
        
        # Optimal-state head.
        if self.use_osd:
            self.gru            = nn.GRU(in_dim + qd_size**2 + q_size*4, gru_dim, num_layers=nlayers, batch_first=True)
            self.gru_2out_k     = nn.Linear(gru_dim, out_dim)
            self.hid_2out_k     = nn.Linear(hid_dim, out_dim)
            self.out_2k         = nn.Linear(out_dim*2, q_size**2)
            self.norm_gru       = nn.LayerNorm(gru_dim)
            with torch.no_grad():
                self.init_weights([self.out_2k], 1e-3)
                self.out_2k.bias.data += torch.eye(q_size).view(-1) * 0.5
        
        # Inertia-bias head  
        self.hid_2out_m     = nn.Linear(hid_dim, out_dim)
        self.out_2m         = nn.Linear(out_dim, qd_size**2)

        # Proportional gain head
        self.lim_p          = nn.Parameter(limit_p, requires_grad=True)
        self.hid_2out_p     = nn.Linear(hid_dim, out_dim)
        self.out_2p         = nn.Linear(out_dim, qd_size)
        
        # Derivative gain head
        self.lim_d          = nn.Parameter(limit_d, requires_grad=True)
        self.hid_2out_d     = nn.Linear(hid_dim, out_dim)
        self.out_2d         = nn.Linear(out_dim, qd_size)

        # Contact head
        self.hid_2out_c     = nn.Linear(hid_dim, out_dim)
        self.out_2c1        = nn.Linear(out_dim+73, out_dim)
        self.out_2c2        = nn.Linear(out_dim, out_dim)
        self.out_2c3        = nn.Linear(out_dim, 2)
        
        # External force head
        self.hid_2out_f     = nn.Linear(hid_dim, out_dim)
        self.out_2f1        = nn.Linear(out_dim+73, out_dim)
        self.out_2f2        = nn.Linear(out_dim, out_dim)
        self.out_2f3        = nn.Linear(out_dim, 2*3)
        
        # Jacobian head
        self.hid_2out_j     = nn.Linear(hid_dim, out_dim)
        self.out_2j         = nn.Linear(out_dim, 2*(qd_size-3)*3)

        self.bone_length    = nn.Parameter(bone_length_h36m.clone(), requires_grad=True)
        self.offset         = nn.Parameter(1e-2 * torch.randn(3).to(device), requires_grad=True)
        self.q_mapp         = nn.Parameter((torch.eye(q_size) + 1e-6*torch.randn(q_size)).to(device), requires_grad=True)        
        self.qd_mapp        = nn.Parameter(dt * torch.randn(qd_size, qd_size).to(device), requires_grad=True)
        self.qd_bias        = nn.Parameter(1e-2 * torch.randn(qd_size).to(device), requires_grad=True)
        self.grf_comp       = nn.Parameter(torch.Tensor([0.15]).to(device), requires_grad=True)
        self.comp_filter    = nn.Parameter(torch.Tensor([0.90]).to(device), requires_grad=True)
        self.obs_model      = nn.Parameter((torch.eye(q_size) + 1e-6*torch.randn(q_size)).to(device), requires_grad=True)
        self.adp_model      = nn.Parameter((torch.eye(q_size) + 1e-6*torch.randn(q_size)).to(device), requires_grad=True)
        
        # Layer norm and activations.
        self.norm_hid       = nn.LayerNorm(hid_dim)
        self.norm_out       = nn.LayerNorm(out_dim)
        self.act            = nn.LeakyReLU()
        self.tanh           = nn.Tanh()
        self.dropout        = nn.Dropout(p=0.9)
        
        # Init weights
        with torch.no_grad():
            self.init_weights([self.out_2m, self.out_2j])
            nn.init.kaiming_uniform_(self.out_2f2.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.xavier_uniform_(self.out_2c2.weight)
            self.out_2f3.bias[[2, 5]] = 9.81
        
    def forward(self, state_input, kalman_in, contact_input, h):
        
        hidden_1            = self.act(self.norm_hid(self.layer1(state_input)))
        hidden_2            = self.act(self.norm_hid(self.layer2(hidden_1)))
        hidden_3            = self.act(self.norm_hid(self.layer3(hidden_2)))
        
        # Optimal-state head
        if self.use_osd:
            gru_in              = torch.cat((state_input, kalman_in), dim=-1)
            gru_out, h          = self.gru(gru_in, h)
            gru_out             = self.act(self.norm_gru(gru_out))
            gru_2k              = self.act(self.gru_2out_k(gru_out))
            out_k               = self.act(self.hid_2out_k(hidden_3))
            opt_input           = self.dropout(torch.cat((gru_2k, out_k), dim=-1))
            kalman              = self.out_2k(opt_input).reshape(-1, self.q_size, self.q_size)

        # Inertia head
        out_m               = self.act(self.hid_2out_m(hidden_3))
        m_base              = self.out_2m(out_m).reshape(-1, self.qd_size, self.qd_size)
        M_bias              = m_base + m_base.permute(0,2,1)
        
        # PD gain head
        out_p               = self.act(self.hid_2out_p(hidden_3))
        out_d               = self.act(self.hid_2out_d(hidden_3))
        p_gains             = torch.diag_embed(self.lim_p * torch.sigmoid(self.out_2p(out_p) / self.pd_temp)).squeeze(1)
        d_gains             = torch.diag_embed(self.lim_d * torch.sigmoid(self.out_2d(out_d) / self.pd_temp)).squeeze(1)

        # Contact head
        out_c               = self.act(self.hid_2out_c(hidden_3))
        c_hid1              = self.tanh(self.out_2c1(torch.cat((out_c, contact_input), dim=-1)))
        c_hid2              = self.tanh(self.norm_out(self.out_2c2(c_hid1)))
        contacts            = torch.sigmoid(self.out_2c3(c_hid2).squeeze(1) / self.c_temp)
        
        # F_ext head
        out_f               = self.act(self.hid_2out_f(hidden_3))
        f_hid1              = self.act(self.out_2f1(torch.cat((out_f, contact_input), dim=-1)))
        f_hid2              = self.act(self.norm_out(self.out_2f2(f_hid1)))
        fext                = self.out_2f3(f_hid2).reshape(-1, 2, 3)
        
        # Jacobian
        out_j               = self.act(self.hid_2out_j(hidden_3))
        jac_int             = self.out_2j(out_j).reshape(-1, 2, (self.qd_size-3), 3)
        jac_ext             = torch.eye(3,3).repeat(jac_int.shape[0], jac_int.shape[1], 1, 1).to(device)
        jacs                = torch.cat((jac_ext, jac_int), dim=2)
        
        if self.use_osd:
            return kalman, h, [p_gains, d_gains], M_bias, contacts, fext, jacs
        else:
            return [p_gains, d_gains], M_bias, contacts, fext, jacs
    
    def init_weights(self, layers, scale=1e-5):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            layer.weight.data += torch.randn(layer.weight.shape) * scale
            nn.init.zeros_(layer.bias)
            layer.bias.data += torch.randn(layer.bias.shape) * scale
            
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.nlayers, batch_size, self.gru_dim).zero_().to(device)
        return hidden

    
    
        