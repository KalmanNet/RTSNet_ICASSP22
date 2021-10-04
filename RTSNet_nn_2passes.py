"""# **Class: RTSNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from RTSNet_nn import RTSNetNN

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

in_mult = 5
out_mult = 40
nGRU = 4

class RTSNetNN_2passes(RTSNetNN):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    #############
    ### Build ###
    #############
    def NNBuild(self, ssModel, infoString = 'fullInfo'):

        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n, infoString = 'fullInfo')
        self.InitSequence(ssModel.m1x_0, ssModel.m2x_0, ssModel.T)

        self.InitKGainNet(ssModel.prior_Q, ssModel.prior_Sigma, ssModel.prior_S)

        self.InitRTSGainNet(ssModel.prior_Q, ssModel.prior_Sigma)

        self.InitKGainNet_pass2()

        self.InitRTSGainNet_pass2()
    
    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet_pass2(self):

        self.seq_len_input = 1
        self.batch_size = 1
        
        # GRU to track Q
        self.d_input_Q_2 = self.m * in_mult
        self.d_hidden_Q_2 = self.m ** 2
        self.GRU_Q_2 = nn.GRU(self.d_input_Q_2, self.d_hidden_Q_2)
        self.h_Q_2 = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q_2).to(dev, non_blocking=True)

        # GRU to track Sigma
        self.d_input_Sigma_2 = self.d_hidden_Q_2 + self.m * in_mult
        self.d_hidden_Sigma_2 = self.m ** 2
        self.GRU_Sigma_2 = nn.GRU(self.d_input_Sigma_2, self.d_hidden_Sigma_2)
        self.h_Sigma_2 = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma_2).to(dev, non_blocking=True)

        # GRU to track S
        self.d_input_S_2 = self.n ** 2 + 2 * self.n * in_mult
        self.d_hidden_S_2 = self.n ** 2
        self.GRU_S_2 = nn.GRU(self.d_input_S_2, self.d_hidden_S_2)
        self.h_S_2 = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_S_2).to(dev, non_blocking=True)

        # Fully connected 1
        self.d_input_FC1_2 = self.d_hidden_Sigma_2
        self.d_output_FC1_2 = self.n ** 2
        self.FC1_2 = nn.Sequential(
                nn.Linear(self.d_input_FC1_2, self.d_output_FC1_2),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2_2 = self.d_hidden_S_2 + self.d_hidden_Sigma_2
        self.d_output_FC2_2 = self.n * self.m
        self.d_hidden_FC2_2 = self.d_input_FC2_2 * out_mult
        self.FC2_2 = nn.Sequential(
                nn.Linear(self.d_input_FC2_2, self.d_hidden_FC2_2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2_2, self.d_output_FC2_2))

        # Fully connected 3
        self.d_input_FC3_2 = self.d_hidden_S_2 + self.d_output_FC2_2
        self.d_output_FC3_2 = self.m ** 2
        self.FC3_2 = nn.Sequential(
                nn.Linear(self.d_input_FC3_2, self.d_output_FC3_2),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4_2 = self.d_hidden_Sigma_2 + self.d_output_FC3_2
        self.d_output_FC4_2 = self.d_hidden_Sigma_2
        self.FC4_2 = nn.Sequential(
                nn.Linear(self.d_input_FC4_2, self.d_output_FC4_2),
                nn.ReLU())
        
        # Fully connected 5
        self.d_input_FC5_2 = self.m
        self.d_output_FC5_2 = self.m * in_mult
        self.FC5_2 = nn.Sequential(
                nn.Linear(self.d_input_FC5_2, self.d_output_FC5_2),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6_2 = self.m
        self.d_output_FC6_2 = self.m * in_mult
        self.FC6_2 = nn.Sequential(
                nn.Linear(self.d_input_FC6_2, self.d_output_FC6_2),
                nn.ReLU())
        
        # Fully connected 7
        self.d_input_FC7_2 = 2 * self.n
        self.d_output_FC7_2 = 2 * self.n * in_mult
        self.FC7_2 = nn.Sequential(
                nn.Linear(self.d_input_FC7_2, self.d_output_FC7_2),
                nn.ReLU())

    #################################################
    ### Initialize Backward Smoother Gain Network ###
    #################################################
    def InitRTSGainNet_pass2(self):
        self.seq_len_input = 1
        self.batch_size = 1

        # BW GRU to track Q
        self.d_input_Q_bw_2 = self.m * in_mult
        self.d_hidden_Q_bw_2 = self.m ** 2
        self.GRU_Q_bw_2 = nn.GRU(self.d_input_Q_bw_2, self.d_hidden_Q_bw_2)
        self.h_Q_bw_2 = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q_bw_2).to(dev, non_blocking=True)

        # BW GRU to track Sigma
        self.d_input_Sigma_bw_2 = self.d_hidden_Q_bw_2 + 2 * self.m * in_mult
        self.d_hidden_Sigma_bw_2 = self.m ** 2
        self.GRU_Sigma_bw_2 = nn.GRU(self.d_input_Sigma_bw_2, self.d_hidden_Sigma_bw_2)
        self.h_Sigma_bw_2 = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma_bw_2).to(dev, non_blocking=True)

        # BW Fully connected 1
        self.d_input_FC1_bw_2 = self.d_hidden_Sigma_bw_2 # + self.d_hidden_Q
        self.d_output_FC1_bw_2 = self.m * self.m
        self.d_hidden_FC1_bw_2 = self.d_input_FC1_bw_2 * out_mult
        self.FC1_bw_2 = nn.Sequential(
                nn.Linear(self.d_input_FC1_bw_2, self.d_hidden_FC1_bw_2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC1_bw_2, self.d_output_FC1_bw_2))

        # BW Fully connected 2
        self.d_input_FC2_bw_2 = self.d_hidden_Sigma_bw_2 + self.d_output_FC1_bw_2
        self.d_output_FC2_bw_2 = self.d_hidden_Sigma_bw_2
        self.FC2_bw_2 = nn.Sequential(
                nn.Linear(self.d_input_FC2_bw_2, self.d_output_FC2_bw_2),
                nn.ReLU())
        
        # BW Fully connected 3
        self.d_input_FC3_bw_2 = self.m
        self.d_output_FC3_bw_2 = self.m * in_mult
        self.FC3_bw_2 = nn.Sequential(
                nn.Linear(self.d_input_FC3_bw_2, self.d_output_FC3_bw_2),
                nn.ReLU())

        # BW Fully connected 4
        self.d_input_FC4_bw_2 = 2 * self.m
        self.d_output_FC4_bw_2 = 2 * self.m * in_mult
        self.FC4_bw_2 = nn.Sequential(
                nn.Linear(self.d_input_FC4_bw_2, self.d_output_FC4_bw_2),
                nn.ReLU())

    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward(self, filter_x):
        self.s_m1x_nexttime = torch.squeeze(filter_x)

    ##############################
    ### Innovation Computation ###
    ##############################
    def S_Innovation(self, filter_x):
        self.filter_x_prior = self.f(filter_x)
        self.dx = self.s_m1x_nexttime - self.filter_x_prior
    
    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est_pass2(self, y):

        obs_diff = y - torch.squeeze(self.y_previous)
        obs_innov_diff = y - torch.squeeze(self.m1y)
        fw_evol_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_posterior_previous)
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous)

        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)


        # Kalman Gain Network Step
        KG = self.KGain_step_pass2(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain_pass2 = torch.reshape(KG, (self.m, self.n))

    ################################
    ### Smoother Gain Estimation ###
    ################################
    def step_RTSGain_est_pass2(self, filter_x_nexttime, smoother_x_tplus2):

        # Reshape and Normalize Delta tilde x_t+1 = x_t+1|T - x_t+1|t+1
        dm1x_tilde = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_tilde_reshape = torch.squeeze(dm1x_tilde)
        bw_innov_diff = func.normalize(dm1x_tilde_reshape, p=2, dim=0, eps=1e-12, out=None)
        
        if smoother_x_tplus2 is None:
            # Reshape and Normalize Delta x_t+1 = x_t+1|t+1 - x_t+1|t (for t = T-1)
            dm1x_input2 = filter_x_nexttime - self.filter_x_prior
            dm1x_input2_reshape = torch.squeeze(dm1x_input2)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=0, eps=1e-12, out=None)
        else:
            # Reshape and Normalize Delta x_t+1|T = x_t+2|T - x_t+1|T (for t = 1:T-2)
            dm1x_input2 = smoother_x_tplus2 - self.s_m1x_nexttime
            dm1x_input2_reshape = torch.squeeze(dm1x_input2)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 7:  x_t+1|T - x_t+1|t
        dm1x_f7 = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_f7_reshape = torch.squeeze(dm1x_f7)
        bw_update_diff = func.normalize(dm1x_f7_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Smoother Gain Network Step
        SG = self.RTSGain_step_pass2(bw_innov_diff, bw_evol_diff, bw_update_diff)

        # Reshape Smoother Gain to a Matrix
        self.SGain_pass2 = torch.reshape(SG, (self.m, self.m))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step_pass2(self, y):

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est_pass2(y)

        # Save KGain in array
        # self.KGain_array[self.i] = self.KGain
        # self.i += 1

        # Innovation
        # y_obs = torch.unsqueeze(y, 1)
        dy = y - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain_pass2, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return torch.squeeze(self.m1x_posterior)

    ####################
    ### RTS Net Step ###
    ####################
    def RTSNet_step_pass2(self, filter_x, filter_x_nexttime, smoother_x_tplus2):
        # Compute Innovation
        self.S_Innovation(filter_x)

        # Compute Smoother Gain
        self.step_RTSGain_est_pass2(filter_x_nexttime, smoother_x_tplus2)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.SGain_pass2, self.dx)
        self.s_m1x_nexttime = filter_x + INOV

        # return
        return torch.squeeze(self.s_m1x_nexttime)

    def KGain_step_pass2(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5_2(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q_2 = self.GRU_Q_2(in_Q, self.h_Q_2)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6_2(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma_2 = self.GRU_Sigma_2(in_Sigma, self.h_Sigma_2)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1_2(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7_2(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S_2 = self.GRU_S_2(in_S, self.h_S_2)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2_2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3_2(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4_2(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma_2 = out_FC4

        return out_FC2

    ##########################
    ### Smoother Gain Step ###
    ##########################
    def RTSGain_step_pass2(self, bw_innov_diff, bw_evol_diff, bw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        bw_innov_diff = expand_dim(bw_innov_diff)
        bw_evol_diff = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)
        
        ####################
        ### Forward Flow ###
        ####################
        
        # FC 3
        in_FC3 = bw_update_diff
        out_FC3 = self.FC3_bw_2(in_FC3)

        # Q-GRU
        in_Q = out_FC3
        out_Q, self.h_Q_bw_2 = self.GRU_Q_bw_2(in_Q, self.h_Q_bw_2)

        # FC 4
        in_FC4 = torch.cat((bw_innov_diff, bw_evol_diff), 2)
        out_FC4 = self.FC4_bw_2(in_FC4)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC4), 2)
        out_Sigma, self.h_Sigma_bw_2 = self.GRU_Sigma_bw_2(in_Sigma, self.h_Sigma_bw_2)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1_bw_2(in_FC1)

        #####################
        ### Backward Flow ###
        #####################

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw_2(in_FC2)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma_bw_2 = out_FC2

        return out_FC1

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, filter_x, filter_x_nexttime, smoother_x_tplus2,pass2=False):
        if pass2:
            if yt is None:
                return self.RTSNet_step_pass2(filter_x, filter_x_nexttime, smoother_x_tplus2)
            else:
                yt = yt.to(dev, non_blocking=True)
                return self.KNet_step_pass2(yt)
        else:
            if yt is None:
                return self.RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
            else:
                yt = yt.to(dev, non_blocking=True)
                return self.KNet_step(yt)

        
