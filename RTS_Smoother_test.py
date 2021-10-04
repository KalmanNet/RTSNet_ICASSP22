import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter
from RTS_Smoother import rts_smoother
from Extended_data import N_T

def S_Test(SysModel, test_input, test_target):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_RTS_linear_arr = torch.empty(N_T)
    start = time.time()
    KF = KalmanFilter(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    RTS = rts_smoother(SysModel)
    
    for j in range(0, N_T):

        KF.GenerateSequence(test_input[j, :, :], KF.T_test)
        RTS.GenerateSequence(KF.x, KF.sigma, RTS.T_test)
        MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x, test_target[j, :, :]).item()
    end = time.time()
    t = end - start
    MSE_RTS_linear_avg = torch.mean(MSE_RTS_linear_arr)
    MSE_RTS_dB_avg = 10 * torch.log10(MSE_RTS_linear_avg)

    print("RTS Smoother - MSE LOSS:", MSE_RTS_dB_avg, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg]



