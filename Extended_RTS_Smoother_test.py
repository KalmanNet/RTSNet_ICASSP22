import torch
import torch.nn as nn
import time
from EKF import ExtendedKalmanFilter
from Extended_RTS_Smoother import Extended_rts_smoother
from Extended_data import N_T

def S_Test(SysModel, test_input, test_target, modelKnowledge = 'full'):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_ERTS_linear_arr = torch.empty(N_T)
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    ERTS = Extended_rts_smoother(SysModel, modelKnowledge)
    ERTS_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    
    for j in range(0, N_T):

        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)
        ERTS.GenerateSequence(EKF.x, EKF.sigma, ERTS.T_test)
        MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x, test_target[j, :, :]).item()
        ERTS_out[j,:,:] = ERTS.s_x
    end = time.time()
    t = end - start
    MSE_ERTS_linear_avg = torch.mean(MSE_ERTS_linear_arr)
    MSE_ERTS_dB_avg = 10 * torch.log10(MSE_ERTS_linear_avg)

    print("Extended RTS Smoother - MSE LOSS:", MSE_ERTS_dB_avg, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out]



