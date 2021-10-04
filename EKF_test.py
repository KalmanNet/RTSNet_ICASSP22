import torch.nn as nn
import torch
import time
from EKF import ExtendedKalmanFilter


def EKFTest(SysModel, test_input, test_target, modelKnowledge = 'full', allStates=True):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    
    for j in range(0, N_T):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        if(allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True,False,True,False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc,:], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array) 
        EKF_out[j,:,:] = EKF.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)
    
    print("Extended Kalman Filter - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]


def EKFTest_evol(SysModel, test_input, test_target, modelKnowledge = 'full'):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='none')
    
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T,SysModel.m, SysModel.T_test)
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.empty([N_T, SysModel.T_test, SysModel.m, SysModel.n])
    KG_trace = torch.empty([SysModel.T_test])
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    
    for j in range(0, N_T):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        MSE_EKF_linear_arr[j,:,:] = loss_fn(EKF.x, test_target[j, :, :])
        KG_array[j,:,:,:] = EKF.KG_array
        EKF_out[j,:,:] = EKF.x
    # Average KG_array over Test Examples

    KG_avg = torch.mean(KG_array,0)
    for j in range(0, SysModel.T_test):
        KG_trace[j] = torch.trace(KG_avg[j,:,:])

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr, [0,1])
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)
    trace_dB_avg = 10* torch.log10(KG_trace)

    return [MSE_EKF_dB_avg, trace_dB_avg]



