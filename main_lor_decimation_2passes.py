import numpy as np
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import pickle
import torch.nn as nn
from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_ERTS_2passes import Pipeline_ERTS as Pipeline

from RTSNet_nn_2passes import RTSNetNN_2passes

# from PF_test import PFTest

from datetime import datetime

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

######################################
###  Compare EKF, RTS and RTSNet   ###
######################################
offset = 0
chop = False
sequential_training = False
path_results = 'ERTSNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/'
data_gen = 'data_gen.pt'
data_gen_file = torch.load(DatafolderName+data_gen, map_location=cuda0)
[true_sequence] = data_gen_file['All Data']

r = torch.tensor([1.])
lambda_q = torch.tensor([0.3873])
traj_resultName = ['traj_lor_dec_RTSNetJ2_r0_2pass.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
# EKFResultName = 'EKF_obsmis_rq1030_T2000_NT100' 

for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("Search 1/q2 [dB]: ", 10 * torch.log10(1/lambda_q[rindex]**2))
   # Q_mod = (lambda_q[rindex]**2) * torch.eye(m)
   # R_mod = (r[rindex]**2) * torch.eye(n)
   # True Model
   sys_model_true = SystemModel(f, lambda_q[rindex], h, r[rindex], T, T_test,m,n)
   sys_model_true.InitSequence(m1x_0, m2x_0)

   # Model with partial Info
   sys_model = SystemModel(fInacc, lambda_q[rindex], h, r[rindex], T, T_test,m,n)
   sys_model.InitSequence(m1x_0, m2x_0)

   #Generate and load data Decimation case (chopped)
   print("Data Gen")
   [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   print("testset size:",test_target.size())
   [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)
   if chop:
      print("chop training data")  
      [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   else:
      print("no chopping") 
      train_target = train_target_long
      train_input = train_input_long
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target_long.size())
   
   ## Load data from Welling's
   # compact_path = "ERTSNet/new_arch_LA/decimation/Welling_Compare/lorenz_trainset300k.pickle"
   # with open(compact_path, 'rb') as f:
   #    data = pickle.load(f)
   # testdata = [data[0][0:T_test], data[1][0:T_test]]
   # states, meas = testdata
   # test_target =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # test_input = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # print("testset size:",test_target.size())
   # traindata = [data[0][T_test:(T_test+T*N_E)], data[1][T_test:(T_test+T*N_E)]]
   # states, meas = traindata
   # train_target =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # train_input = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # [train_target, train_input] = Short_Traj_Split(train_target, train_input, T)
   # cvdata = [data[0][(T_test+T*N_E):], data[1][(T_test+T*N_E):]]
   # states, meas = cvdata
   # cv_target_long =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # cv_input_long = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # [cv_target_long, cv_input_long] = Short_Traj_Split(cv_target_long, cv_input_long, T)
   # print("trainset size:",train_target.size())
   # print("cvset size:",cv_target_long.size())

   # Particle filter
   # print("Start PF test")
   # [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out, t_PF] = PFTest(sys_model_true, test_input, test_target, init_cond=None)
   # print(f"MSE PF J=5: {MSE_PF_dB_avg} [dB] (T = {T_test})")
   # [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial, t_PF] = PFTest(sys_model, test_input, test_target, init_cond=None)
   # print(f"MSE PF J=2: {MSE_PF_dB_avg} [dB] (T = {T_test})")
   
   # EKF
   # print("Start EKF test")
   # [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model_true, test_input, test_target)
   # print(f"MSE EKF J=5: {MSE_EKF_dB_avg} [dB] (T = {T_test})")
   # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model, test_input, test_target)
   # print(f"MSE EKF J=2: {MSE_EKF_dB_avg_partial} [dB] (T = {T_test})")

   # # MB Extended RTS
   # print("Start RTS test")
   # [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model_true, test_input, test_target)
   # print(f"MSE RTS J=5: {MSE_ERTS_dB_avg} [dB] (T = {T_test})")
   # [MSE_ERTS_linear_arr_partial, MSE_ERTS_linear_avg_partial, MSE_ERTS_dB_avg_partial, ERTS_out_partial] = S_Test(sys_model, test_input, test_target)
   # print(f"MSE RTS J=2: {MSE_ERTS_dB_avg_partial} [dB] (T = {T_test})")
   
   # KNet with model mismatch
   # ## Build Neural Network
   # KNet_model = KalmanNetNN()
   # KNet_model.NNBuild(sys_model)
   # ## Train Neural Network
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   # KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
   # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results, sequential_training)
   # ## Test Neural Network
   # # KNet_Pipeline.model = torch.load('KNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
   # [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_KG_array, knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   # # Print MSE Cross Validation
   # print("MSE Test:", MSE_test_dB_avg, "[dB]")

   # RTSNet with model mismatch
   ## Build Neural Network
   print("RTSNet with model mismatch")
   RTSNet_model = RTSNetNN_2passes()
   RTSNet_model.NNBuild(sys_model)
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=1, learningRate=1e-3, weightDecay=1e-4)
   NumofParameter = RTSNet_Pipeline.count_parameters()
   print("Number of parameters for RTSNet: ",NumofParameter)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results)
   ## Test Neural Network
   # RTSNet_Pipeline.model = torch.load('ERTSNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   # Print MSE Cross Validation
   print("MSE Test:", MSE_test_dB_avg, "[dB]")
   
   # Save trajectories
   trajfolderName = 'ERTSNet' + '/'
   DataResultName = traj_resultName[rindex]
   target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
   input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
   torch.save({#'PF J=5':PF_out,
               #'PF J=2':PF_out_partial,
               'True':target_sample,
               'Observation':input_sample,
               # 'EKF J=5':EKF_out,
               # 'EKF J=2':EKF_out_partial,
               # 'RTS J=5':ERTS_out,
               # 'RTS J=2':ERTS_out_partial,
               'RTSNet': rtsnet_out,
               }, trajfolderName+DataResultName)

   ## Save histogram
   # MSE_ResultName = 'Partial_MSE_KNet' 
   # torch.save(KNet_MSE_test_dB_avg,trajfolderName + MSE_ResultName)

   





