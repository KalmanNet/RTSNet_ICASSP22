import torch
import torch.nn as nn
import random
from Plot import Plot
import time

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, nclt=False, sequential_training=False, rnn=False, epochs=None, train_IC=None, CV_IC=None):

        N_E = train_input.size()[0]
        N_CV = cv_input.size()[0]

        MSE_cv_linear_batch = torch.empty([N_CV]).to(dev, non_blocking=True)
        MSE_cv_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
        MSE_cv_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)

        MSE_train_linear_batch = torch.empty([self.N_B]).to(dev, non_blocking=True)
        MSE_train_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
        MSE_train_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)


        ##############
        ### Epochs ###
        ##############

        MSE_cv_dB_opt = 1000
        MSE_cv_idx_opt = 0

        if epochs is None:
            N = self.N_Epochs
        else:
            N = epochs

        for ti in range(0, N):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, N_CV):
                self.model.i = 0
                # Initialize next sequence
                if(sequential_training):
                    if(nclt):
                        init_conditions = torch.reshape(cv_input[j,:,0], SysModel.m1x_0.shape)
                    elif CV_IC is None:
                        init_conditions = torch.reshape(cv_target[j,:,0], SysModel.m1x_0.shape)
                    else:
                        init_conditions = SysModel.m1x_0
                else:
                    init_conditions = SysModel.m1x_0

                self.model.InitSequence(init_conditions, SysModel.m2x_0, SysModel.T)

                y_cv = cv_input[j, :, :]

                x_Net_cv = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)

                for t in range(0, SysModel.T):
                    x_Net_cv[:,t] = self.model(y_cv[:,t])
                

                # Compute Training Loss
                if(nclt):
                    if x_Net_cv.size()[0]==6:
                        mask = torch.tensor([True,False,False,True,False,False])
                    else:
                        mask = torch.tensor([True,False,True,False])
                    MSE_cv_linear_batch[j] = self.loss_fn(x_Net_cv[mask], cv_target[j, :, :]).item()
                else:
                    MSE_cv_linear_batch[j] = self.loss_fn(x_Net_cv, cv_target[j, :, :]).item()

            # Average
            MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            MSE_cv_dB_epoch[ti] = 10 * torch.log10(MSE_cv_linear_epoch[ti])

            if(MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):

                MSE_cv_dB_opt = MSE_cv_dB_epoch[ti]
                MSE_cv_idx_opt = ti
                if(rnn):
                    torch.save(self.model, path_results + 'best-model_rnn.pt')
                else:
                    torch.save(self.model, path_results + 'best-model.pt')

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                self.model.i = 0
                n_e = random.randint(0, N_E - 1)

                y_training = train_input[n_e, :, :]

                if(sequential_training):
                    if(nclt):
                        init_conditions = torch.reshape(cv_input[j,:,0], SysModel.m1x_0.shape)
                    elif CV_IC is None:
                        init_conditions = torch.reshape(cv_target[j,:,0], SysModel.m1x_0.shape)
                    else:
                        init_conditions = SysModel.m1x_0
                else:
                    init_conditions = SysModel.m1x_0


                self.model.InitSequence(init_conditions, SysModel.m2x_0, SysModel.T)
                

                x_Net_training = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
                
                for t in range(0, SysModel.T):
                    x_Net_training[:,t] = self.model(y_training[:,t])

                # Compute Training Loss
                #LOSS = loss_fn(x_Net_training, train_target[n_e, :, :])
                if(nclt):
                    if x_Net_training.size()[0]==6:
                        mask = torch.tensor([True,False,False,True,False,False])
                    else:
                        mask = torch.tensor([True,False,True,False])
                    LOSS = self.loss_fn(x_Net_training[mask], train_target[n_e, :, :])
                else:
                    LOSS = self.loss_fn(x_Net_training, train_target[n_e, :, :])

                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS
                #print(x_Net_training)



            # Average
            MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            MSE_train_dB_epoch[ti] = 10 * torch.log10(MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()


            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward(retain_graph=True)

            #torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=2.0, norm_type=2)


            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", MSE_cv_dB_epoch[ti], "[dB]")

            if (ti > 1):
                d_train = MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]
                d_cv    = MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")

        return [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, nclt=False, rnn=False, IC=None):

        N_T = test_input.size()[0]
        self.MSE_test_linear_arr = torch.empty([N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        if(rnn):
            Model = torch.load(path_results+'best-model_rnn.pt', map_location=dev)
        else:
            Model = torch.load(path_results+'best-model.pt', map_location=dev)

        Model.eval()
        torch.no_grad()

        self.KGain_array = torch.zeros((SysModel.T_test, Model.m, Model.n))
        self.x_out_array = torch.empty(N_T,SysModel.m, SysModel.T_test)
        
        start = time.time()
        for j in range(0, N_T):
            Model.i = 0
            # Unrolling Forward Pass
            if nclt:
                Model.InitSequence(SysModel.m1x_0, SysModel.m2x_0, SysModel.T_test)
            elif IC is None:
                Model.InitSequence(torch.unsqueeze(test_target[j, :, 0], dim=1), SysModel.m2x_0, SysModel.T_test)
            else:
                init_cond = torch.reshape(IC[j, :], SysModel.m1x_0.shape)
                Model.InitSequence(init_cond, SysModel.m2x_0, SysModel.T_test)

            
            y_mdl_tst = test_input[j, :, :]

            x_Net_mdl_tst = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
            
            for t in range(0, SysModel.T_test):
                x_Net_mdl_tst[:,t] = Model(y_mdl_tst[:,t])
            
            if(nclt):
                if x_Net_mdl_tst.size()[0] == 6:
                    mask = torch.tensor([True,False,False,True,False,False])
                else:
                    mask = torch.tensor([True,False,True,False])
                self.MSE_test_linear_arr[j] = loss_fn(x_Net_mdl_tst[mask], test_target[j, :, :]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_Net_mdl_tst, test_target[j, :, :]).item()
            self.x_out_array[j,:,:] = x_Net_mdl_tst

            try:
                self.KGain_array = torch.add(Model.KGain_array, self.KGain_array)
                self.KGain_array /= N_T
            except:
                self.KGain_array = None

        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, self.KGain_array, self.x_out_array, t]

    def NNTest_evol(self, SysModel, test_input, test_target, path_results):

        N_T = test_input.size()[0]
        MSE_test_linear_arr = torch.empty(N_T,SysModel.m, SysModel.T_test)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='none')

        Model = torch.load(path_results+'best-model.pt', map_location=dev)

        Model.eval()
        torch.no_grad()

        trace_avg = torch.zeros([SysModel.T_test])

        for j in range(0, N_T):
            Model.i = 0
            # Unrolling Forward Pass
            
            Model.InitSequence(torch.unsqueeze(test_target[j, :, 0], dim=1), SysModel.m2x_0, SysModel.T_test)
                      
            y_mdl_tst = test_input[j, :, :]

            x_Net_mdl_tst = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
            
            for t in range(0, SysModel.T_test):
                x_Net_mdl_tst[:,t] = Model(y_mdl_tst[:,t])
                     
            MSE_test_linear_arr[j, :, :] = loss_fn(x_Net_mdl_tst, test_target[j, :, :])

        # Average
        MSE_test_avg = torch.mean(MSE_test_linear_arr, [0,1])

        for j in range(0, SysModel.T_test):
            error_covariance = torch.mm((torch.mm((torch.eye(SysModel.m) - Model.KGain_array[j,:,:]),Model.KGain_array[j,:,:])),torch.inverse(torch.eye(SysModel.m) - Model.KGain_array[j,:,:]))
            cov_trace = torch.trace(error_covariance)
            trace_avg[j] = cov_trace

        self.MSE_test_dB_avg = 10 * torch.log10(MSE_test_avg)
        self.trace_dB_avg = 10* torch.log10(trace_avg)

        return [self.MSE_test_dB_avg, self.trace_dB_avg]


    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)