import torch
import torch.nn as nn
import time
import random
from Plot import Plot_extended as Plot

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")

class Pipeline_ERTS:

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
        self.SysModel = ssModel

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
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.9, patience=20)


    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, nclt=False, sequential_training=False, rnn=False, epochs=None, train_IC=None, CV_IC=None):

        self.N_E = train_input.size()[0]
        self.N_CV = cv_input.size()[0]

        MSE_cv_linear_batch = torch.empty([self.N_CV]).to(dev, non_blocking=True)
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)

        MSE_train_linear_batch = torch.empty([self.N_B]).to(dev, non_blocking=True)
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        if epochs is None:
            N = self.N_Epochs
        else:
            N = epochs

        for ti in range(0, N):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            
            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                
                self.model.i = 0
                n_e = random.randint(0, self.N_E - 1)
                if(sequential_training):
                    if(nclt):
                        init_conditions = torch.reshape(cv_input[j,:,0], SysModel.m1x_0.shape)
                    elif CV_IC is None:
                        init_conditions = torch.reshape(cv_target[j,:,0], SysModel.m1x_0.shape)
                    else:
                        init_conditions = SysModel.m1x_0
                else:
                    init_conditions = SysModel.m1x_0

                y_training = train_input[n_e, :, :]
                self.model.InitSequence(init_conditions, SysModel.m2x_0, SysModel.T)
                x_out_training_forward = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
                x_out_training = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                x_out_training[:, SysModel.T-1] = x_out_training_forward[:, SysModel.T-1] # backward smoothing starts from x_T|T 
                self.model.InitBackward(x_out_training[:, SysModel.T-1]) 
                x_out_training[:, SysModel.T-2] = self.model(None, x_out_training_forward[:, SysModel.T-2], x_out_training_forward[:, SysModel.T-1],None)
                for t in range(SysModel.T-3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t], x_out_training_forward[:, t+1],x_out_training[:, t+2])
                
                x_out_train_forward_2 = torch.empty(SysModel.m,SysModel.T).to(dev, non_blocking=True)
                x_out_train_2 = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
                for t in range(0, SysModel.T):
                    x_out_train_forward_2[:, t] = self.model(x_out_training[:, t], None, None, None,pass2=True)
                x_out_train_2[:, SysModel.T-1] = x_out_train_forward_2[:, SysModel.T-1] # backward smoothing starts from x_T|T 
                self.model.InitBackward(x_out_train_2[:, SysModel.T-1]) 
                x_out_train_2[:, SysModel.T-2] = self.model(None, x_out_train_forward_2[:, SysModel.T-2], x_out_train_forward_2[:, SysModel.T-1],None,pass2=True)
                for t in range(SysModel.T-3, -1, -1):
                    x_out_train_2[:, t] = self.model(None, x_out_train_forward_2[:, t], x_out_train_forward_2[:, t+1],x_out_training[:, t+2],pass2=True)          
                x_out_training = x_out_train_2
                # Compute Training Loss
                if(nclt):
                    if x_out_training.size()[0]==6:
                        mask = torch.tensor([True,False,False,True,False,False])
                    else:
                        mask = torch.tensor([True,False,True,False])
                    LOSS = self.loss_fn(x_out_training[mask], train_target[n_e, :, :])
                else:
                    LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :])

                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                for j in range(0, self.N_CV):
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

                    self.model.InitSequence(init_conditions, SysModel.m2x_0, SysModel.T_test)   
                    y_cv = cv_input[j, :, :]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
                    for t in range(0, SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                    x_out_cv[:, SysModel.T_test-1] = x_out_cv_forward[:, SysModel.T_test-1] # backward smoothing starts from x_T|T
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test-1]) 
                    x_out_cv[:, SysModel.T_test-2] = self.model(None, x_out_cv_forward[:, SysModel.T_test-2], x_out_cv_forward[:, SysModel.T_test-1],None)
                    for t in range(SysModel.T_test-3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t+1],x_out_cv[:, t+2])
                    
                    x_out_cv_forward_2 = torch.empty(SysModel.m,SysModel.T).to(dev, non_blocking=True)
                    x_out_cv_2 = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
                    for t in range(0, SysModel.T):
                        x_out_cv_forward_2[:, t] = self.model(x_out_cv[:, t], None, None, None,pass2=True)
                    x_out_cv_2[:, SysModel.T-1] = x_out_cv_forward_2[:, SysModel.T-1] # backward smoothing starts from x_T|T 
                    self.model.InitBackward(x_out_cv_2[:, SysModel.T-1]) 
                    x_out_cv_2[:, SysModel.T-2] = self.model(None, x_out_cv_forward_2[:, SysModel.T-2], x_out_cv_forward_2[:, SysModel.T-1],None,pass2=True)
                    for t in range(SysModel.T-3, -1, -1):
                        x_out_cv_2[:, t] = self.model(None, x_out_cv_forward_2[:, t], x_out_cv_forward_2[:, t+1],x_out_cv[:, t+2],pass2=True)          
                    x_out_cv = x_out_cv_2
                    # Compute Training Loss
                    if(nclt):
                        if x_out_cv.size()[0]==6:
                            mask = torch.tensor([True,False,False,True,False,False])
                        else:
                            mask = torch.tensor([True,False,True,False])
                        MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv[mask], cv_target[j, :, :]).item()
                    else:
                        MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item()

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results + 'best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
            
            
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, nclt=False, rnn=False, IC=None):

        self.N_T = test_input.size()[0]

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        self.model = torch.load(path_results+'best-model.pt', map_location=dev)

        self.model.eval()

        torch.no_grad()

        x_out_array = torch.empty(self.N_T,SysModel.m, SysModel.T_test)
        start = time.time()
        for j in range(0, self.N_T):
            if nclt:
                self.model.InitSequence(SysModel.m1x_0, SysModel.m2x_0, SysModel.T_test)
            elif IC is None:
                self.model.InitSequence(torch.unsqueeze(test_target[j, :, 0], dim=1), SysModel.m2x_0, SysModel.T_test)
            else:
                init_cond = torch.reshape(IC[j, :], SysModel.m1x_0.shape)
                self.model.InitSequence(init_cond, SysModel.m2x_0, SysModel.T_test)

            y_mdl_tst = test_input[j, :, :]

            x_out_test_forward_1 = torch.empty(SysModel.m,SysModel.T_test).to(dev, non_blocking=True)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
            for t in range(0, SysModel.T_test):
                x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
            x_out_test[:, SysModel.T_test-1] = x_out_test_forward_1[:, SysModel.T_test-1] # backward smoothing starts from x_T|T 
            self.model.InitBackward(x_out_test[:, SysModel.T_test-1]) 
            x_out_test[:, SysModel.T_test-2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test-2], x_out_test_forward_1[:, SysModel.T_test-1],None)
            for t in range(SysModel.T_test-3, -1, -1):
                x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t+1],x_out_test[:, t+2])
            
            ########################################################################
            # Second pass
            x_out_test_forward_2 = torch.empty(SysModel.m,SysModel.T_test).to(dev, non_blocking=True)
            x_out_test_2 = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
            for t in range(0, SysModel.T_test):
                x_out_test_forward_2[:, t] = self.model(x_out_test[:, t], None, None, None,pass2=True)
            x_out_test_2[:, SysModel.T_test-1] = x_out_test_forward_2[:, SysModel.T_test-1] # backward smoothing starts from x_T|T 
            self.model.InitBackward(x_out_test_2[:, SysModel.T_test-1]) 
            x_out_test_2[:, SysModel.T_test-2] = self.model(None, x_out_test_forward_2[:, SysModel.T_test-2], x_out_test_forward_2[:, SysModel.T_test-1],None,pass2=True)
            for t in range(SysModel.T_test-3, -1, -1):
                x_out_test_2[:, t] = self.model(None, x_out_test_forward_2[:, t], x_out_test_forward_2[:, t+1],x_out_test[:, t+2],pass2=True)          
            x_out_test = x_out_test_2

            if(nclt):
                if x_out_test.size()[0] == 6:
                    mask = torch.tensor([True,False,False,True,False,False])
                else:
                    mask = torch.tensor([True,False,True,False])
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[mask], test_target[j, :, :]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()
            x_out_array[j,:,:] = x_out_test
        
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_array, t]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, self.N_B, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    def PlotTrain_RTS(self, MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg):
    
        self.Plot = Plot(self.folderName, self.modelName)

        # self.Plot.NNPlot_epochs(self.N_E,self.N_Epochs, self.N_B, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                                # self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, self.MSE_test_linear_arr)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)