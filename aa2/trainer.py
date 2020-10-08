import os
import torch
from torch import optim
import torch.nn as nn
import torchtext as tt
from torch.autograd import Variable
import numpy as np

class Batcher:
    def __init__(self, X, y, device, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.size()[0], device=self.device)
        permX = self.X[permutation]
        permy = self.y[permutation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)
        
        self.curr_iter += 1
        return zip(splitX, splity)
    
    
class Trainer:

    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)
        self.epochs = 50
        self.device = torch.device('cuda:3')
        
        
    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, model, model_name):
        # Finish this function so that it loads a model and return the appropriate variables
        self.model = model
        self.model.load_state_dict(torch.load(os.path.join(self.dump_folder, model_name + ".pt")))
        self.model.eval()
        pass


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it set up model then trains and saves it.
            
        self.batch_size = hyperparamaters['batch_size']
        self.lr = hyperparamaters['learning_rate']
        nlayers = hyperparamaters['number_layers']
        b = Batcher(train_X, train_y, self.device, batch_size=self.batch_size, max_iter=self.epochs)
        m = model_class(train_X.shape[2], nlayers, 1000, 103)
        m = m.to(self.device)    
        
        #m.train()
        self.val_X=val_X
        self.val_y=val_y
        if hyperparamaters['optimizer'] == "adam":
            optimizer = optim.Adam(m.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(m.parameters(), lr=self.lr)
        loss = nn.L1Loss()

        e = 0
        for split in b:
            tot_loss = 0
            for X, y in split:
                optimizer.zero_grad()
                o = m(X.float())
                l = loss(o, y.float())
                tot_loss += l
                l.backward()
                optimizer.step()
            print("Total loss in epoch {} is {}.".format(e, tot_loss))
            e += 1
        
        accuracy = 0
        recall = 0
        scores = [accuracy, recall]
#        self.save_model(e, m, optimizer, loss, scores, hyperparamaters, 'model_1')

#         with torch.no_grad():
#             tag_scores = m(train_X.float())
#             print('output dim:', tag_scores.shape)
#             print('scores', tag_scores)
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, tests it and prints results.
        pass
