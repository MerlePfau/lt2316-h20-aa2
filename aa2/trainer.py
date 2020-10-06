import os
import torch
from torch import optim

device = torch.device('cuda:1')


class Batcher:
    def __init__(self, X, y, device, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        self.device = device
        self.batch_size=batch_size
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
        self.epochs = 1
        


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
        b = Batcher(train_X, train_y, device, batch_size=self.batch_size, max_iter=self.epochs)
        self.val_X=val_X
        self.val_y=val_y
        optimizer = optim.SGD(model_class.parameters(), lr=self.lr)
        #self.optimizer=hyperparamaters['optimizer']

        for e in range(self.epochs):
            for split in b:
                tot_loss = 0
                for batch in split:
                    optimizer.zero_grad()
                    o = model_class(batch[0])
                    l = loss(o.reshape(batch_size), batch[1])
                    tot_loss += l
                    l.backward()
                    optimizer.step()
            print("Total loss in epoch {} is {}.".format(epoch, tot_loss))
        self.save_model()
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass
