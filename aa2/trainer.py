import os
import torch
from torch import optim
import torch.nn as nn
import torchtext as tt
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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
        self.epochs = 100
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


    def load_model(self, model_path):
        # Finish this function so that it loads a model and return the appropriate variables
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        hyperparamaters = checkpoint['hyperparamaters']
        loss = checkpoint['loss']
        scores = checkpoint['scores']
        model_name = checkpoint['model_name']

        return epoch, model_state_dict, optimizer_state_dict, hyperparamaters, loss, scores, model_name


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it set up model then trains and saves it.
            
        self.batch_size = hyperparamaters['batch_size']
        self.lr = hyperparamaters['learning_rate']
        nlayers = hyperparamaters['number_layers']
        hidden_dim = hyperparamaters['hidden_size']
        model_name = hyperparamaters['model']
        del hyperparamaters['model']
        
        b = Batcher(train_X, train_y, self.device, batch_size=self.batch_size, max_iter=self.epochs)
        m = model_class(train_X.shape[2], nlayers, hidden_dim, 103)
        m = m.to(self.device)    
        
        

        if hyperparamaters['optimizer'] == "adam":
            optimizer = optim.Adam(m.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(m.parameters(), lr=self.lr)
        loss = nn.L1Loss()

        e = 0
        for split in b:
            m.train()
            tot_loss = 0
            for X, y in split:
                optimizer.zero_grad()
                o = m(X.float(), self.device)
                l = loss(o, y.float()).to(self.device)
                tot_loss += l
                l.backward()
                optimizer.step()          
        
            self.val_X=val_X
            self.val_y=val_y

            m.eval()

            bl = Batcher(self.val_X, self.val_y, self.device, batch_size=self.batch_size, max_iter=1)
            y_true = []
            y_pred = []
            for split in bl:
                for X, y in split:
                    predictions = m(X.float(), self.device)
                    labels = y
                    for i in range(predictions.shape[0]):
                        predict_sent = predictions[i].tolist()
                        label_sent = labels[i].tolist()
                        for j in range(len(predict_sent)):
                            predict_tok = round(predict_sent[j])
                            label_tok = label_sent[j]
                            y_true.append(label_tok)
                            y_pred.append(predict_tok)
            scores = {}
            accuracy = accuracy_score(y_true, y_pred, normalize=True)
            scores['accuracy'] = accuracy
            recall = recall_score(y_true, y_pred, average='weighted')
            scores['recall'] = recall
            precision = precision_score(y_true, y_pred, average='weighted')
            scores['precision'] = precision
            f = f1_score(y_true, y_pred, average='weighted')
            scores['f1_score'] = f
            print("{}: Total loss in epoch {} is: {}      |      F1 score in validation is: {}".format(model_name, e, tot_loss, f))
            e += 1
        self.save_model(e, m, optimizer, tot_loss, scores, hyperparamaters, model_name)

        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, tests it and prints results.
        trained_epochs, model_state_dict, optimizer_state_dict, trained_hyperparamaters, trained_loss, trained_scores, model_name = self.load_model(best_model_path)

        
        batch_size = trained_hyperparamaters['batch_size']
        nlayers = trained_hyperparamaters['number_layers']
        hidden_dim = trained_hyperparamaters['hidden_size']
        lr = trained_hyperparamaters['learning_rate']
        
        m = model_class(test_X.shape[2], nlayers, hidden_dim, 103)
        
        m.load_state_dict(model_state_dict)
        
        m = m.to(self.device) 
        
        m.eval()
        
        b = Batcher(test_X, test_y, self.device, batch_size, max_iter=1)
        
        y_true = []
        y_pred = []
        for split in b:
            for X, y in split:
                predictions = m(X.float(), self.device)
                labels = y
                for i in range(predictions.shape[0]):
                    predict_sent = predictions[i].tolist()
                    label_sent = labels[i].tolist()
                    for j in range(len(predict_sent)):
                        predict_tok = round(predict_sent[j])
                        label_tok = label_sent[j]
                        y_true.append(label_tok)
                        y_pred.append(predict_tok)
        scores = {}
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        scores['accuracy'] = accuracy
        recall = recall_score(y_true, y_pred, average='weighted')
        scores['recall'] = recall
        precision = precision_score(y_true, y_pred, average='weighted')
        scores['precision'] = precision
        f = f1_score(y_true, y_pred, average='weighted')
        scores['f1_score'] = f
        print('model:', model_name, 'accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)
        return scores
