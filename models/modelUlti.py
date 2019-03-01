import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score


class ModelUlti(object):
    def __init__(self, torchModel=None):
        self.torchModel = torchModel

    def train_batch(self, batchIter, num_epohs=10):
        self.torchModel.train() 
        optimizer = optim.Adam(self.torchModel.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for epoh in range(num_epohs):
            all_loss = []
            print("training epoch "+str(epoh+1))
            for x, y in batchIter:
                #print(y)
                optimizer.zero_grad()
                #self.torchModel.zero_grad()
                current_batch_out = self.torchModel(x)
                #print(current_batch_out)
                #out_pred = torch.max(current_batch_out, 1)[1]
                #print(out_pred)
                #real_y = torch.max(y, 1)[1]
                #print(current_batch_out)
                loss = criterion(current_batch_out, y)
                loss.backward()
                optimizer.step()
                loss_value = loss.data.item()
                all_loss.append(loss_value)
            print(sum(all_loss)/len(all_loss))
        self.torchModel.eval()

    def prediction(self, batchIter):
        all_prediction_score = torch.tensor([], dtype=torch.float)
        all_prediction = torch.tensor([], dtype=torch.long)
        all_true_label = torch.tensor([], dtype=torch.long)
        for x, true_label in batchIter:
            #print(x)
            current_batch_out = F.softmax(self.torchModel(x), dim=1)
            #print(current_batch_out)
            label_prediction = torch.max(current_batch_out, 1)[1]
            #print(current_batch_out.shape)
            #print(current_batch_out[:,1])

            #for i in range(len(label_prediction)):
            #    current_score_idx = label_prediction[i]
            #    score = current_batch_out[i][current_score_idx.data.item()]
            #    print(score)
            all_prediction_score = torch.cat((all_prediction_score, current_batch_out[:,1]))
            all_prediction = torch.cat((all_prediction, label_prediction))
            all_true_label = torch.cat((all_true_label, true_label))
        return all_prediction, all_true_label, all_prediction_score


    def evaluation(self, batchIter, class_ids=[0,1], output_f_measure=False, output_roc_score=False):
        output_dict = {}
        all_prediction, true_label, all_prediction_score = self.prediction(batchIter)
        num_correct = (all_prediction == true_label).float().sum()
        accuracy = num_correct / len(all_prediction)
        if output_f_measure:
            output_dict['f-measure'] = {}
            for class_id in class_ids:
                #print("f measure for class"+str(class_id))
                f_measure_score = self.fMeasure(all_prediction, true_label, class_id)
                output_dict['f-measure']['class '+str(class_id)] = f_measure_score
        if output_roc_score:
            roc_score = self.roc_auc(all_prediction_score.detach().numpy(), true_label.detach().numpy())
            output_dict['roc-auc'] = roc_score
        output_dict['accuracy'] = accuracy
        return output_dict

    def fMeasure(self, all_prediction, true_label, class_id):
        mask = [class_id] * len(all_prediction)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        pred_mask = mask_tensor == all_prediction
        true_mask = mask_tensor == true_label
        pc = 0
        for i in range(len(pred_mask)):
            if pred_mask[i] == 1:
                if true_mask[i] == 1:
                    pc+=1

        rc = 0
        for i in range(len(true_mask)):
            if true_mask[i] == 1:
                if pred_mask[i] == 1:
                    rc+=1
        #print(pc, rc)

        #print(pred_mask.sum())
        #print(true_mask.sum())
        precision = float(pc)/pred_mask.float().sum()
        #print(precision)
        recall = float(rc)/true_mask.float().sum()
        #print(recall)
        f_measure = 2*((precision*recall)/(precision+recall))
        #print(f_measure)
        return precision, recall, f_measure

    def roc_auc(self, all_prediction_score, true_label):
        roc_score = roc_auc_score(true_label, all_prediction_score)
        #print(roc_score)
        return roc_score
        
        

        
    def saveModel(self, save_path):
        torch.save(self.torchModel, save_path)


    def loadModel(self, load_path):
        self.torchModel = torch.load(load_path)
        self.torchModel.eval()












