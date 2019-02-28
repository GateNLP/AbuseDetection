import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ModelUlti(object):
    def __init__(self, torchModel):
        self.torchModel = torchModel

    def train_batch(self, batchIter, num_epohs=10):
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

    def prediction(self, batchIter):
        all_prediction = torch.tensor([], dtype=torch.long)
        all_true_label = torch.tensor([], dtype=torch.long)
        for x, true_label in batchIter:
            current_batch_out = torch.sigmoid(self.torchModel(x))
            label_prediction = torch.max(current_batch_out, 1)[1]
            all_prediction = torch.cat((all_prediction, label_prediction))
            all_true_label = torch.cat((all_true_label, true_label))
        return all_prediction, all_true_label


    def evaluation(self, batchIter, class_ids=[0,1]):
        all_prediction, true_label = self.prediction(batchIter)
        num_correct = (all_prediction == true_label).float().sum()
        accuracy = num_correct / len(all_prediction)
        for class_id in class_ids:
            print("f measure for class"+str(class_id))
            self.fMeasure(all_prediction, true_label, class_id)
        return accuracy

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
        print(pc, rc)

        print(pred_mask.sum())
        print(true_mask.sum())
        precision = float(pc)/pred_mask.float().sum()
        print(precision)
        recall = float(rc)/true_mask.float().sum()
        print(recall)
        f_measure = 2*((precision*recall)/(precision+recall))
        print(f_measure)



        

