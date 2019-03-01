import torch
import torch.nn as nn
import torch.nn.functional as F


class BlstmCNNModel(nn.Module):
    def __init__(self, sentence_length, vocab_size, embedding_dim, output_size, kernel_dim=32, lstm_dim=32, dropout=0.2, bn_momentum=0.2):
        super(BlstmCNNModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.conv2 = nn.Conv2d(1, kernel_dim, (2,lstm_dim*2))
        self.conv2_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)
        self.pool2 = nn.MaxPool1d(sentence_length-1)

        self.conv3 = nn.Conv2d(1, kernel_dim, (3,lstm_dim*2))
        self.conv3_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)
        self.pool3 = nn.MaxPool1d(sentence_length-2)

        self.conv4 = nn.Conv2d(1, kernel_dim, (4,lstm_dim*2))
        self.conv4_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)
        self.pool4 = nn.MaxPool1d(sentence_length-3)

        self.conv5 = nn.Conv2d(1, kernel_dim, (4,lstm_dim*2))
        self.conv5_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)
        self.pool5 = nn.MaxPool1d(sentence_length-3)

        self.conv6 = nn.Conv2d(1, kernel_dim, (4,lstm_dim*2))
        self.conv6_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)
        self.pool6 = nn.MaxPool1d(sentence_length-3)



        self.fc_hidden1 = nn.Linear(kernel_dim*5, 32)

        #self.fc = nn.Linear(kernel_dim*3,output_size)
        self.fc = nn.Linear(32,output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print("checking shapes")
        #print(x.shape) # [batch size, sentence length]
        embedded = self.embedding(x)
        #print(embedded.shape)
        #print(x.shape) # [batch size, sentence length, embedding dim]
        lstmed, hidden = self.lstm1(embedded)
        #print(lstmed.shape)
        lstmed = lstmed.unsqueeze(1) # add 1 channel to cnn
        #print(lstmed.shape)
        #print(x.shape) # [batch size, channle, sentence length, embedding dim]
        conved_2 = F.relu(self.conv2(lstmed)).squeeze(3) # conv over embedding size, left 1 in last
        conved_2 = self.conv2_bn(conved_2)
        conved_3 = F.relu(self.conv3(lstmed)).squeeze(3)
        conved_3 = self.conv3_bn(conved_3)
        conved_4 = F.relu(self.conv4(lstmed)).squeeze(3) 
        conved_4 = self.conv4_bn(conved_4)
        conved_5 = F.relu(self.conv5(lstmed)).squeeze(3)
        conved_5 = self.conv5_bn(conved_5)
        conved_6 = F.relu(self.conv6(lstmed)).squeeze(3)
        conved_6 = self.conv6_bn(conved_6)




        pooled_2 = self.pool2(conved_2).squeeze(2)
        pooled_3 = self.pool3(conved_3).squeeze(2)
        pooled_4 = self.pool4(conved_4).squeeze(2)
        pooled_5 = self.pool5(conved_5).squeeze(2)
        pooled_6 = self.pool6(conved_6).squeeze(2)

        #print(pooled_3.shape)
        #concat = torch.cat((pooled_2,pooled_3,pooled_4), dim=1)
        concat = torch.cat((pooled_2, pooled_3, pooled_4, pooled_5, pooled_6), dim=1)
        #print(concat.shape)
        #concat = self.dropout(concat)

        #print(x.shape) #[batch size, cnn output channel, sentence_length-kernel_size+1, 1]
        #x = x.squeeze(3) # remove last dim
        #print(x.shape) # [batch size, cnn output channel, sentence_length-kernel_size+1]
        #x = self.pool2(x) # max over time pooling, left 1 in last
        #print(x.shape) # [batch size, cnn output channel, 1]
        #x = x.squeeze(2) # remove last dim
        #print(x.shape) # [batch size, cnn output channel]
        #x = x.view(-1, self.num_flat_features(x)) # flatten
        #print(x.shape) # [batch size, cnn output channel]
        hidden1 = torch.sigmoid(self.fc_hidden1(concat))
        out = self.fc(hidden1)
        #out = F.sigmoid(self.fc(concat))
        #x = self.fc(x)
        #print(x.shape)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
