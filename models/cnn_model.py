import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, sentence_length, vocab_size, embedding_dim, output_size, kernel_dim=32, kernel_sizes=(3, 4, 5), dropout=0.2):
        super(CNNModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv2 = nn.Conv2d(1, kernel_dim, (2,embedding_dim))
        self.pool2 = nn.MaxPool1d(sentence_length-1)

        self.conv3 = nn.Conv2d(1, kernel_dim, (3,embedding_dim))
        self.pool3 = nn.MaxPool1d(sentence_length-2)

        self.conv4 = nn.Conv2d(1, kernel_dim, (4,embedding_dim))
        self.pool4 = nn.MaxPool1d(sentence_length-3)

        self.fc_hidden1 = nn.Linear(kernel_dim*3, 32)

        #self.fc = nn.Linear(kernel_dim*3,output_size)
        self.fc = nn.Linear(32,output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print("checking shapes")
        #print(x.shape) # [batch size, sentence length]
        embedded = self.embedding(x)
        #print(x.shape) # [batch size, sentence length, embedding dim]
        embedded = embedded.unsqueeze(1) # add 1 channel to cnn
        #print(x.shape) # [batch size, channle, sentence length, embedding dim]
        conved_2 = F.relu(self.conv2(embedded)).squeeze(3) # conv over embedding size, left 1 in last
        conved_3 = F.relu(self.conv3(embedded)).squeeze(3)
        conved_4 = F.relu(self.conv4(embedded)).squeeze(3) 

        pooled_2 = self.pool2(conved_2).squeeze(2)
        pooled_3 = self.pool3(conved_3).squeeze(2)
        pooled_4 = self.pool4(conved_4).squeeze(2)
        
        #print(pooled_3.shape)
        #concat = torch.cat((pooled_2,pooled_3,pooled_4), dim=1)
        concat = torch.cat((pooled_2, pooled_3, pooled_4), dim=1)
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

    
