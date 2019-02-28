import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, sentence_length, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(SimpleModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv2 = nn.Conv2d(1, kernel_dim, (2,embedding_dim))
        self.pool2 = nn.MaxPool1d(sentence_length-1)
        self.fc = nn.Linear(kernel_dim,output_size)

    def forward(self, x):
        #print("checking shapes")
        #print(x.shape) # [batch size, sentence length]
        x = self.embedding(x)
        #print(x.shape) # [batch size, sentence length, embedding dim]
        x = x.unsqueeze(1) # add 1 channel to cnn
        #print(x.shape) # [batch size, channle, sentence length, embedding dim]
        x = F.relu(self.conv2(x)) # conv over embedding size, left 1 in last
        #print(x.shape) #[batch size, cnn output channel, sentence_length-kernel_size+1, 1]
        x = x.squeeze(3) # remove last dim
        #print(x.shape) # [batch size, cnn output channel, sentence_length-kernel_size+1]
        x = self.pool2(x) # max over time pooling, left 1 in last
        #print(x.shape) # [batch size, cnn output channel, 1]
        x = x.squeeze(2) # remove last dim
        #print(x.shape) # [batch size, cnn output channel]
        x = x.view(-1, self.num_flat_features(x)) # flatten
        #print(x.shape) # [batch size, cnn output channel]
        x = F.sigmoid(self.fc(x))
        #x = self.fc(x)
        #print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
