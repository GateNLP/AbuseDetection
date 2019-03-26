from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)



class TextClassBiLstmCnnSimpleSingle(CustomModule):
    def __init__(self, dataset, config={}, maxSentLen=500, kernel_size =[3,4,5], cnn_dim=64, lstm_dim=128, dropout=0.6, bn_momentum=0.1):
        super().__init__(config=config)
        #super(TextClassBiLstmCnnSingle, self).__init__()
        self.maxSentLen=maxSentLen
        self.n_classes = dataset.get_info()["nClasses"]
        self.kernel_size = kernel_size

        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        vocab_size = vocab.n
        logger.debug("Initializing module TextClassCNNsingle for classes: %s and vocab %s" %
                     (self.n_classes, vocab_size, ))   

        self.embedding = EmbeddingsModule(vocab)
        embedding_dim = self.embedding.emb_dims
        self.lstm1 = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True, dropout=dropout)

        self.cnn_layers = torch.nn.ModuleDict()
        self.cnn_layers_names = []
        self.cnn_bn = torch.nn.ModuleDict()
        self.cnn_bn_names= []

        for K in kernel_size:
            cnn_layername = "cnn_K{}".format(K)
            current_cnn_layer = torch.nn.Conv1d(in_channels=lstm_dim*2, 
                                                out_channels=cnn_dim, 
                                                kernel_size=K, 
                                                padding=int(K/2))
            cnn_bn_layername = "bn_K{}".format(K)
            current_bn = nn.BatchNorm1d(cnn_dim, momentum=bn_momentum)

            self.cnn_layers.add_module(cnn_layername, current_cnn_layer)
            self.cnn_bn.add_module(cnn_bn_layername, current_bn)
            self.cnn_layers_names.append(cnn_layername)
            self.cnn_bn_names.append(cnn_bn_layername)

        #self.fc_hidden1 = nn.Linear(kernel_dim*3, 16)

        self.fc = nn.Linear(cnn_dim*len(kernel_size),self.n_classes)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)



    def forward(self, batch):
        batch = torch.LongTensor(batch[0])
        sent_len = batch.shape[1]
        batchsize = batch.shape[0] 
        if self.maxSentLen:
            if sent_len > self.maxSentLen:
                batch = batch[:,:self.maxSentLen]
            elif sent_len < self.maxSentLen:
                zero_pad = torch.zeros(batchsize, self.maxSentLen-sent_len, dtype=torch.long)
                batch = torch.cat((batch, zero_pad),dim=1)

        if self.on_cuda():
            batch.cuda()
            for modules in self.cnn_layers:
                self.cnn_layers[modules].cuda()
            for modules in self.cnn_bn:
                self.cnn_bn[modules].cuda()
            self.lstm1.cuda()

        embedded = self.embedding(batch)
        lstmed, hidden = self.lstm1(embedded)
    
        convd = []
        lstmed = lstmed.transpose(1,2)
        for i in range (len(self.cnn_layers_names)):
            current_cnn_layer_name = self.cnn_layers_names[i]
            current_cnn_bn_layer_name = self.cnn_bn_names[i]
            current_conved = F.relu(self.cnn_layers[current_cnn_layer_name](lstmed))
            current_conved = self.cnn_bn[current_cnn_bn_layer_name](current_conved)
            current_conved = F.max_pool1d(current_conved, current_conved.shape[2]).squeeze(2)
            convd.append(current_conved)

        concat = torch.cat(convd, dim=1)
        concat = self.dropout(concat)
        out = self.fc(concat)
        out = self.logsoftmax(out)
        return out


    def get_lossfunction(self, config={}):
        # IMPORTANT: for the target indices, we use -1 for padding by default!
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9, weight_decay=0.05)
        optimizer = torch.optim.Adam(parms, lr=0.015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

