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



class TextClassResiduleBiLstmCnnSingle(CustomModule):
    def __init__(self, dataset, config={}, maxSentLen=500, kernel_size =[3,4,5], cnn_dim=128, lstm_dim=64, dropout=0.6, bn_momentum=0.1):
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
        self.residule_pool = torch.nn.ModuleDict()
        self.residule_pool_names = []

        self.residule_conv = torch.nn.ModuleDict()
        self.residule_conv_names = []

        for K in kernel_size:
            current_cnn_layer = torch.nn.Conv1d(in_channels=lstm_dim*2, 
                                                out_channels=cnn_dim, 
                                                kernel_size=K, 
                                                padding=int(K/2))
            current_bn = nn.BatchNorm1d(cnn_dim, momentum=bn_momentum)
            if (K % 2) == 0:
                current_residule_pool = nn.Linear(maxSentLen+1, 1)
            else:
                current_residule_pool = nn.Linear(maxSentLen, 1)
            current_residule_conv = nn.Linear(maxSentLen, maxSentLen+1)


            cnn_layername = "cnn_K{}".format(K)
            cnn_bn_layername = "bn_K{}".format(K)
            residule_pool_layername = "poolresidule_K{}".format(K)
            residule_conv_layername = "convresidule_K{}".format(K)


            self.cnn_layers.add_module(cnn_layername, current_cnn_layer)
            self.cnn_bn.add_module(cnn_bn_layername, current_bn)
            self.residule_pool.add_module(residule_pool_layername, current_residule_pool)
            self.residule_conv.add_module(residule_conv_layername, current_residule_conv)
            self.cnn_layers_names.append(cnn_layername)
            self.cnn_bn_names.append(cnn_bn_layername)
            self.residule_pool_names.append(residule_pool_layername)
            self.residule_conv_names.append(residule_conv_layername)
            

            


        self.fc = nn.Linear(cnn_dim*len(kernel_size),self.n_classes)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.residual_lstm = nn.Linear(embedding_dim, lstm_dim*2)

    





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
        residual_embd = self.residual_lstm(embedded)
        lstmed, hidden = self.lstm1(embedded)
        lstmed = lstmed + residual_embd
        residule_lstmed = lstmed
    
        convd = []
        convd_residule = []
        lstmed = lstmed.transpose(1,2)
        residule_lstmed = residule_lstmed.transpose(1,2)

        for i in range (len(self.cnn_layers_names)):
            current_cnn_layer_name = self.cnn_layers_names[i]
            current_cnn_bn_layer_name = self.cnn_bn_names[i]
            current_residule_pool_layer_name = self.residule_pool_names[i] 
            current_residule_conv_layer_name = self.residule_conv_names[i]
            current_conved = F.relu(self.cnn_layers[current_cnn_layer_name](lstmed))
            current_conved = self.cnn_bn[current_cnn_bn_layer_name](current_conved)
            conv_shape = current_conved.shape
            #print(current_conved.shape)
            if current_conved.shape[2] > self.maxSentLen:
                current_residule_lstmed = self.residule_conv[current_residule_conv_layer_name](residule_lstmed)
            else:
                current_residule_lstmed = residule_lstmed

            #print(residule_lstmed.shape)
            current_conved = current_conved + current_residule_lstmed
            residule_current_conved = current_conved
            #print(current_residule_pool_layer_name)
            #print(residule_current_conved.shape)
            residule_current_conved = self.residule_pool[current_residule_pool_layer_name](residule_current_conved).squeeze(2)

            current_conved = F.max_pool1d(current_conved, current_conved.shape[2]).squeeze(2)
            current_conved = current_conved + residule_current_conved
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

