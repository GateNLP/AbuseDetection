from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import math
import random

class Torch_Mini_Batch(object):
    def __init__(self, inputReader, gensimDict, batch_size=32, sentence_length=50, pad_last_batch=False):
        self.inputReader = inputReader
        self.batch_size = batch_size
        self.gensimDict = gensimDict
        self.pad_last_batch = pad_last_batch
        self.sentence_length = sentence_length
        self.num_batch = self._get_num_batch()
        self.reserved_sample = [] # list of reserved sample to fill last batch 
        self.reset()

    def _finalise(self, current_x, current_y):
        x = pad_sequences(current_x, maxlen=self.sentence_length)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(current_y, dtype=torch.long)
        #y = y.view(-1, 1)
        return x, y


    def __iter__(self):
        self.batch_id = 0
        self.inputReader.reset()
        return self

    def __next__(self):
        current_x = []
        current_y = []
        if self.batch_id < self.num_batch:
            while(len(current_x) < self.batch_size):
                try:
                    raw_x, raw_y = next(self.inputReader)
                    doc_idx = self.gensimDict.doc2idx(raw_x)
                    if len(self.reserved_sample) < self.batch_size:
                        self.reserved_sample.append([doc_idx, raw_y])
                    else:
                        r = random.random()
                        if r > 0.5:
                            self.reserved_sample.pop()
                            self.reserved_sample.append([doc_idx, raw_y])
                except StopIteration:
                    if self.pad_last_batch:
                        random.shuffle(self.reserved_sample)
                        doc_idx, raw_y = self.reserved_sample.pop()
                    else:
                        break
                current_x.append(doc_idx)
                current_y.append(raw_y)
            self.batch_id += 1
            return self._finalise(current_x, current_y)
                
        else:
            raise StopIteration
            


    def _get_num_batch(self):
        num_samples = len(self.inputReader)
        num_batch = math.ceil(num_samples/self.batch_size)
        return num_batch
        
    def __len__(self):
        return self.num_batch


    def reset(self):
        self.batch_id = 0
        self.inputReader.reset()

