# -*- coding: utf-8 -*-
import csv
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim import corpora
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from abuseUlti import CSV_Reader, Torch_Mini_Batch
#from models import SimpleModel as Net
from models import CNNModel as Net
from models import ModelUlti

def csvTextProcess(raw_text):
    if raw_text[0] == '"':
        raw_text = raw_text[1:]
    if raw_text[-1] == '"':
        raw_text = raw_text[:-1]
    tokens = word_tokenize(raw_text)
    
    return tokens

training_file = sys.argv[1]
test_file = sys.argv[2]
raw_x_list = [] #[rawtext, label]
raw_y_list = []


csvReader = CSV_Reader(training_file, label_field=0, text_callback=csvTextProcess, return_text_only=True)
print(next(csvReader))
csvReader.reset()
special_tokens = {'pad': 0}
gemsim_dictionary = corpora.Dictionary(csvReader)
gemsim_dictionary.patch_with_special_tokens(special_tokens)
#print(gemsim_dictionary.token2id)
csvReader = CSV_Reader(training_file, label_field=0, text_callback=csvTextProcess, label_callback=float, return_text_only=False)
embedding_output_dim = 32 #32
sentence_length = 50
vocab_size = len(gemsim_dictionary.token2id)
output_size = 2

mini_batch = Torch_Mini_Batch(csvReader, gemsim_dictionary, pad_last_batch=True)



#x, y = next(mini_batch)
#print(x)
#print(y)
#
net = Net(sentence_length, vocab_size, embedding_output_dim, output_size)
model_ulti = ModelUlti(net)
model_ulti.train_batch(mini_batch, num_epohs=15)

#print(net)
mini_batch.reset()
x, y = next(mini_batch)
out = net(x)
print(out)
#print(out.shape)
#print(y.shape)
#net.zero_grad()  
#loss.backward()
#


testReader = CSV_Reader(test_file, text_callback=csvTextProcess, return_text_only=False)
test_batch = Torch_Mini_Batch(csvReader, gemsim_dictionary)
pred,y = model_ulti.prediction(test_batch)
print(pred)




testReader = CSV_Reader(test_file, text_callback=csvTextProcess, label_field=0, label_callback=float, return_text_only=False)
test_batch = Torch_Mini_Batch(csvReader, gemsim_dictionary)
accuracy = model_ulti.evaluation(test_batch)
print(accuracy)






