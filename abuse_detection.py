# -*- coding: utf-8 -*-
import csv
import sys
from optparse import OptionParser
import configparser

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


parser = OptionParser()
parser.add_option("--configFile", help="xml file input", type=str)
options, arguments = parser.parse_args()
config_file = options.configFile
config = configparser.ConfigParser()
config.read(config_file)

if 'Train' in config:
    training_file = config['Train']['training_file']
    csv_text_field = int(config['Train']['csv_text_field'])
    csv_label_field = int(config['Train']['csv_label_field'])

    if 'build_dictionary' in config['Train']:
        train_reader = CSV_Reader(training_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=csvTextProcess, label_callback=float, return_text_only=True)
        special_tokens = {'pad': 0, "unk": 1}
        gemsim_dictionary = corpora.Dictionary(train_reader)
        gemsim_dictionary.patch_with_special_tokens(special_tokens)
        gemsim_dictionary.save(config['Train']['build_dictionary'])

    train_reader = CSV_Reader(training_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=csvTextProcess, label_callback=float, return_text_only=False)
    word_embedding_dim = int(config['Train']['word_embedding_dim'])
    max_sentence_length = int(config['Train']['max_sentence_length'])
    vocab_size = len(gemsim_dictionary.token2id) + 1
    print(vocab_size)
    output_dim = int(config['Train']['output_dim'])
    net = Net(max_sentence_length, vocab_size, word_embedding_dim, output_dim)
    batchIter = Torch_Mini_Batch(train_reader, gemsim_dictionary, pad_last_batch=True)
    model_ulti = ModelUlti(torchModel=net)
    num_epochs = int(config['Train']['num_epochs'])
    model_ulti.train_batch(batchIter, num_epohs=num_epochs)
    model_save_path = config['Train']['model_save_path']
    model_ulti.saveModel(model_save_path)

if 'Validation' in config:
    validation_file = config['Validation']['validation_file']
    valReader = CSV_Reader(validation_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=csvTextProcess, label_callback=float, return_text_only=False)
    valBatchIter = Torch_Mini_Batch(valReader, gemsim_dictionary, pad_last_batch=False) 
    accuracy = model_ulti.evaluation(valBatchIter)
    print(accuracy)

if 'Test' in config:
    test_file = config['Test']['test_file']
    dictionary_load_path = config['Test']['dictionary_load_path']
    model_load_path = config['Test']['model_load_path']
    csv_text_field = int(config['Test']['csv_text_field'])
    csv_label_field = int(config['Test']['csv_label_field'])

    testReader = CSV_Reader(test_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=csvTextProcess, label_callback=float, return_text_only=False)
    gemsim_dictionary = corpora.Dictionary.load(dictionary_load_path)
    model_ulti = ModelUlti()
    model_ulti.loadModel(model_load_path)

    testBatchIter = Torch_Mini_Batch(testReader, gemsim_dictionary, pad_last_batch=False)
    accuracy = model_ulti.evaluation(testBatchIter)
    print(accuracy)




    





#training_file = sys.argv[1]
#test_file = sys.argv[2]
#raw_x_list = [] #[rawtext, label]
#raw_y_list = []
#
#
#csvReader = CSV_Reader(training_file, label_field=0, text_callback=csvTextProcess, return_text_only=True)
#print(next(csvReader))
#csvReader.reset()
#special_tokens = {'pad': 0, "unk": 1}
#gemsim_dictionary = corpora.Dictionary(csvReader)
#gemsim_dictionary.patch_with_special_tokens(special_tokens)
##print(gemsim_dictionary.token2id)
#csvReader = CSV_Reader(training_file, label_field=0, text_callback=csvTextProcess, label_callback=float, return_text_only=False)
#embedding_output_dim = 32 #32
#sentence_length = 50
#vocab_size = len(gemsim_dictionary.token2id)
#output_size = 2
#
#mini_batch = Torch_Mini_Batch(csvReader, gemsim_dictionary, pad_last_batch=True)
#
#
#
##x, y = next(mini_batch)
##print(x)
##print(y)
##
#net = Net(sentence_length, vocab_size, embedding_output_dim, output_size)
#model_ulti = ModelUlti(net)
#model_ulti.train_batch(mini_batch, num_epohs=15)
#
##print(net)
#mini_batch.reset()
#x, y = next(mini_batch)
#out = net(x)
#print(out)
##print(out.shape)
##print(y.shape)
##net.zero_grad()  
##loss.backward()
##
#
#
#testReader = CSV_Reader(test_file, text_callback=csvTextProcess, return_text_only=False)
#test_batch = Torch_Mini_Batch(csvReader, gemsim_dictionary)
#pred,y = model_ulti.prediction(test_batch)
#print(pred)
#
#
#
#
#testReader = CSV_Reader(test_file, text_callback=csvTextProcess, label_field=0, label_callback=float, return_text_only=False)
#test_batch = Torch_Mini_Batch(csvReader, gemsim_dictionary)
#accuracy = model_ulti.evaluation(test_batch)
#print(accuracy)
#





