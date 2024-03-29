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
#from models import CNNModel as Net
from models import BlstmCNNModel as Net
from models import ModelUlti

parser = OptionParser()
parser.add_option("--configFile", help="xml file input", type=str)
options, arguments = parser.parse_args()
config_file = options.configFile
config = configparser.ConfigParser()
config.read(config_file)

text_callback = None
label_callback = None

if 'Call_Backs' in config:
    sys.path.append(config['Call_Backs']['call_back_function_path'])
    if 'text_callback' in config['Call_Backs']:
        from abuse_callbacks import text_callback
    if 'label_callback' in config['Call_Backs']:
        from abuse_callbacks import label_callback

if 'Train' in config:
    training_file = config['Train']['training_file']
    csv_text_field = int(config['Train']['csv_text_field'])
    csv_label_field = int(config['Train']['csv_label_field'])

    if 'build_dictionary' in config['Train']:
        train_reader = CSV_Reader(training_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=text_callback, label_callback=label_callback, return_text_only=True)
        special_tokens = {'pad': 0, "unk": 1}
        gemsim_dictionary = corpora.Dictionary(train_reader)
        gemsim_dictionary.patch_with_special_tokens(special_tokens)
        gemsim_dictionary.save(config['Train']['build_dictionary'])
        print(gemsim_dictionary.token2id)

    train_reader = CSV_Reader(training_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=text_callback, label_callback=label_callback, return_text_only=False)
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
    valReader = CSV_Reader(validation_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=text_callback, label_callback=label_callback, return_text_only=False)
    valBatchIter = Torch_Mini_Batch(valReader, gemsim_dictionary, pad_last_batch=False) 
    evaluation_score = model_ulti.evaluation(valBatchIter, output_f_measure=True, output_roc_score=True)
    print(evaluation_score)

if 'Test' in config:
    test_file = config['Test']['test_file']
    dictionary_load_path = config['Test']['dictionary_load_path']
    model_load_path = config['Test']['model_load_path']
    csv_text_field = int(config['Test']['csv_text_field'])
    csv_label_field = int(config['Test']['csv_label_field'])

    testReader = CSV_Reader(test_file, delimiter=',', quotechar='"', skip_row=1, text_field=csv_text_field, label_field=csv_label_field, text_callback=text_callback, label_callback=label_callback, return_text_only=False)
    gemsim_dictionary = corpora.Dictionary.load(dictionary_load_path)
    #print(gemsim_dictionary.token2id)
    model_ulti = ModelUlti()
    model_ulti.loadModel(model_load_path)

    testBatchIter = Torch_Mini_Batch(testReader, gemsim_dictionary, pad_last_batch=False)
    evaluation_score = model_ulti.evaluation(testBatchIter, output_f_measure=True, output_roc_score=True)
    print(evaluation_score)
