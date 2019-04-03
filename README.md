# AbuseDetection

## Data

### GATE corpus

500 documents, one tweet per doc, partially annotated. "Text" annotations with "Abuse" feature containing "1" or "0" (text).

Class 0: 343  
Class 1: 11  

### Kaggle corpora
"Key" set contains "Text" annotations with "insult" feature containing 1 or 0.

**gate-train** (this also appears in SVN as "gold-standard-abusive-tweets"--don't use that as JPs version is cleaned)

Class 0: 2898  
Class 1: 1049  

**gate-test_with_solutions** (this also appears in SVN as "gold-standard-abusive-tweets-2"--don't use that as JPs version is cleaned)

Class 0: 1954  
Class 1: 693

**gate-impermium_verification_labels**

Class 0: 1158  
Class 1: 1077



## Evaluation

### Gazetteer

#### ICWSM result

Since the approach was done from a word list found online without looking at any data, both training and test set were evaluated in [the extended paper on arXiv](https://arxiv.org/pdf/1804.01498.pdf):

TRAINING:  
Accuracy: 0.78  
Cohen’s Kappa: 0.39   
Precision: 0.62  
Recall: 0.45  
F1: 0.53  

TEST:  
Accuracy: 0.78  
Cohen’s Kappa:  0.36  
Precision: 0.60  
Recall: 0.43  
F1: 0.50  

[ICWSM paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/viewFile/17861/17060)  
Training and test combined, I believe:

Accuracy: 0.78   
Cohen’s Kappa: 0.37   
Precision: 0.61   
Recall: 0.44   
F1: 0.51  

#### ITV result

Some low recall words were removed, such as "kill", in the version used for the ITV work. These words for removal were selected based on the Kaggle training data and discussed as a group, since we didn't uncritically remove words that defied common sense and didn't appear often enough in the Kaggle data to really give us a reading. I evaluated this on Johann's new version of the data, which is cleaned up a little I believe, used text spans from the test data and marked any text containing and "AbuseLookup" as 1 for "insult". I did this manually in the GATE GUI. The scripts are in my personal files but are trivial.

Accuracy: 0.8009    
Cohen’s Kappa: 0.4074    
Scott's Pi: 0.3973    
Precision: 0.7015    
Recall: 0.4170    
F1: 0.5231    

### Learning Framework
Trained on gate-train, Tested on gate-test_with_solutions

Accuracy:	0.8058
Cohen’s Kappa: 0.4566

Five run BLSTMCNN without Shortcut
Accuracy: 80.614

Five run BLSTMCNN with Shortcut(residual) average accuracy:
Accuracy: 81.014


Five run valila CNN with glove 100 dim twitter embed
Accuracy: 82.51

```

Elmo + CNN (see below for model)
First run: Accuracy: 85.95, Cohen's Kappa: 62.92, Pi's Kapaa: 62.90
Second run: Accuracy: 83.94, Cohen's Kappa: 50.84, Pi's Kapaa: 49.62

```

```
ELMO + CNN Model:
module=TextClassELMO_CNNsingle(
  (elmo): Elmo(
    (_elmo_lstm): _ElmoBiLm(
      (_token_embedder): _ElmoCharacterEncoder(
        (char_conv_0): Conv1d(16, 32, kernel_size=(1,), stride=(1,))
        (char_conv_1): Conv1d(16, 32, kernel_size=(2,), stride=(1,))
        (char_conv_2): Conv1d(16, 64, kernel_size=(3,), stride=(1,))
        (char_conv_3): Conv1d(16, 128, kernel_size=(4,), stride=(1,))
        (char_conv_4): Conv1d(16, 256, kernel_size=(5,), stride=(1,))
        (char_conv_5): Conv1d(16, 512, kernel_size=(6,), stride=(1,))
        (char_conv_6): Conv1d(16, 1024, kernel_size=(7,), stride=(1,))
        (_highways): Highway(
          (_layers): ModuleList(
            (0): Linear(in_features=2048, out_features=4096, bias=True)
            (1): Linear(in_features=2048, out_features=4096, bias=True)
          )
        )
        (_projection): Linear(in_features=2048, out_features=512, bias=True)
      )
      (_elmo_lstm): ElmoLstm(
        (forward_layer_0): LstmCellWithProjection(
          (input_linearity): Linear(in_features=512, out_features=16384, bias=False)
          (state_linearity): Linear(in_features=512, out_features=16384, bias=True)
          (state_projection): Linear(in_features=4096, out_features=512, bias=False)
        )
        (backward_layer_0): LstmCellWithProjection(
          (input_linearity): Linear(in_features=512, out_features=16384, bias=False)
          (state_linearity): Linear(in_features=512, out_features=16384, bias=True)
          (state_projection): Linear(in_features=4096, out_features=512, bias=False)
        )
        (forward_layer_1): LstmCellWithProjection(
          (input_linearity): Linear(in_features=512, out_features=16384, bias=False)
          (state_linearity): Linear(in_features=512, out_features=16384, bias=True)
          (state_projection): Linear(in_features=4096, out_features=512, bias=False)
        )
        (backward_layer_1): LstmCellWithProjection(
          (input_linearity): Linear(in_features=512, out_features=16384, bias=False)
          (state_linearity): Linear(in_features=512, out_features=16384, bias=True)
          (state_projection): Linear(in_features=4096, out_features=512, bias=False)
        )
      )
    )
    (_dropout): Dropout(p=0.5)
    (scalar_mix_0): ScalarMix(
      (scalar_parameters): ParameterList(
          (0): Parameter containing: [torch.cuda.FloatTensor of size 1 (GPU 0)]
          (1): Parameter containing: [torch.cuda.FloatTensor of size 1 (GPU 0)]
          (2): Parameter containing: [torch.cuda.FloatTensor of size 1 (GPU 0)]
      )
    )
    (scalar_mix_1): ScalarMix(
      (scalar_parameters): ParameterList(
          (0): Parameter containing: [torch.cuda.FloatTensor of size 1 (GPU 0)]
          (1): Parameter containing: [torch.cuda.FloatTensor of size 1 (GPU 0)]
          (2): Parameter containing: [torch.cuda.FloatTensor of size 1 (GPU 0)]
      )
    )
  )
  (layers): Sequential(
    (layer_cnns): LayerCNN(
      (layers): Sequential(
        (transpose): Transpose4CNN()
        (CNNs): ListModule(
          (modulelist): ModuleList(
            (0): Sequential(
              (conv1d_K3): Conv1d(2048, 100, kernel_size=(3,), stride=(1,), padding=(1,))
              (batchnorm1d_K3): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlin_K3): ReLU()
              (maxpool_K3): MaxFrom1d()
              (dropout_K3): Dropout(p=0.6)
            )
            (1): Sequential(
              (conv1d_K4): Conv1d(2048, 100, kernel_size=(4,), stride=(1,), padding=(2,))
              (batchnorm1d_K4): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlin_K4): ReLU()
              (maxpool_K4): MaxFrom1d()
              (dropout_K4): Dropout(p=0.6)
            )
            (2): Sequential(
              (conv1d_K5): Conv1d(2048, 100, kernel_size=(5,), stride=(1,), padding=(2,))
              (batchnorm1d_K5): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlin_K5): ReLU()
              (maxpool_K5): MaxFrom1d()
              (dropout_K5): Dropout(p=0.6)
            )
          )
        )
        (concat): Concat()
      )
    )
    (linear): Linear(in_features=300, out_features=2, bias=True)
    (logsoftmax): LogSoftmax()
  )
)
optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.015
    weight_decay: 0
)
lossfun=NLLLoss()
```


