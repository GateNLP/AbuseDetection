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

[ICWSM paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/viewFile/17861/17060)  
Trained on "gold-standard-abusive-tweets" and tested on "gold-standard-abusive-tweets-2":

Accuracy: 0.78   
Cohen’s Kappa: 0.37  
Precision: 0.61  
Recall: 0.44  
F1: 0.51

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
