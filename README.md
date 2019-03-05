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

Existing gazetteer-based abuse classification performance is 0.78 (Cohen’s Kappa: 0.37), with a precision of 0.61, a recall of 0.44 and an F1 0.51 on the Kaggle data. From [ICWSM paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/viewFile/17861/17060). Though note that the gazetteer approach wasn't constructed from the Kaggle data, which is more US English.


Intergrated Learning framework blstmcnn model have 0.8446969696969697 accuracy on 5% hold out validation
