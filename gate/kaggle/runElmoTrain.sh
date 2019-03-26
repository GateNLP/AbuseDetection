#!/bin/bash
python FileJsonPyTorch/gate-lf-pytorch-json/train.py ~/data/abuseDetection/AbuseDetection/gate/kaggle/crvd.meta.json ~/data/abuseDetection/AbuseDetection/gate/kaggle/Test --module TextClassELMO_CNNsingle --elmo /export/data/models/elmo/ --cuda True --es_metric accuracy
