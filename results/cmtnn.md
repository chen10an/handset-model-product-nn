# Complementary Neural networks

## Data and Preprocessing
* id=2

## Model from handset_model_current.py
best iteration in terms of val_precision (early stop in terms of val_loss):

Epoch 10/10
3625/3627 [============================>.] - ETA: 0s - loss: 0.2693 - acc: 0.8863 - precision: 0.8435 - recall: 0.9497 - fmeasure: 0.8931 - gmean: 0.8949Epoch 00009: val_acc improved from 0.80583 to 0.81775, saving model to handset_weights.best.hdf5
3627/3627 [==============================] - 56s - loss: 0.2693 - acc: 0.8863 - precision: 0.8435 - recall: 0.9497 - fmeasure: 0.8931 - gmean: 0.8949 - val_loss: 0.3165 - val_acc: 0.8178 - val_precision: **0.0172** - val_recall: 0.4400 - val_fmeasure: 0.0326 - val_gmean: 0.0837

## Same model w/ cmtnn and 0.5 threshold
* truth prediction probability > threshold --> 1 else 0
* falsity prediction probability < threshold --> 1 else 0
* 0.5 threshold makes this comparable to the regular truth model which uses 0.5 by default

[CMTNN] | train | eval
--- | --- | ---
auc | 0.919812 | 0.708753
accuracy | 0.848510	| 0.844710
precision | 0.031612 | **0.018243**
recall | 0.991831	| 0.571429

train-confusion-matrix:

[CMTNN] | predicted - | predicted +
--- | --- | ---
actual - | 393635 | 70671
actual + | 19 | 2307

test-confusion-matrix:

[CMTNN] | predicted - | predicted +
--- | --- | ---
actual - | 98211 | 17867
actual + | 249 | 332

train-true-pos-rate: 0.991831	eval-true-pos-rate: 0.571429
