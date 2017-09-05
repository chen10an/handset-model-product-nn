# Benchmarks

## Data and Preprocessing
* id=6 (COLS_TERM)

## Model from handset_model_current.py
best iteration in terms of val_precision (early stop in terms of val_loss):

Epoch 9/10
3625/3627 [============================>.] - ETA: 0s - loss: 0.2901 - acc: 0.8776 - precision: 0.8375 - recall: 0.9379 - fmeasure: 0.8846 - gmean: 0.8861Epoch 00008: val_acc improved from 0.82000 to 0.82657, saving model to handset_weights.best.hdf5
3627/3627 [==============================] - 49s - loss: 0.2901 - acc: 0.8776 - precision: 0.8375 - recall: 0.9379 - fmeasure: 0.8846 - gmean: 0.8862 - val_loss: 0.2955 - val_acc: 0.8266 - **val_precision: 0.0172** - val_recall: 0.4290 - val_fmeasure: 0.0327 - val_gmean: 0.0831

---

## Data and Preprocessing
* id=2 (f_classif)

## Model from handset_model_current.py
best iteration in terms of val_precision (early stop in terms of val_loss):

Epoch 10/10
3623/3627 [============================>.] - ETA: 0s - loss: 0.2726 - acc: 0.8853 - precision: 0.8424 - recall: 0.9487 - fmeasure: 0.8921 - gmean: 0.8938Epoch 00009: val_acc improved from 0.81462 to 0.83791, saving model to handset_weights.best.hdf5
3627/3627 [==============================] - 47s - loss: 0.2726 - acc: 0.8852 - precision: 0.8424 - recall: 0.9487 - fmeasure: 0.8921 - gmean: 0.8938 - val_loss: 0.2806 - val_acc: 0.8379 - val_precision: **0.0189** - val_recall: 0.4351 - val_fmeasure: 0.0358 - val_gmean: 0.0876
