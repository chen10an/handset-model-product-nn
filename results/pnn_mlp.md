# Combining Categorical PNN with Numerical MLP

This is a continuation of "Product-based Neural Networks + Other Models."

---
_note: the following implementation is for testing purposes and not an optimal way to combine new code (numerical MLP) with the existing code (PNN1)_

_note 2: only PNN1 has been combined with a numerical MLP, using numerical features with the other models in models.py will not work_

---

## Data and Preprocessing
* using oversampling batch generator: ratio=3 (neg to pos)
  * middle-sized ratio: increase precision while not decreasing true positive rate by too much
* id=2

## SETTINGS
* early stopping happens when eval precision stops improving (it does not consider accuracy)

```py
min_round = 1
num_round = 20
early_stop_round = 5
batch_size = 256

algo = 'pnn1'
ratio = 3

include_num = True
```

_note: PNN1 uses learning rate 0.01 for these tests_

## Numerical Model: Multilayered Perceptron
* 3 hidden layers (same number of nodes in each layer)
* experimenting with different number of nodes: 128, 256, 512 (changed in models.py)

---

## Abbreviations
* PNN1+MLP
  * combined model of categorical PNN1 and numerical Multilayered Perceptron
  * uses 7num + 8cat features selected via f_classif
  * ratio=3
* PNN1_all
  * PNN1 model
  * uses all categorical features
  * ratio=3

---

## Results: PNN1+MLP w/ f_classif features

### MLP nodes: 128, best iteration: 6/19 (0-19)
loss (with l2 norm):0.412021

[PNN1+MLP 128] | train | eval
--- | --- | ---
auc | 0.565362 | 0.785801
accuracy | 0.920031	| 0.904868
precision | 0.009449 | **0.021650**
recall | 0.144884 | 0.409639

train-confusion-matrix:

[PNN1+MLP 128] | predicted - | predicted +
--- | --- | ---
actual - | 428979 | 35327
actual + | 1989 | 337

test-confusion-matrix:

[PNN1+MLP 128] | predicted - | predicted +
--- | --- | ---
actual - | 105323 | 10755
actual + | 343 | 238

train-true-pos-rate: 0.144884	eval-true-pos-rate: 0.409639

### MLP nodes: 256, best iteration: 6/19
loss (with l2 norm):0.382223

[PNN1+MLP 256] | train | eval
--- | --- | ---
auc | 0.533828 | 0.773173
accuracy | 0.908594	accuracy | 0.895259
precision | 0.008411 | **0.020043**
recall | 0.148323	| 0.418244

train-confusion-matrix:

[PNN1+MLP 256] | predicted - | predicted +
--- | --- | ---
actual - | 423634 | 40672
actual + | 1981 | 345

test-confusion-matrix:

[PNN1+MLP 256] | predicted - | predicted +
--- | --- | ---
actual - | 104197 | 11881
actual + | 338 | 243

train-true-pos-rate: 0.148323	eval-true-pos-rate: 0.418244

### MLP nodes: 512, best iteration: 10/19
loss (with l2 norm):0.310854

[PNN1+MLP 512] | train | eval
--- | --- | ---
auc | 0.529898 | 0.705757
accuracy | 0.904526	| 0.893056
precision | 0.008817 | **0.019627**
recall | 0.162941 | 0.418244

train-confusion-matrix:

[PNN1+MLP 512] | predicted - | predicted +
--- | --- | ---
actual - | 421702 | 42604
actual + | 1947 | 379

test-confusion-matrix:

[PNN1+MLP 512] | predicted - | predicted +
--- | --- | ---
actual - | 103940 | 12138
actual + | 338 | 243

train-true-pos-rate: 0.162941	eval-true-pos-rate: 0.418244

### Comparison: PNN1_all, best iteration 3/9 (0-9)

loss (with l2 norm):0.398679

[PNN1 3] | train | eval
--- | --- | ---
auc | 0.854312 | 0.829090
accuracy | 0.920479 | 0.920538
precision | 0.028496 | **0.026691**
recall | 0.451849	| 0.421687

train-confusion-matrix:

[PNN1 3] |predicted - |predicted + |
--- | --- | ---
actual - | 428474 | 35832
actual + | 1275 | 1051

test-confusion-matrix:

[PNN1 3] |predicted - |predicted + |
--- | --- | ---
actual - | 107144 | 8934
actual + | 336 | 245

train-true-pos-rate: 0.451849	eval-true-pos-rate: 0.421687

---

## Discussion
* PNN1+MLP does not perform as well as PNN1_all in terms of precision
  * maybe this is because of the lack of features: 7num + 8cat < all cat
  * **--> experiment with more features**
* PNN1+MLP's eval precision and recall are significantly better than its training precision and recall
  * maybe the testing data coincidentally fits relatively well with the prediction?
  * **--> use another train-test split of the data**
* from epoch to epoch, PNN1+MLP seems to be trying to find a balance between no. FP and no. TP
  * for this unbalanced data:
  * if the model predicts less positives in general: FP decreases significantly, TP also decreases but not as significantly --> increase precision
  * if the model predicts more positives in general: FP increases significantly, TP also increases but not as significantly --> decrease precision
  * so
  * rather than doing the above, this combined model experiments with less FP/ more TP from epoch to epoch to try to find a balance
  * **--> so it seems like this combined model is learning in a reasonable way**
* using more/less nodes in the hidden layer does not seem to affect the performance metrics greatly
  * but using less nodes (128) seems to give a slightly better precision than using more nodes
  * using less nodes also saves computation time
  * **--> will be using 128 as the default number of nodes in each hidden layer**

---

## Data and Preprocessing
* using oversampling batch generator: ratio=3 (neg to pos)
  * middle-sized ratio: increase precision while not decreasing true positive rate by too much
* id=4

## SETTINGS
* early stopping happens when eval precision stops improving (it does not consider accuracy)

```py
min_round = 1
num_round = 20
early_stop_round = 5
batch_size = 256

algo = 'pnn1'
ratio = 3

include_num = True
```

_note: PNN1 uses learning rate 0.01 for these tests_

## Results: PNN1+MLP w/ all features from the feature_selection directory
best iteration: 10/19 (0-19)

loss (with l2 norm):0.397568

[PNN1+MLP 128] | train | eval
--- | --- | ---
auc | 0.579379 | 0.749153
accuracy | 0.919294	| 0.909557
precision | 0.009305 | **0.021501**
recall | 0.144024	| 0.385542

train-confusion-matrix:

[PNN1+MLP 128] | predicted - | predicted +
--- | --- | ---
actual - | 428637 | 35669
actual + | 1991 | 335

test-confusion-matrix:

[PNN1+MLP 128] | predicted - | predicted +
--- | --- | ---
actual - | 105884  | 10194
actual + | 357 | 224

train-true-pos-rate: 0.144024	eval-true-pos-rate: 0.385542
