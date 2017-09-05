# Product-based Neural Networks + Other Models

## Data and Preprocessing
* using oversampling batch generator: ratio=1 (neg to pos), testing different ratios
* id=3

## Settings
```py
min_round = 1
num_round = 10
early_stop_round = 3
batch_size = 256
```
_note: PNN1 uses learning_rate=0.1 in these tests_

---

## Results with ratio 1
* metrics are calculated with a 0.5 threshold (np.rint): probability >= 0.5 --> 1 else --> 0
* loss is training loss
* best iteration is in terms of eval precision

### PNN1: best iteration 4/9 (0-9)

[PNN1 1] | train | eval
--- | --- | ---
auc | 0.855859 | 0.825572
accuracy | 0.793469	| 0.792729
precision | 0.017910 | **0.016077**
recall | 0.751075	| 0.674699

loss (with l2 norm):0.467841

7th (last) iter:
train-true-pos-rate: 0.862425	eval-true-pos-rate: 0.802065

### PNN2: iterated to last (9th) iteration

[PNN2 1] | train | eval
--- | --- | ---
auc | 0.858755 | 0.826880
accuracy | 0.776441	| 0.776005
precision | 0.017037 | **0.015621**
recall | 0.773431	| 0.709122

loss (with l2 norm):0.464051

train-confusion-matrix:

[PNN2 1] | predicted - | predicted +
--- | --- | ---
actual - | 360513 | 103793
actual + | 527 | 1799

test-confusion-matrix:

[PNN2 1] |predicted - |predicted + |
--- | --- | ---
actual - | 90116 | 25962
actual + | 169 | 412

train-true-pos-rate: 0.773431	eval-true-pos-rate: 0.709122

### LR: iterated to last iteration (9th)
* precision improvements from iteration to iteration are very small (~0.01%), but these improvements are steady: maybe more iterations can achieve a good result?

[LR 1] | train | eval
--- | --- | ---
auc | 0.839477 | 0.816301
accuracy | 0.733818	| 0.733900
precision | 0.014615 | **0.013946**
recall | 0.788908	| 0.752151

loss (with l2 norm):0.494188

train-confusion-matrix:

[LR 1] |predicted - |predicted + |
--- | --- | ---
actual - | 340588 | 123718
actual + | 491 | 1835

test-confusion-matrix:

[LR 1] |predicted - |predicted + |
--- | --- | ---
actual - | 85179 | 30899
actual + | 144 | 437

train-true-pos-rate: 0.788908	eval-true-pos-rate: 0.752151

### FM: best iteration 1/9

[FM 1] | train | eval
--- | --- | ---
auc | 0.881552 | 0.822284
accuracy | 0.861015	| 0.859334
precision | 0.024473 | **0.019955**
recall | 0.691745	| 0.566265

loss (with l2 norm):0.451215

train-confusion-matrix:

[FM 1] |predicted - |predicted + |
--- | --- | ---
actual - | 400168 | 64138
actual + | 717 | 1609

test-confusion-matrix:

[FM 1] |predicted - |predicted + |
--- | --- | ---
actual - | 99920 | 16158
actual + | 252 | 329

train-true-pos-rate: 0.691745	eval-true-pos-rate: 0.566265

### FNN: best iteration 5

[FNN 1] | train | eval
--- | --- | ---
auc | 0.852493 | 0.827143
accuracy | 0.749130 | 0.749441
precision | 0.015636 | **0.014736**
recall | 0.796217 | 0.748709

loss (with l2 norm):0.471578

train-confusion-matrix:

[FNN 1] |predicted - |predicted + |
--- | --- | ---
actual - | 347716 | 116590
actual + | 474 | 1852

test-confusion-matrix:

[FNN 1] |predicted - |predicted + |
--- | --- | ---
actual - | 86994 | 29084
actual + | 146 | 435

train-true-pos-rate: 0.796217	eval-true-pos-rate: 0.748709

### CCPM
* can't run this algo with all cat features, uses too much memory
* ran this with id=2 cat features
  * model was unable to predict any positive labels (fp and tp = 0)
  * probably due to lack of features/ data

---

## Testing with different ratios
* PNN1 and FM performed best in terms of eval precision
* there are still too many false positives
  * aka the model is predicting _too many_ positives
  * maybe this is due to the 1:1 (neg:pos) batches when the actual data is extremely unbalanced (neg >> pos)

**so I will test PNN1 and FM with a larger neg to pos ratio**

---

## Results with ratio 3
### PNN1: best iteration 3/9

[PNN1 3] | train | eval
--- | --- | ---
auc | 0.854312 | 0.829090
accuracy | 0.920479 | 0.920538
precision | 0.028496 | **0.026691**
recall | 0.451849	| 0.421687

loss (with l2 norm):0.398679

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

### FM: best iteration 1/9

[FM 3] | train | eval
--- | --- | ---
auc | 0.869354 | 0.826937
accuracy | 0.937143 | 0.935556
precision | 0.035118 | **0.029312**
recall | 0.438521 | 0.371773

loss (with l2 norm):0.393296

train-confusion-matrix:

[FM 3] |predicted - |predicted + |
--- | --- | ---
actual - | 436281 | 28025
actual + | 1306 | 1020

test-confusion-matrix:

[FM 3] |predicted - |predicted + |
--- | --- | ---
actual - | 108925 | 7153
actual + | 365 | 216

train-true-pos-rate: 0.438521	eval-true-pos-rate: 0.371773

---

## Results with ratio 5
### PNN1: best iteration 1/9

[PNN1 5] | train | eval
--- | --- | ---
auc | 0.846192 | 0.822441
accuracy | 0.971714	| 0.971978
precision | 0.042575 | **0.042857**
recall | 0.217541	| 0.216867

loss (with l2 norm):0.335066

train-confusion-matrix:

[PNN1 5] |predicted - |predicted + |
--- | --- | ---
actual - | 452927 | 11379
actual + | 1820 | 506

test-confusion-matrix:

[PNN1 5] |predicted - |predicted + |
--- | --- | ---
actual - | 113264 | 2814
actual + | 455 | 126

train-true-pos-rate: 0.217541	eval-true-pos-rate: 0.216867

### FM best iteration 2/9

[FM 5] | train | eval
--- | --- | ---
auc | 0.872484 | 0.825481
accuracy | 0.972479	| 0.972227
precision | 0.047893 | **0.039168**
recall | 0.239467 | 0.194492

loss (with l2 norm):0.315098

train-confusion-matrix:

[FM 5] |predicted - |predicted + |
--- | --- | ---
actual - | 453233 | 11073
actual + | 1769 | 557

test-confusion-matrix:

[FM 5] |predicted - |predicted + |
--- | --- | ---
actual - | 113306 | 2772
actual + | 468 | 113

train-true-pos-rate: 0.239467	eval-true-pos-rate: 0.194492

---

## Discussion
* increasing the ratio has led to:
  * inc precision
  * inc accuracy
  * AND
  * dec true positive rate
* so increasing the ratio does not lead to more true positives, but it leads to _much_ less false positives, thus increasing the precision ratio
* now that the model is predicting less wrong, **how can it predict more right?**

## Next Steps
* combine the categorical model with a numerical nn
  * according to the paper and the results above, PNN1 performs relatively well --> will be the only categorical model used for further experimentation
