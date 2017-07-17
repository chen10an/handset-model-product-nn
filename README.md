# handset-model-product-nn
Telenor Handset Model w/ Product-based Neural Networks

## Paper
[_Product-based Neural Networks for User Response Prediction_][paper]
![PNN Diagram](quiver-image-url/B06EF60B39897EE320BCE26CFB1F498E.jpg =500x)

### Implementation
This paper has a [github repo][repo] with implementations of models discussed in the paper. This handset-model-product-nn repo includes the paper's files with my modifications (the original files have no comments, the comments in the modified files describe my modifications).

_note: models.py also includes implementations of other types of models (for comparison)_

[paper]: https://arxiv.org/pdf/1611.00144.pdf
[repo]: https://github.com/Atomu2014/product-nets

---
## Data and Preprocessing (from handset_model_copy4.py)
* all _categorical_ features from handset_data_train_wo_X.csv
* one-hot (encoded as 0/1)
* using oversampling batch generator
* reproduce results: split id=2

### Input for main.py using split 2 (from [utils.py][utils])
* SX_TRAIN: scipy.sparse.csr.csr_matrix, (466632, 756)
* Y_TRAIN: numpy.ndarray, 466632
* SX_TEST: scipy.sparse.csr.csr_matrix, (116659, 756)
* Y_TEST: numpy.ndarray, 116659

```py
train_data = utils.SX_TRAIN, utils.Y_TRAIN
test_data = utils.SX_TEST, utils.Y_TEST
```
[utils]: https://github.com/chen10an/handset-model-product-nn/blob/master/product_nets_master/python/utils.py

## Possible Settings in main.py
```py
min_round = 1
num_round = 10
early_stop_round = 3
batch_size = 256

algo = 'pnn1'
ratio = 1  # neg to pos ratio for OversamplingBatchGenerator
```

---
## Using OversamplingBatchGenerator (from handset_model_copy4.py) in main.py
```py
gen = handset_model_copy4.OverSamplingBatchGenerator(data_train_dict, batch_size=batch_size, r=1)
```
Replace 
```py
X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
``` 
with
```py
n = next(gen.generator())
n_cat = n[0][1]

if algo in {'fnn', 'ccpm', 'pnn1', 'pnn2'}:
    fields = utils.split_data_gen(n_cat)  # slight modification of utils.split_data
    
    X_i = []
    for f in fields:
        w = np.where(f==1)
        indices = [[w[0][i], w[1][i]] for i in range(len(w[0]))]

        indices = np.array(indices, dtype='int32')
        values = np.array([1 for i in range(len(indices))])
        shape = f.shape
        X_i.append((indices, values, shape))
else:
    w = np.where(n_cat==1)
    indices = [[w[0][i], w[1][i]] for i in range(len(w[0]))]

    indices = np.array(indices, dtype='int32')
    values = np.array([1 for i in range(len(indices))])
    shape = n_cat.shape
    X_i = (indices, values, shape)

y_i = np.reshape(n[1], -1).astype(int)
```
inside the j loop in train(model)
