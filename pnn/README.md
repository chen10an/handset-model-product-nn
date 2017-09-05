# pnn
Telenor Handset Model with Product-based Neural Networks

## Paper
[_Product-based Neural Networks for User Response Prediction_][paper]

### Implementation
This paper has a [github repo][repo] with implementations of models discussed in the paper. Here are the files with my modifications (the original files have no comments, the comments in the modified files describe my modifications):
* [main.py][main.py]
* [models.py][models.py]
* [utils.py][utils.py]

[paper]: https://arxiv.org/pdf/1611.00144.pdf
[repo]: https://github.com/Atomu2014/product-nets
[main.py]: main.py
[models.py]: product_nets_master/python/models.py
[utils.py]: product_nets_master/python/utils.py

---

### Input for main.py
* sX: scipy.sparse.csr.csr_matrix, (466632, 756)
* y: numpy.ndarray, 466632
* sX_test: scipy.sparse.csr.csr_matrix, (116659, 756)
* y_test: numpy.ndarray, 116659
```py
train_data = sX, y
test_data = sX_test, y_test
```

---

## Using OversamplingBatchGenerator
```py
gen = handset_model.OverSamplingBatchGenerator(data_train_dict, batch_size=batch_size, r=1)
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
