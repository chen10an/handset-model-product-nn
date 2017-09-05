import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

import argparse
import pandas as pd
from scipy import sparse


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('main.py'))))
import handset_model_current as handset_model

DTYPE = tf.float32

# FIELD_SIZES = [0] * 26
# with open('../data/featindex.txt') as fin:
#     for line in fin:
#         line = line.strip().split(':')
#         if len(line) > 1:
#             f = int(line[0]) - 1
#             FIELD_SIZES[f] += 1
# print('field sizes:', FIELD_SIZES)
# FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
# INPUT_DIM = sum(FIELD_SIZES)

# FIELD_SIZES = [94316, 99781,     6,    23, 34072, 12723]
# FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(6)]
# INPUT_DIM = sum(FIELD_SIZES)

SX_TRAIN = None
Y_TRAIN = None
SX_TEST = None
Y_TEST = None
FIELD_SIZES = None

DATA_TRAIN_DICT = None
DATA_TEST_DICT = None

FIELD_OFFSETS = None
INPUT_DIM = None

# numerical input size
INPUT_DIM_NUM = None

OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3


def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Expected True or False, but got {}".format(string))


def setArgs():
    # parser arguments that are not related to data preprocessing are not used in any way
    # in this code (utils.py, main.py, models.py)
    # e.g. parameters for creating a model (epochs, batch_size, ratio, etc.)

    # but something must be passed to these irrelevant arguments
    # to call handset_model.load_and_preprocess_data(args)

    # minimal preprocessing
    parser = argparse.ArgumentParser()

    # model hyperparameters
    # small number of epochs for experimentation
    parser.add_argument('--epochs', default=10, type=int,
                        help="Nr of epochs. Default is 100", dest="epochs")
    parser.add_argument('--batch_size', default=256, type=int,
                        help="Batch size. Default is 32", dest="batch_size")
    parser.add_argument('--earlystop', default=3, type=int,
                        help="Number of epochs with no improvement after which training will be stopped.",
                        dest="earlystop")
    parser.add_argument('--verbose', default=True, type=bool_arg, help="If True (default), verbose output",
                        dest="verbose")

    # cross_val is not ready to be used
    parser.add_argument('--cross_val', default=0, type=int,
                        help="Number of folds (if bigger than 0) to use for cross validation. Default is 0.",
                        dest="cross_val")

    # no applying class weights
    parser.add_argument('--apply_class_weights', default=False, type=bool_arg,
                        help="If True, apply different loss weights (based on frequency of samples) to different "
                             "classes.",
                        dest="apply_class_weights")

    # no smooth factor
    parser.add_argument('--smooth_factor', default=0, type=float,
                        help="Smooth factor to be used when calculating class weights, so that highly unfrequent "
                        "classes do not get huge weights.",
                        dest="smooth_factor")

    # oversampling with neg to pos ratio=3
    parser.add_argument('--oversample', default=True, type=bool_arg,
                        help="If True (default), apply oversampling to generate balanced batches.",
                        dest="oversample")
    parser.add_argument('--ratio', default=3, type=int,
                        help="Ratio of negative to positive samples to use for balanced batch generation "
                             "(if oversample=True)",
                        dest="ratio")

    # activation: prelu
    parser.add_argument('--activation', default='prelu',
                        help="NN activation to be used. Default is prelu",
                        dest="activation")

    # no x_vars
    parser.add_argument('--x_vars', default=False, type=bool_arg, help="If True (default), include X variables",
                        dest="x_vars")

    # standardize numerical data
    parser.add_argument('--std', default=True, type=bool_arg, help="If True (default), standardize data.",
                        dest="std")

    # no pca
    parser.add_argument('--pca_whiten', default=False, type=bool_arg, help="If True (default), PCA-whiten data.",
                        dest="pca_whiten")
    parser.add_argument('--pca_reduce', default=0, type=float,
                        help="{0, 1, 0<x<1} If 0, no dimensionality reduction is done. If 1, Thomas P. Minka's method "
                             "('Automatic Choice of Dimensionality for PCA'. NIPS 2000) is used to determine the "
                             "number of dimensions to keep. If 0 < pca_reduce < 1, enough number of dimensions will "
                             "be kept to keep 'pca_reduce' percentage of variance explained. Default is 0.9.",
                        dest="pca_reduce")

    # one-hot encode cat data (embeddings are not used)
    parser.add_argument('--cat_enc', default='one-hot',
                        help="Encoding to be used for categorical variables. Default is 'integer' "
                             "(embedding layers will then be used). Other alternatives: 'hashing_char', "
                             "'hashing_all', 'one-hot'.",
                        dest="cat_enc")

    # no log transform
    parser.add_argument('--log_xform', default=False, type=bool_arg, help="If True (default), log-transform data.",
                        dest="log_xform")

    # encode categorical and binary data as 1/0
    parser.add_argument('--binary_enc', default=True, type=bool_arg,
                        help="If False (default), the negative cases of binary variables will be represented as -1 "
                             "instead of 0.", dest="binary_enc")

    # id for saving/ loading
    parser.add_argument('--data_split_id', default=2, type=int,
                        help="Id for the train-test data split to be used. If a new id is given, a new data split "
                             "will be generated and saved to disk with the given id. If id is 0 (default), a new "
                             "split will be generated, but not saved to disk. If a previously used id is given, "
                             "a previously generated and saved data split with that id will be used.",
                        dest="data_split_id")
    parser.add_argument("-f")
    args = parser.parse_args()
    return args

# read and format data from handset_model
def read_data(args):
    print("reading and formatting data (with handset_model)...")
    data_train, data_test, cat_levels = handset_model.load_and_preprocess_data(
        args)
    X_train = data_train['cat']  # cat only
    y_train = data_train['labels']['TARGET_S_TO_S_APPLE']

    X_train = np.array(X_train)
    y_train = np.reshape(np.array(y_train), [-1])
    sX_train = sparse.csr_matrix(X_train)

    y_test = data_test['labels']['TARGET_S_TO_S_APPLE']
    X_test = data_test['cat']  # cat only

    X_test = np.array(X_test)
    y_test = np.reshape(np.array(y_test), [-1])
    sX_test = sparse.csr_matrix(X_test)

    field_sizes = [0 for i in handset_model.CATEGORICAL_COLS]
    for col in data_train['cat'].columns:
        for cat_col in handset_model.CATEGORICAL_COLS:
            if cat_col in col:
                field_sizes[
                    handset_model.CATEGORICAL_COLS.index(cat_col)] += 1

    global SX_TRAIN
    SX_TRAIN = sX_train
    print("SX_TRAIN: ", type(SX_TRAIN), SX_TRAIN.shape)

    global Y_TRAIN
    Y_TRAIN = y_train
    print("Y_TRAIN: ", type(Y_TRAIN), Y_TRAIN.shape)

    global SX_TEST
    SX_TEST = sX_test
    print("SX_TEST: ", type(SX_TEST), SX_TEST.shape)

    global Y_TEST
    Y_TEST = y_test
    print("Y_TEST: ", type(Y_TEST), Y_TEST.shape)

    global FIELD_SIZES
    FIELD_SIZES = field_sizes
    print("FIELD_SIZES: ", FIELD_SIZES)

    global DATA_TRAIN_DICT
    DATA_TRAIN_DICT = data_train

    global DATA_TEST_DICT
    DATA_TEST_DICT = data_test

    global FIELD_OFFSETS
    FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
    print("FIELD_OFFSETS: ", FIELD_OFFSETS)

    global INPUT_DIM
    INPUT_DIM = sum(FIELD_SIZES)
    print("INPUT_DIM: ", INPUT_DIM)

    global INPUT_DIM_NUM
    INPUT_DIM_NUM = data_train['num'].shape[1]
    print("INPUT_DIM_NUM: ", INPUT_DIM_NUM)

# def read_data(file_name):
#     X = []
#     y = []
#     with open(file_name) as fin:
#         for line in fin:
#             fields = line.strip().split()
#             y_i = int(fields[0])
#             X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
#             y.append(y_i)
#             X.append(X_i)
#     y = np.reshape(np.array(y), [-1])
#     X = libsvm_2_coo(X, (len(X), INPUT_DIM)).tocsr()
#     return X, y


def read_data_tsv(file_name):
    data = np.loadtxt(file_name, delimiter='\t', dtype=np.float32)
    X, y = np.int32(data[:, :-1]), data[:, -1]
    X = libsvm_2_coo(X, (len(X), INPUT_DIM)).tocsr()
    return X, y / 5


def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]


def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    n = 0
    for d in libsvm_data:
        coo_rows.extend([n] * len(d))
        coo_cols.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.ones_like(coo_rows)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)


def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs


def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels


def split_data(data):
    fields = []
    for i in range(len(FIELD_OFFSETS) - 1):
        start_ind = FIELD_OFFSETS[i]
        end_ind = FIELD_OFFSETS[i + 1]
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, FIELD_OFFSETS[-1]:])
    return fields, data[1]

# modified for OverSamplingBatchGenerator


def split_data_gen(data):
    fields = []
    for i in range(len(FIELD_OFFSETS) - 1):
        start_ind = FIELD_OFFSETS[i]
        end_ind = FIELD_OFFSETS[i + 1]
        field_i = data[:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[:, FIELD_OFFSETS[-1]:])
    return fields


def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print('load variable map from', init_path, load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(
                tf.zeros(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(
                tf.ones(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(
                tf.ones(var_shape, dtype=dtype) * init_method)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method])
            else:
                print('BadParam: init method', init_method, 'shape',
                      var_shape, load_var_map[init_method].shape)
        else:
            print('BadParam: init method', init_method)
    return var_map


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + \
        indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + \
        indices[:, 1] * shape[2] * shape[3] + \
        indices[:, 2] * shape[3] + indices[:, 3]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def max_pool_2d(params, k):
    k = int(k)
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat([r1, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_2d(params, indices), [-1, k])


def max_pool_3d(params, k):
    k = int(k)
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat([r1, r2, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


def max_pool_4d(params, k):
    k = int(k)
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
    r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
    r2 = tf.tile(r2, [1, shape[2] * k])
    r3 = tf.tile(r3, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    r3 = tf.tile(tf.reshape(r3, [-1, 1]), [shape[0] * shape[1], 1])
    indices = tf.concat([r1, r2, r3, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_4d(params, indices), [-1, shape[1], shape[2], k])
