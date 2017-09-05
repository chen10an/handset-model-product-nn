import numpy as np
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from product_nets_master.python import utils
from product_nets_master.python.models import LR, FM, PNN1, PNN2, FNN, CCPM

import time
from tqdm import tqdm
import pickle

import handset_model_copy4

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# train_file = '../data/train.yx.txt'
# test_file = '../data/test.yx.txt'
# fm_model_file = '../data/fm.model.txt'

# modified to read and format data from handset_model_copy4
args = utils.setArgs()
utils.read_data(args)

data_train_dict = utils.DATA_TRAIN_DICT  # dict from handset_model_copy4

train_data = utils.SX_TRAIN, utils.Y_TRAIN
train_data = utils.shuffle(train_data)
test_data = utils.SX_TEST, utils.Y_TEST

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS
input_dim = utils.INPUT_DIM

# SETTINGS
# reduce rounds so that training doesn't take too long (for experimentation)
min_round = 1
num_round = 10
early_stop_round = 3
batch_size = 256

algo = 'ccpm'
ratio = 3

def train(model):
    print("training model with %s" % (algo))
    history_score = []
    history_pred = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []

            # modified for OversamplingBatchGenerator
            gen = handset_model_copy4.OverSamplingBatchGenerator(data_train_dict, batch_size=batch_size, r=ratio)
            for j in tqdm(range(int(np.floor(train_size / batch_size + 1)))):
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
                # X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)

        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]

        train_preds = model.run(model.y_prob, utils.slice(train_data)[0])
        test_preds = model.run(model.y_prob, utils.slice(test_data)[0])

        roc_auc_train = roc_auc_score(train_data[1], train_preds)
        roc_auc_test = roc_auc_score(test_data[1], test_preds)

        # added more metrics
        a_train = accuracy_score(train_data[1], np.rint(train_preds))
        a_test = accuracy_score(test_data[1], np.rint(test_preds))

        p_train = precision_score(train_data[1], np.rint(train_preds))
        p_test = precision_score(test_data[1], np.rint(test_preds))

        r_train = recall_score(train_data[1], np.rint(train_preds))
        r_test = recall_score(test_data[1], np.rint(test_preds))

        m_train = confusion_matrix(train_data[1], np.rint(train_preds))
        m_test = confusion_matrix(test_data[1], np.rint(test_preds))
        true_pos_rate_train = m_train[1][1]/(m_train[1][1]+m_train[1][0])
        true_pos_rate_test = m_test[1][1]/(m_test[1][1]+m_test[1][0])

        print('[%d]' % (i))
        print('loss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (np.mean(ls), roc_auc_train, roc_auc_test))
        print('train-accuracy: %f\teval-accuracy: %f' % (a_train, a_test))
        print('train-precision: %f\teval-precision: %f' % (p_train, p_test))
        print('train-recall: %f\teval-recall: %f' % (r_train, r_test))

        print('train-confusion-matrix:\n', m_train)
        print('test-confusion-matrix:\n', m_test)
        print('train-true-pos-rate: %f\teval-true-pos-rate: %f' % (true_pos_rate_train, true_pos_rate_test))

        history_pred.append((train_preds, test_preds))
        history_score.append(p_test)  # score in terms of precision

        # return history of precision score and best iteration in terms of precision
        if i > min_round and i > early_stop_round:
            # if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
            #             -1 * early_stop_round] < 1e-5:
            # fixed implementation of early stopping
            if np.argmax(history_score) == i - early_stop_round and history_score[i] - history_score[
                        i - early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-precision: %f' % (
                    np.argmax(history_score), np.max(history_score)))

                # return predictions to save later
                best_iter = np.argmax(history_score)
                return history_pred, best_iter

            if i == num_round - 1:

                # return predictions to save later
                best_iter = num_round - 1;
                return history_pred, best_iter


if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0
    }

    model = LR(**lr_params)
elif algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }

    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'random_seed': 0
    }

    model = FNN(**fnn_params)
elif algo == 'ccpm':
    ccpm_params = {
        'layer_sizes': [field_sizes, 10, 5, 3],
        'layer_acts': ['tanh', 'tanh', 'none'],
        'drop_out': [0, 0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': 0
    }

    model = CCPM(**ccpm_params)
elif algo == 'pnn1':
    pnn1_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN1(**pnn1_params)
elif algo == 'pnn2':
    pnn2_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN2(**pnn2_params)

if algo in {'fnn', 'ccpm', 'pnn1', 'pnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)

# save predictions
history_pred, best_iter = train(model)

history_pred_path = "saved-models/%s_%d_history_pred.pickle" % (algo, ratio)
pickle.dump((history_pred, best_iter), open(history_pred_path, "wb"))
print("saved history of predictions to %s" % (history_pred_path))

# doesn't work:
# model_path = "saved-models/%s_model.pickle" % (algo)
# pickle.dump(model, open(model_path, "wb"))
# print("saved model to %s" % (model_path))

# model.dump("saved-models/%s_model.pickle" % (algo)) haven't debugged this yet

# X_i, y_i = utils.slice(train_data, 0, 100)
# fetches = [model.tmp1, model.tmp2]
# tmp1, tmp2 = model.run(fetches, X_i, y_i)
# print tmp1.shape
# print tmp2.shape
