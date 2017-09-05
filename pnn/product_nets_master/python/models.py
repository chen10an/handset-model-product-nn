import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf

from product_nets_master.python import utils

dtype = utils.DTYPE


class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self, fetches, X=None, y=None, X_num=None, mode='train'):
        feed_dict = {}
        if type(self.X) is list:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        else:
            feed_dict[self.X] = X
        if y is not None:
            feed_dict[self.y] = y
        if self.layer_keeps is not None:
            if mode == 'train':
                feed_dict[self.layer_keeps] = self.keep_prob_train
            elif mode == 'test':
                feed_dict[self.layer_keeps] = self.keep_prob_test

        # numerical input
        if X_num is not None:
            feed_dict[self.X_num] = X_num

        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.items():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)


class LR(Model):
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2, l2_weight=0,
                 random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('v', [input_dim, factor_order], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']

            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b + p, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class FNN(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                self.layer_keeps[0])

            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.reshape(l, [-1])
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class CCPM(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        embedding_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = embedding_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('f1', [embedding_order, layer_sizes[2], 1, 2], 'tnormal', dtype))
        init_vars.append(('f2', [embedding_order, layer_sizes[3], 2, 2], 'tnormal', dtype))
        init_vars.append(('w1', [2 * 3 * embedding_order, 1], 'tnormal', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                utils.activate(
                    tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i]
                               for i in range(num_inputs)], 1),
                    layer_acts[0]),
                self.layer_keeps[0])
            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embedding_order, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    num_inputs / 2),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                utils.activate(
                    tf.reshape(l, [-1, embedding_order * 3 * 2]),
                    layer_acts[1]),
                self.layer_keeps[1])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[2]),
                self.layer_keeps[2])

            l = tf.reshape(l, [-1])
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class PNN1(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, kernel_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None, include_num=True):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):  # when len=3 nothing in this loop is executed
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():  # added name scopes for visualization in tensorboard
            if random_seed is not None:
                tf.set_random_seed(random_seed)

            with tf.name_scope("cat_inputs"):
                self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            with tf.name_scope("labels"):
                self.y = tf.placeholder(dtype)

            with tf.name_scope("layer_keeps"):
                self.keep_prob_train = 1 - np.array(drop_out)
                self.keep_prob_test = np.ones_like(drop_out)
                self.layer_keeps = tf.placeholder(dtype)

            with tf.name_scope("cat_layer_1"):
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
                xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
                x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)

                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    self.layer_keeps[0])

            with tf.name_scope("cat_product_layer_2"):
                w1 = self.vars['w1']
                k1 = self.vars['k1']
                b1 = self.vars['b1']

                p = tf.reduce_sum(
                    tf.reshape(
                        tf.matmul(
                            tf.reshape(
                                tf.transpose(
                                    tf.reshape(l, [-1, num_inputs, factor_order]),
                                    [0, 2, 1]),
                                [-1, num_inputs]),
                            k1),
                        [-1, factor_order, layer_sizes[2]]),
                    1)

                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, w1) + b1 + p,
                        layer_acts[1]),
                    self.layer_keeps[1])

            with tf.name_scope("cat_final_layers"):
                for i in range(2, len(layer_sizes) - 1):  # when len=3 nothing in this loop is executed
                    wi = self.vars['w%d' % i]
                    bi = self.vars['b%d' % i]
                    l = tf.nn.dropout(
                        utils.activate(
                            tf.matmul(l, wi) + bi,
                            layer_acts[i]),
                        self.layer_keeps[i])

            # added nn for numerical features
            if include_num:
                with tf.name_scope("num_sub_graph"):
                    # Hidden layer sizes
                    n_1 = 128
                    n_2 = 128
                    n_3 = 128

                    with tf.name_scope("num_input"):
                        self.X_num = tf.placeholder(dtype, [None, utils.INPUT_DIM_NUM])

                    with tf.name_scope("num_weights"):
                        weights_num = {
                            'l_1': tf.Variable(tf.random_normal([utils.INPUT_DIM_NUM, n_1]), dtype=dtype),
                            'l_2': tf.Variable(tf.random_normal([n_1, n_2]), dtype=dtype),
                            'l_3': tf.Variable(tf.random_normal([n_2, n_3]), dtype=dtype),
                        }

                    with tf.name_scope("num_biases"):
                        biases_num = {
                            'l_1': tf.Variable(tf.random_normal([n_1]), dtype=dtype),
                            'l_2': tf.Variable(tf.random_normal([n_2]), dtype=dtype),
                            'l_3': tf.Variable(tf.random_normal([n_3]), dtype=dtype),
                        }

                    def model_num(X, W, b):
                        l_1 = tf.add(tf.matmul(X, W['l_1']), b['l_1'])
                        l_1 = tf.nn.relu(l_1)

                        l_2 = tf.add(tf.matmul(l_1, W['l_2']), b['l_2'])
                        l_2 = tf.nn.relu(l_2)

                        l_3 = tf.add(tf.matmul(l_2, W['l_3']), b['l_3'])
                        l_3 = tf.nn.relu(l_3)
                        return l_3

                    with tf.name_scope("num_output_before_sigmoid"):
                        l_num = model_num(self.X_num, weights_num, biases_num)

                    with tf.name_scope("combined_output_before_sigmoid"):
                        l = tf.concat([l, l_num], axis=1)

                        w_out = tf.Variable(tf.random_normal([n_3 + layer_sizes[-1], 1]), dtype=dtype)
                        b_out = tf.Variable(tf.random_normal([1]), dtype=dtype)
                        l = tf.add(tf.matmul(l, w_out), b_out)

            with tf.name_scope("output_before_sigmoid"):
                l = tf.reshape(l, [-1])

            with tf.name_scope("sigmoid_output"):
                self.y_prob = tf.sigmoid(l)

            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
                if layer_l2 is not None:  # default: [0, 0] (nothing added to loss function)
                    # for i in range(num_inputs):
                    self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                    for i in range(1, len(layer_sizes) - 1):
                        wi = self.vars['w%d' % i]
                        # bi = self.vars['b%d' % i]
                        self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
                if kernel_l2 is not None:  # default: 0 (nothing added to loss function)
                    self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            with tf.name_scope("optimizer"):
                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

            # visualize in tensorboard
            writer = tf.summary.FileWriter('pnn1_logs')
            writer.add_graph(self.sess.graph)


class PNN2(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, kernel_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [factor_order * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, factor_order]), 1)
            p = tf.reshape(
                tf.matmul(tf.reshape(z, [-1, factor_order, 1]),
                          tf.reshape(z, [-1, 1, factor_order])),
                [-1, factor_order * factor_order])
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + tf.matmul(p, k1) + b1,
                    layer_acts[1]),
                self.layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.reshape(l, [-1])
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
