import tensorflow as tf
import numpy as np
from lab1 import data_shuffle

DATA_TR_X = 'tr_x'
DATA_TR_Y = 'tr_y'
DATA_V_X = 'v_x'
DATA_V_Y = 'v_y'
DATA_TE_X = 'te_x'
DATA_TE_Y = 'te_y'
HIDDEN_FC_LAYER_WEIGHT = 'HFCL'
OUT_FC_LAYER_WEIGHT = 'OFCL'
PRINT_TRAIN_OUTPUT = True
DISPLAY_FREQUENCY = 100

# Hyper Params
HP_BATCH_SIZE = 'batch_size'
HP_LEARNING_RATE = 'learning_rate'
HP_UNITS = 'units'
HP_L1_SCALE = 'l1_scale'
HP_L2_SCALE = 'l2_scale'
HP_DROPOUT_RATE = 'dropout_rate'
HP_DECAY_RATE = 'decay_rate'
HP_EPOCHS = 'epochs'
HP_ACT_FUNC = 'act_func'


class LeNet5:
    def __init__(self, data, in_size, out_size):
        self.tr_x = data[DATA_TR_X]
        self.tr_y = data[DATA_TR_Y]
        self.v_x = data[DATA_V_X]
        self.v_y = data[DATA_V_Y]
        self.te_x = data[DATA_TE_X]
        self.te_y = data[DATA_TE_Y]

        self.in_size = in_size
        self.out_size = out_size

        self.use_adaptive_lr = False
        self.use_l1_reg = False
        self.use_l2_reg = True
        self.use_dropout = True

    def init_sess(self):
        self.sess = tf.InteractiveSession()

    def run_sess(self):
        self.sess.run(self.init)

    def init_in_out_layers(self):
        self.nn_in = tf.placeholder(tf.float32, shape=[None, self.in_size], name='IN')
        self.nn_out = tf.placeholder(tf.float32, shape=[None, self.out_size], name='OUT')

    def init_nn(self):
        self.init_in_out_layers()

        self.init_layers()
        self.init_loss()
        self.init_learning_rate()
        self.init_optimizer()

        self.correct_prediction = tf.equal(tf.argmax(self.output_logits, 1), tf.argmax(self.nn_out, 1),
                                           name='correct_prediction')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

        self.init = tf.global_variables_initializer()

    def init_hp(self, hp):
        self.learning_rate = hp[HP_LEARNING_RATE]
        self.units = hp[HP_UNITS]
        self.l1_scale = hp[HP_L1_SCALE]
        self.l2_scale = hp[HP_L2_SCALE]
        self.batch_size = hp[HP_BATCH_SIZE]
        self.dropout_rate = hp[HP_DROPOUT_RATE]
        self.decay_rate = hp[HP_DECAY_RATE]
        self.epochs = hp[HP_EPOCHS]
        self.act_func = hp[HP_ACT_FUNC]
        self.train_size = len(self.tr_y)

    def init_layers(self):
        in_nn_reshaped = tf.reshape(self.nn_in, [-1, 28, 28, 1])

        k_conv_size = [5, 5]
        k_pool_size = [2, 2]

        l1 = self.conv_layer(in_nn_reshaped, 1, 6, k_conv_size, f'{HIDDEN_FC_LAYER_WEIGHT}0', 'SAME', 0)

        l2 = self.avg_pool(l1, k_pool_size, 'VALID')

        l3 = self.conv_layer(l2, 6, 16, k_conv_size, f'{HIDDEN_FC_LAYER_WEIGHT}1', 'VALID', 1)

        l4 = self.avg_pool(l3, k_pool_size, 'VALID')

        fc1 = self.full_connected_layer(l4, self.units[0], f'{HIDDEN_FC_LAYER_WEIGHT}2', 2)

        fc2 = self.full_connected_layer(fc1, self.units[1], f'{HIDDEN_FC_LAYER_WEIGHT}3', 3)

        self.output_logits = self.full_connected_layer(fc2, self.out_size, OUT_FC_LAYER_WEIGHT, output=True)

    def init_loss(self):
        loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.nn_out, logits=self.output_logits),
            name='loss_function'
        )

        l1_penalty = 0
        l2_penalty = 0

        weights = []
        for i in range(4):
            weights.append(get_tensor_by_name(f'W_HFCL{i}'))
        weights.append(get_tensor_by_name('W_OFCL'))

        if self.use_l1_reg:
            w_penalty = 0
            for w in weights:
                w_penalty += l1_loss(w)
            l1_penalty = self.l1_scale * w_penalty

        if self.use_l2_reg:
            w_penalty = 0
            for w in weights:
                w_penalty += tf.nn.l2_loss(w)
            l2_penalty = self.l2_scale * w_penalty

        self.loss_function = loss_function + l1_penalty + l2_penalty

    def init_learning_rate(self):
        global_step = None
        learn_rate = self.learning_rate

        if self.use_adaptive_lr:
            global_step = tf.Variable(0, trainable=False)

            learn_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step,
                self.train_size,
                self.decay_rate,
                staircase=True
            )

        self.learn_rate = learn_rate
        self.global_step = global_step

    def init_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learn_rate
        ).minimize(
            loss=self.loss_function,
            global_step=self.global_step
        )

        self.optimizer = optimizer

    def avg_pool(self, input_data, pool_shape, padding):
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 2, 2, 1]
        return tf.nn.avg_pool(input_data, ksize=ksize, strides=strides, padding=padding)

    def conv_layer(self, input_data, num_input_channels, num_filters, filter_shape, name, padding, layer_number=None):
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

        W = weight_variable(name, conv_filt_shape)
        b = bias_variable(name, [num_filters])

        out_layer = tf.nn.conv2d(input_data, W, [1, 1, 1, 1], padding=padding) + b

        out_layer = tf.nn.relu(out_layer)

        if self.use_dropout:
            out_layer = tf.nn.dropout(out_layer, rate=self.dropout_rate[layer_number])

        return out_layer

    def full_connected_layer(self, x, num_units, name, layer_number=None, output=False):
        in_dim = 1

        for i in x.shape:
            if i.value is not None:
                in_dim *= i.value

        W = weight_variable(name, shape=[in_dim, num_units])
        b = bias_variable(name, [num_units])

        flattened_x = tf.reshape(x, [-1, in_dim])
        layer = tf.matmul(flattened_x, W) + b

        if not output:
            layer = self.act_func(layer)

        if self.use_dropout and not output:
            layer = tf.nn.dropout(layer, rate=self.dropout_rate[layer_number])

        return layer

    def fit(self):
        global_step = 0

        loss_function = self.loss_function
        accuracy = self.accuracy
        batch_size = self.batch_size
        sess = self.sess
        nn_in = self.nn_in
        nn_out = self.nn_out

        num_tr_iter = int(len(self.tr_y) / batch_size)
        num_v_iter = int(len(self.v_y) / batch_size)

        best_score = {
            'accuracy': 0,
            'loss': 0
        }

        for epoch in range(self.epochs):
            print('Training epoch: {}'.format(epoch + 1))

            tr_x, tr_y = data_shuffle(self.tr_x, self.tr_y)
            v_x, v_y = data_shuffle(self.v_x, self.v_y)

            for iteration in range(num_tr_iter):
                global_step += 1
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(tr_x, tr_y, start, end)

                feed_dict_batch = {
                    nn_in: x_batch,
                    nn_out: y_batch
                }

                self.sess.run(self.optimizer, feed_dict=feed_dict_batch)

                if iteration % DISPLAY_FREQUENCY == 0:
                    loss_batch, acc_batch = sess.run([loss_function, accuracy], feed_dict=feed_dict_batch)

                    if PRINT_TRAIN_OUTPUT:
                        print("iter {0:3d}:\t TRAIN Loss={1:.2f},\t Accuracy={2:.01%}".format(iteration, loss_batch,
                                                                                              acc_batch))
            correct_predictions = []
            for iteration in range(num_v_iter):
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(v_x, v_y, start, end)

                feed_dict_valid = {
                    nn_in: x_batch,
                    nn_out: y_batch
                }

                correct_prediction = sess.run([self.correct_prediction], feed_dict=feed_dict_valid)
                correct_predictions.extend(correct_prediction[0])

            acc_valid = sess.run(tf.reduce_mean(tf.cast(correct_predictions, tf.float32)))

            if acc_valid > best_score['accuracy']:
                best_score['accuracy'] = acc_valid

            if PRINT_TRAIN_OUTPUT:
                print('---------------------------------------------------------')
                print("Epoch: {0}:\t VALID Accuracy={1:.01%}".format(epoch + 1, acc_valid))
                print('---------------------------------------------------------')

        return best_score

    def test_nn(self):
        feed_dict_valid = {
            self.nn_in: self.te_x,
            self.nn_out: self.te_y
        }

        loss_valid, acc_valid = self.sess.run([self.loss_function, self.accuracy], feed_dict=feed_dict_valid)

        print('---------------------------------------------------------')
        print("TEST Loss: {0:.2f}, Accuracy: {1:.01%}".format(loss_valid, acc_valid))
        print('---------------------------------------------------------')

        return acc_valid, loss_valid


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(f'{name}:0')


def weight_variable(name, shape):
    initial = tf.truncated_normal_initializer(stddev=0.01)

    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initial)


def bias_variable(name, shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)

    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def get_next_batch(data_x, data_y, start, end):
    x_batch = data_x[start:end]
    y_batch = data_y[start:end]

    return x_batch, y_batch


def l1_loss(w):
    return tf.reduce_sum(tf.abs(w))


def reset_graph():
    tf.reset_default_graph()
