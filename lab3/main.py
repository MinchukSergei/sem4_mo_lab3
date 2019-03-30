import numpy as np
import tensorflow as tf
from lab1 import small_data, large_data, split_data, get_unique_data, encode_classes, generate_one_hot_encoded_class

from lab3.conv_nn import HP_BATCH_SIZE, HP_LEARNING_RATE, HP_UNITS, HP_L1_SCALE, HP_L2_SCALE, ConvNN
from lab3.conv_nn import DATA_TR_X, DATA_TR_Y, DATA_V_X, DATA_TE_X, DATA_TE_Y, DATA_V_Y
from lab3.conv_nn import HP_DROPOUT_RATE, HP_DECAY_RATE, HP_ACT_FUNC, HP_EPOCHS
from lab3.grid_search import GridSearch
from lab3.conv_nn import reset_graph
from lab3.le_net_5 import LeNet5

RND_SEED = None
tf.random.set_random_seed(RND_SEED)
np.random.seed(RND_SEED)

# Img props
img_w = 28
img_h = 28
img_size = img_h * img_w
img_classes = 10

# # Learning hyper params
epochs = 100
batch_size = 100
learning_rate = 0.001
units = [300, 300]

# # L1 Regularization
use_l1_regularization = True
l1_scale = 0.001

# # L2 Regularization
use_l2_regularization = True
l2_scale = 0.001

# # Dropout
use_dropout = True
dropout_rate = 0.15

# # Adaptive lr
use_adaptive_lr = True
decay_rate = 0.95


def main():
    tr_x, tr_y, v_x, v_y, te_x, te_y = prepare_data(small_data)

    # fc_nn = ConvNN(
    #     {
    #         DATA_TR_X: tr_x,
    #         DATA_TR_Y: tr_y,
    #         DATA_V_X: v_x,
    #         DATA_V_Y: v_y,
    #         DATA_TE_X: te_x,
    #         DATA_TE_Y: te_y
    #     },
    #     img_size,
    #     img_classes
    # )

    # grid_search = GridSearch(
    #     fc_nn,
    #     {
    #         HP_BATCH_SIZE: [300],
    #         HP_LEARNING_RATE: [0.003],
    #         HP_UNITS: [512],
    #         HP_L1_SCALE: [0],
    #         HP_L2_SCALE: [0.00005],
    #         HP_DROPOUT_RATE: [[0.05, 0.1, 0.4]],
    #         HP_DECAY_RATE: [0.80],
    #         HP_ACT_FUNC: [tf.nn.relu],
    #         HP_EPOCHS: [100]
    #     },
    #     fit_model,
    #     'accuracy',
    #     1
    # )

    lenet5 = LeNet5(
        {
            DATA_TR_X: tr_x,
            DATA_TR_Y: tr_y,
            DATA_V_X: v_x,
            DATA_V_Y: v_y,
            DATA_TE_X: te_x,
            DATA_TE_Y: te_y
        },
        img_size,
        img_classes
    )

    grid_search = GridSearch(
        lenet5,
        {
            HP_BATCH_SIZE: [128],
            HP_LEARNING_RATE: [0.005],
            HP_UNITS: [[120, 84]],
            HP_L1_SCALE: [0],
            HP_L2_SCALE: [0.0001],
            HP_DROPOUT_RATE: [[0.1, 0.2, 0.3, 0.4]],
            HP_DECAY_RATE: [0.85],
            HP_ACT_FUNC: [tf.nn.relu],
            HP_EPOCHS: [1000]
        },
        fit_model,
        'accuracy',
        1
    )

    grid_search.execute()

    print(grid_search.get_results())


def prepare_data(path_to_data):
    one_hot_encoded_labels = generate_one_hot_encoded_class()

    images, labels = get_unique_data(path_to_data)
    labels = encode_classes(labels, one_hot_encoded_labels)

    tr_x, tr_y, v_x, v_y, te_x, te_y = split_data(images, labels, 0.8, 0.1, 0.1)
    tr_x = tr_x.reshape(tr_x.shape[0], -1)
    v_x = v_x.reshape(v_x.shape[0], -1)
    te_x = te_x.reshape(te_x.shape[0], -1)

    return tr_x, tr_y, v_x, v_y, te_x, te_y


def fit_model(nn, hp):
    reset_graph()

    nn.init_hp(hp)
    nn.init_nn()
    nn.init_sess()
    nn.run_sess()

    best_score = nn.fit()
    # fc_nn.test_nn()
    return best_score


if __name__ == '__main__':
    main()
