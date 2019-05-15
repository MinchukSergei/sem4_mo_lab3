from lab1 import small_data, large_data, split_data, get_unique_data, encode_classes, generate_one_hot_encoded_class

# Img props
from lab3.conv_nn import ConvNN
from lab3.le_net_5 import Lenet5

img_w = 28
img_h = 28
img_size = img_h * img_w
img_classes = 10


def main():
    nn = Lenet5(
        prepare_data(small_data),
        img_size,
        img_classes,
        epochs=50
    )

    nn.build_model()
    nn.fit_model()
    nn.test_model()


def prepare_data(path_to_data):
    one_hot_encoded_labels = generate_one_hot_encoded_class()

    images, labels = get_unique_data(path_to_data)
    labels = encode_classes(labels, one_hot_encoded_labels)

    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(images, labels, 0.92, 0.03, 0.05)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train / 255, y_train), (x_valid / 255, y_valid), (x_test / 255, y_test)


if __name__ == '__main__':
    main()
