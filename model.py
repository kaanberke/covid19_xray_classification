from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten,
                                     BatchNormalization, Activation, Dropout)
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential


class TestCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        print('\nTesting loss: {:.2f}, acc: {:.2f}\n'.format(loss, acc))


def scheduler(epoch, lr):
    return lr * 1e-6 if epoch >= 10 and epoch % 100 == 0 else lr


def default_conv(filters,
                 kernel_size=(3, 3),
                 strides=1,
                 padding="SAME",
                 **kwargs):
    return [
        Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               **kwargs),
        BatchNormalization(),
        Activation(relu),
        Dropout(0.5),
    ]


layers = [
    Input(shape=(224, 224, 1)),
    *default_conv(128, (7, 7), strides=2, padding="SAME"),
    *default_conv(64, (3, 3), strides=1, padding="SAME"),
    *default_conv(32, (3, 3), strides=1, padding="SAME"),
    Flatten(),
    Dense(128, relu),
    Dense(64, relu),
    Dropout(0.5),
    Dense(3, activation="softmax"),
]


def create_model():
    model = Sequential(layers=layers)
    return model
