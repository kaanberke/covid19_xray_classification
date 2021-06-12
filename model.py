from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten,
                                     BatchNormalization, Activation, Dropout)
from tensorflow.keras.activations import relu


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


def create_model():
    layers = [
        *default_conv(128, (3, 3), strides=1, padding="VALID"),
        *default_conv(256, (3, 3), strides=1, padding="VALID"),

        Flatten(),
        Dense(32, relu),
        Dropout(0.5),
        Dense(3, activation="softmax"),

    ]

    input_layer = Input((224, 224, 3))
    vgg16_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=input_layer,
        classes=3,
    )
    for layer in vgg16_model.layers:
        layer.trainable = False

    model = None
    for layer in layers:
        if model is None:
            model = layer(vgg16_model.layers[-1].output)
        else:
            model = layer(model)

    model = Model(inputs=vgg16_model.inputs, outputs=model)
    return model
