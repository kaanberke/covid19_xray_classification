from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dropout

from data_loader import DataLoader
import argparse
import os
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense,
                                     Flatten)
from tensorflow.keras import Model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class TestCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        print('\nTesting loss: {:.2f}, acc: {:.2f}\n'.format(loss, acc))


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data",
    help="Data directory which involves images and metadata.csv")
parser.add_argument("-e",
                    "--epochs",
                    default=100,
                    type=int,
                    help="Number of epochs")
parser.add_argument(
    "-b",
    "--batch_size",
    default=32,
    type=int,
    help=
    "Batch size, the number of training examples utilized in one iteration",
)
parser.add_argument("-t",
                    "--target_size",
                    nargs=2,
                    default=(224, 224),
                    type=int,
                    help="The width and length of the images to be trained")
args = parser.parse_args()
assert args.data and os.path.exists(
    args.data), "Please provide a valid data path.."

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
TARGET_SIZE = args.target_size

d = DataLoader(args.data, BATCH_SIZE, TARGET_SIZE)
train_generator, val_generator = d.get_image_generator()

input_layer = Input(shape=(224, 224, 1), name="original_img")
x = Conv2D(128, 3, activation="relu")(input_layer)
x = Conv2D(64, 3, activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, 3, activation="relu")(x)
x = Dropout(0.5)(x)
x = Conv2D(32, 3, activation="relu")(x)
x = Flatten()(x)
output_layer = Dense(2, activation="softmax")(x)

model = Model(input_layer, output_layer)
model.summary()

opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=[categorical_accuracy])

if not os.path.exists("models"):
    os.makedirs("models")
model_filepath = os.path.join("models", "model.{epoch:02d}-{val_loss:.4f}.h5")
cb = [
    EarlyStopping(patience=20),
    ModelCheckpoint(filepath=model_filepath, save_best_only=True),
    TensorBoard(log_dir="logs")
]
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // train_generator.batch_size,
    epochs=EPOCHS,
    verbose=1,
    shuffle=True,
    callbacks=[TestCallback(val_generator)])
