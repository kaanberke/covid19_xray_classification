import argparse
import os
from pathlib import Path
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                        LearningRateScheduler)
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from model import create_model, scheduler

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

p = Path(args.data)
types = ("**/*.jpg", "**/*.jpeg", "**/*.png")
files_grabbed = []

for files in types:
    files_grabbed.extend(p.glob(files))

df = pd.DataFrame(columns=("filename", "label"))

for idx in range(len(files_grabbed)):
    df = df.append(
        {
            "filename": files_grabbed[idx].name,
            "label": files_grabbed[idx].parent.name
        },
        ignore_index=True)

df = df.sample(frac=1)
df.to_csv("training_labels.csv")

model = create_model()
model.summary()
opt = SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
# metrics=[categorical_accuracy])

train_data = pd.read_csv('training_labels.csv')
Y = train_data[['label']]
kf = KFold(n_splits=5)
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


def get_model_name(k):
    return 'last_model_' + str(k) + '.h5'


VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = './models/'
fold_var = 1

image_dir = "./all_data/"
idg = ImageDataGenerator(rescale=1. / 255)

for train_index, val_index in kf.split(np.zeros(len(Y)), Y):
    print("#" * 30)
    print("Fold no:", fold_var)
    print("#" * 30)

    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]

    train_data_generator = idg.flow_from_dataframe(training_data,
                                                   directory=image_dir,
                                                   x_col="filename",
                                                   y_col="label",
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   color_mode="grayscale",
                                                   target_size=TARGET_SIZE)

    valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                   directory=image_dir,
                                                   x_col="filename",
                                                   y_col="label",
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   color_mode="grayscale",
                                                   target_size=TARGET_SIZE)

    cb = [
        # EarlyStopping(patience=20),
        ModelCheckpoint(save_dir + get_model_name(fold_var),
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="min"),
        TensorBoard(log_dir="logs"),
        LearningRateScheduler(scheduler)
    ]

    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    history = model.fit(train_data_generator,
                        epochs=EPOCHS,
                        callbacks=cb,
                        validation_data=valid_data_generator)

    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights(save_dir + "model_" + str(fold_var) + ".h5")

    results = model.evaluate(valid_data_generator)
    results = dict(zip(model.metrics_names, results))

    VALIDATION_ACCURACY.append(results['acc'])
    VALIDATION_LOSS.append(results['loss'])

    tf.keras.backend.clear_session()

    fold_var += 1
