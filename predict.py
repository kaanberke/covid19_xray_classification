from tensorflow.python.keras.models import load_model

from pathlib import Path
import numpy as np
# from .model import create_model
# from tensorflow.keras.optimizers import SGD
# from collections import Counter

# TODO
LABELS = [
    "COVID",
    "NORMAL",
    "PNEUMONIA",
]

# No need to create model whilst loading one.
"""
model = create_model()
# model.summary()
opt = SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
"""


def predict(models_folder: Path, image: np.array):
    # Voting System ( Can be activated later. )
    """
    models_path = Path(models_folder)
    img = np.expand_dims(image, axis=0)
    predictions = []
    for model_path in models_path.glob("vgg16_transfer_learning_model.h5"):
        model.load_weights(model_path)
        prediction = model.predict(img)
        max_arg = np.argmax(prediction)
        predictions.append(max_arg)
    result = Counter(predictions).most_common(1)[0][0]
    return result
    """

    model = load_model(models_folder / "vgg16_transfer_learning_model.h5")
    img = np.expand_dims(image, axis=0)
    prediction = model.predict(img)
    result = np.argmax(prediction)
    return result
