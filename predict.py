from tensorflow.keras.optimizers import SGD
from model import create_model
from pathlib import Path
import numpy as np
from collections import Counter

# TODO
LABELS = [
    "COVID",
    "PNEUMONIA",
    "NORMAL"
]

model = create_model()
# model.summary()
opt = SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])


def predict(models_folder: str, image: np.array):
    models_path = Path(models_folder)
    img = np.expand_dims(image, axis=0)
    predictions = []
    for model_path in models_path.glob("model_*.h5"):
        model.load_weights(model_path)
        prediction = model.predict(img)
        max_arg = np.argmax(prediction)
        predictions.append(max_arg)
    result = Counter(predictions).most_common(1)[0][0]
    return result
