import pandas as pd
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
from typing import Tuple

DATA_GEN_ARGS = dict(rescale=1. / 255.,
                     validation_split=0.25,
                     rotation_range=18,
                     brightness_range=[0.8, 1.0],
                     zoom_range=[0.05, 0.15],
                     width_shift_range=0.1,
                     height_shift_range=0.1)


class DataLoader(object):
    def __init__(self, data_dir: str, batch_size: int,
                 target_size: Tuple[int, int]):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self._df_total = None
        self.data_gen = ImageDataGenerator(**DATA_GEN_ARGS)

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, new_data_dir: str):
        assert os.path.exists(
            new_data_dir), "Please provide a valid data path.."
        self._data_dir = new_data_dir

    @data_dir.deleter
    def data_dir(self):
        print("Data directory cannot be deleted.")

    @property
    def df_total(self) -> pd.DataFrame:
        return self._df_total

    @df_total.setter
    def df_total(self, value):
        assert isinstance(value, pd.DataFrame), "Please provide a DataFrame.."
        self._df_total = value

    @df_total.deleter
    def df_total(self):
        print("df_total cannot be deleted.")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        assert isinstance(value, int), "Batch size must be an integer.."
        self._batch_size = value

    @batch_size.deleter
    def batch_size(self):
        print("Batch size value cannot be deleted.")

    @property
    def target_size(self) -> Tuple[int, int]:
        return self._target_size

    @target_size.setter
    def target_size(self, value: Tuple[int, int]):
        assert (isinstance(value[0], int) and
                isinstance(value[1], int)),\
            "Target size must be a tuple that contains two integer values."
        self._target_size = value

    @target_size.deleter
    def target_size(self):
        print("Target size value cannot be deleted.")

    def get_image_generator(
            self) -> Tuple[DataFrameIterator, DataFrameIterator]:
        train_generator = self.data_gen.flow_from_directory(
            directory=self._data_dir,
            subset="training",
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical",
            color_mode="grayscale",
            target_size=self.target_size)

        valid_generator = self.data_gen.flow_from_directory(
            directory=self._data_dir,
            subset="validation",
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical",
            color_mode="grayscale",
            target_size=self.target_size)

        return train_generator, valid_generator


if __name__ == "__main__":
    ...
