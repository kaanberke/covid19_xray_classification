import pandas as pd
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
from typing import Tuple


class DataLoader(object):
    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._df_total = None

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, new_data_dir: str):
        if not os.path.exists(new_data_dir):
            print("Please enter an exist data directory.")
            return
        self._data_dir = new_data_dir

    @data_dir.deleter
    def data_dir(self):
        print("Data directory cannot be deleted.")

    @property
    def df_total(self) -> pd.DataFrame:
        return self._df_total

    @df_total.setter
    def df_total(self, value):
        print("df_total cannot be set exclusively.")

    @df_total.deleter
    def df_total(self):
        print("df_total cannot be deleted.")

    def _prepare_data(self):
        df = pd.read_csv(os.path.join(self._data_dir, "metadata.csv"),
                         delimiter=',')
        no_finding_df = df[df.finding == "No Finding"].copy()
        covid_df = df[df.finding == "Pneumonia/Viral/COVID-19"]
        del df

        for df, finding in zip([no_finding_df, covid_df], ["Normal", "Covid"]):
            df.loc[:, ["finding"]] = finding
            df.drop(df.columns.difference(["finding", "folder", "filename"]),
                    1,
                    inplace=True)

        self._df_total = pd.concat([no_finding_df, covid_df])
        self._df_total.reset_index(drop=True, inplace=True)

    def get_image_generator(
            self) -> Tuple[DataFrameIterator, DataFrameIterator]:
        self._prepare_data()
        assert len(self._df_total) > 0, "Insufficient number of samples."
        datagen = ImageDataGenerator(rescale=1. / 255.,
                                     validation_split=0.25,
                                     rotation_range=18,
                                     brightness_range=[0.8, 1.0],
                                     zoom_range=[0.05, 0.15])
        train_generator = datagen.flow_from_dataframe(
            dataframe=self._df_total,
            directory=os.path.join(self._data_dir,
                                   self._df_total.loc[0, "folder"]),
            x_col="filename",
            y_col="finding",
            subset="training",
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(224, 224))

        valid_generator = datagen.flow_from_dataframe(
            dataframe=self._df_total,
            directory=os.path.join(self._data_dir,
                                   self._df_total.loc[0, "folder"]),
            x_col="filename",
            y_col="finding",
            subset="validation",
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(224, 224))

        return train_generator, valid_generator


if __name__ == "__main__":
    ...
