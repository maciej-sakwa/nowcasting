"""Image generation"""

from typing import Tuple

import pandas as pd
from keras.utils import Sequence, to_categorical
import numpy as np
import cv2
import os


class DataGenerator_SCNN(Sequence):

    'Generates data for Keras'
    def __init__(self, dataframe, batch_size=128, dim=(128, 128, 3), channel_IMG = 1, shuffle=False, iftest=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel_IMG = channel_IMG
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            for i_IMG in range(len(img_row)): # stack images in channels
                img = cv2.imread(img_row[i_IMG], 0)
                if img is None:
                    raise ValueError(f'{img_row[i_IMG]} does not exist')
                else:
                    X[n_index, :, :, i_IMG] = img / 255.0
            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, y, file_name
        elif self.iftest == False:
            return X, y


class DataGeneratorClass_SCNN(Sequence):

    'Generates data for Keras'
    def __init__(self, dataframe, batch_size=128, dim=(128, 128, 3), channel_IMG = 1, classes = 4,shuffle=False, iftest=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel_IMG = channel_IMG
        self.classes = classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            for i_IMG in range(len(img_row)): # stack images in channels
                img = cv2.imread(img_row[i_IMG], 0)
                if img is None:
                    raise ValueError(f'{img_row[i_IMG]} does not exist')
                else:
                    X[n_index, :, :, i_IMG] = img / 255.0
            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, to_categorical(y, num_classes=self.classes), file_name
        elif self.iftest == False:
            return X, to_categorical(y, num_classes=self.classes)


class DataGeneratorDiff_SCNN(Sequence):
    """Generates batches of images as well as their associated class labels on the fly.
    To be passed as argument in the fit_generator function of Keras."""
    'Generates data for Keras'

    def __init__(self, dataframe, batch_size=128, dim=(128, 128, 3), channel_IMG=1, diff=2, shuffle=False, iftest=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel_IMG = channel_IMG
        self.diff = diff
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        self.dim2 = (self.dim[0], self.dim[1], self.dim[-1] + self.diff)
        X = np.empty((self.batch_size, *self.dim2), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            for i_IMG in range(len(img_row)-1):  # stack images in channels
                img0 = cv2.imread(img_row[i_IMG], 0)
                img2 = cv2.imread(img_row[i_IMG+1], 0)
                img1 = img2-img0
                X[n_index, :, :, i_IMG * self.diff] = img0 / 255.0
                X[n_index, :, :, i_IMG * self.diff + 1] = img1 / 255.0
                X[n_index, :, :, i_IMG * self.diff + 2] = img2 / 255.0

            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, y, file_name
        elif self.iftest == False:
            return X, y


class DataGeneratorGHI_SCNN(Sequence):

    'Generates data for Keras'
    def __init__(self, dataframe, img_folder, batch_size=128, dim=(128, 128, 3), channel_IMG = 1, shuffle=False, iftest=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_folder = img_folder
        self.channel_IMG = channel_IMG
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            irr = vector_row.Irr
            for i_IMG in range(len(img_row)): # stack images in channels
                img = cv2.imread(os.path.join(self.img_folder, img_row[i_IMG]), 0)
                ghi = irr[i_IMG]
                w = int(np.floor(ghi/255.0))     # number of white pixels (val=255)
                last = np.mod(ghi, 255.0)       # value of the last pixel
                img[0][:w] = 255.0
                img[0][w] = last

                #cv2.imshow('color image', img)
                #cv2.waitKey(0)
                #cv2.imwrite(r'C:\Users\PeoPort\Downloads\img_p.png', img)

                if img is None:
                    raise ValueError(f'{img_row[i_IMG]} does not exist')
                else:
                    X[n_index, :, :, i_IMG] = img / 255.0
            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, y, file_name
        elif self.iftest == False:
            return X, y