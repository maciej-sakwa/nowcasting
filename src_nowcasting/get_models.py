"""Model definitions."""


from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import models
from keras import backend as K
from keras import callbacks


def SCNN(input_shape=(128, 128, 1)) -> models.Sequential:
    """Definition of a eCNN using keras Sequential API"""

    model = models.Sequential([
        Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dropout(0.2),
        Dense(256, activation='linear'),
        Dropout(0.2),
        Dense(1)
    ])

    # model.summary()
    return model


def SCNN_small_1(input_shape=(128, 128, 1)) -> models.Sequential:
    """Definition of a eCNN using keras Sequential API"""

    model = models.Sequential([
        Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dropout(0.2),
        Dense(256, activation='linear'),
        Dropout(0.2),
        Dense(1)
    ])

    # model.summary()
    return model


def SCNN_small_2(input_shape=(128, 128, 1)) -> models.Sequential:
    """Definition of a eCNN using keras Sequential API"""

    model = models.Sequential([
        Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),


        Flatten(),
        Dropout(0.2),
        Dense(256, activation='linear'),
        Dropout(0.2),
        Dense(1)
    ])

    # model.summary()
    return model



class OneCycleScheduler(callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
        
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
        
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)

