"""Model definitions."""


from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Softmax
from keras import models
from keras.applications.resnet import ResNet50


def forecastnet(input_img_size: int, n_channels: int, n_classes: int) -> models.Sequential:
    """Returns ForecastNet as a Keras model_type."""

    # Initialization.
    model = models.Sequential()

    # Add layers.
    model.add(Conv2D(
        filters=32,
        kernel_size=4,
        input_shape=(input_img_size, input_img_size, n_channels),
        strides=1,
        activation='relu'))

    model.add(MaxPooling2D(pool_size=(4, 4),
                           strides=4))

    model.add(Conv2D(
        8,
        kernel_size=10,
        activation='relu'
    ))

    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())

    model.add(Dense(n_classes, activation='softmax'))

    return model


def SCNN_diff(input_shape=(128, 128, 1)) -> models.Sequential:
    model = models.Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    # model.summary()
    return model


def SCNN(input_shape=(128, 128, 1)) -> models.Sequential:
    model = models.Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    # model.summary()
    return model


def scnn_classes(input_shape=(128, 128, 1), n_classes: int = 4) -> models.Sequential:
    model = models.Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='relu'))
    model.add(Softmax())
    return model
