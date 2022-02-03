from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, GlobalAveragePooling2D, MaxPool2D, UpSampling2D,Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import applications as efn
from tensorflow import keras

def convolutional_neural_network(UP_1, UP_2, SIZE_1, SIZE_2, num_classes=2):
    """
    Keras model with trander layer
    """

    base_model = efn.DenseNet169(weights='imagenet', include_top=False, input_shape=(SIZE_1, SIZE_2, 3))
    i = 0
    for layer in base_model.layers:
        if i < 100:
            layer.trainable = False
        else:
            layer.trainable = True
        i = i + 1
         # if isinstance(layer, BatchNormalization):
        #     layer.trainable = True
        # else:
        #     layer.trainable = False

    model = Sequential()
    model.add(UpSampling2D(size=(UP_1, UP_2)))

    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def simple_cnn(num_classes=2):
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                            input_shape=(3, 3, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1)),
        # keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        # keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        # keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(250, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model