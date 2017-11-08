from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import RMSprop

# Almost a LeNet: http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html
def almost_LeNet(in_shape=(1,28,28), n_classes=10):
    model = Sequential([
        Convolution2D(6, 5, 5, border_mode='valid', input_shape=in_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Activation('sigmoid'),

        Convolution2D(16, 5, 5, border_mode='valid'),
        MaxPooling2D(pool_size=(2, 2)),
        Activation('sigmoid'),
        Dropout(0.5),

        Convolution2D(120, 1, 1, border_mode='valid'),

        Flatten(),
        Dense(84),
        Activation('sigmoid'),
        Dense(n_classes),
        Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def all_conv_net_ref_a(in_shape, n_classes):
    model = Sequential([
        Convolution2D(96, 5, 5, border_mode='valid', input_shape = in_shape),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
        Activation("relu"),
        Dropout(0.5),

        Convolution2D(192, 5, 5, border_mode='valid'),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
        Activation("relu"),
        Dropout(0.5),

        Convolution2D(192, 3, 3, border_mode='valid'),
        Activation("relu"),
        Convolution2D(192, 1, 1, border_mode='valid'),
        Activation("relu"),
        Convolution2D(n_classes, 1, 1, border_mode='valid'),
        Activation("relu"),
        Flatten(),

        Dense(10),
        Activation('softmax')
    ])

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    #model.summary()

    return model

def all_conv_net_ref_c(in_shape, n_classes):
    model = Sequential([
        Convolution2D(96, 3, 3, border_mode='valid', input_shape = in_shape),
        Activation("relu"),
        Convolution2D(96, 3, 3, border_mode='valid'),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
        Activation("relu"),
        Dropout(0.5),

        Convolution2D(96, 3, 3, border_mode='valid', input_shape = in_shape),
        Activation("relu"),
        Convolution2D(96, 3, 3, border_mode='valid'),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
        Activation("relu"),
        Dropout(0.5),

        Convolution2D(192, 3, 3, border_mode='valid'),
        Activation("relu"),
        Convolution2D(192, 1, 1, border_mode='valid'),
        Activation("relu"),
        Convolution2D(n_classes, 1, 1, border_mode='valid'),
        Activation("relu"),
        Flatten(),

        Dense(10),
        Activation('softmax')
    ])

    #rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #model.summary()

    return model
