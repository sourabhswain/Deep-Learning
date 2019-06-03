from keras.applications import resnet50
import os
import glob
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.utils import to_categorical
from keras import optimizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_dir = "imagesTest"
n_train = 6252
num_train = [2120, 916, 832, 1167, 1217]
n_classes = 5
n_test = 2680
batch_size = 32
epochs = 75



def initiate():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255)

    # build the VGG16 network
    img_width = 200
    img_height = 200
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)



    #n_filters1 = 32
    #n_filters2 = 64
    #pool_size = 2

    pretrained_model = applications.VGG16(
        include_top=False,
        input_shape=(img_width, img_height, 3),
        weights='imagenet'
    )


    output = Flatten()(pretrained_model.output)

    #model = Model(pretrained_model.input, output)

    for layer in pretrained_model.layers[:-3]:
        layer.trainable = False

    #vgg_out = model.predict_generator(train_generator, 4999)

    #vgg_out = BatchNormalization()(vgg_out)
    vgg_out = Dropout(0.5)(output)
    vgg_out = Dense(256, activation='relu')(vgg_out)
    vgg_out = BatchNormalization()(vgg_out) #Worse performance. no learning at all.
    vgg_out = Dropout(0.5)(vgg_out)
    vgg_out = Dense(5, activation='softmax')(vgg_out)

    model_fc = Model(pretrained_model.input, vgg_out)



    model_fc.summary(line_length=200)

    lr = 0.001
    model_fc.compile(optimizer=optimizers.sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])



    #model = Sequential()
    #model.add(Convolution2D(n_filters1, kernel_size=(3,3), padding="same", input_shape=(img_width, img_height, 3)))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    #model.add(Convolution2D(n_filters2, kernel_size=(3,3), padding="same" ))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D( pool_size = (pool_size, pool_size)))

    #model.add(Flatten())
    #model.add(Dense(256))
    #model.add(Activation("relu"))
    #model.add(Dropout(0.5))
    #model.add(Dense(5, activation="softmax"))


    samples_per_epoch = 1000
    validation_steps = 1253 // 64



    model_fc.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps = np.ceil(1253 / batch_size),
        shuffle=True
        )

    target_dir = './models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model_fc.save('./models/model_cnn.h5')
    model_fc.save_weights('./models/weights_cnn.h5')


initiate()
