from  __future__ import absolute_import
from __future__ import print_function


import os
import shutil

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import BaseLogger, TensorBoard, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras import objectives
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from skimage.transform import resize
from keras.layers import Embedding
import keras.optimizers
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from ImageDataGeneratorCustom import ImageDataGeneratorCustom

def convnet_model_():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
 
    convnet_model = convnet_model_()
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model

deep_rank_model = deep_rank_model()

for layer in deep_rank_model.layers:
    print (layer.name, layer.output_shape)

model_path = "./deep_ranking"

class DataGenerator(object):
    def __init__(self, params, target_size=(224, 224)):
        self.params = params
        self.target_size = target_size
        self.idg = ImageDataGeneratorCustom(**params)

    def get_train_generator(self, batch_size):
        return self.idg.flow_from_directory("/home/shared/CS341/Dataprocessing/train",
                                            batch_size=batch_size,
                                            target_size=self.target_size,shuffle=False,
                                            triplet_path  ='/home/shared/CS341/Dataprocessing/triplets_new.txt'
                                           )

    def get_test_generator(self, batch_size):
        return self.idg.flow_from_directory("/home/shared/CS341/Dataprocessing/finaltest/train",
                                            batch_size=batch_size,
                                            target_size=self.target_size, shuffle=False,
                                            triplet_path  ='/home/shared/CS341/Dataprocessing/triplets_finaltest.txt'
                                        )





dg = DataGenerator({
    "rescale": 1. / 255,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.2,
    "shear_range": 0.2,
    "rotation_range": 30,
"fill_mode": 'nearest' 
}, target_size=(224, 224))

batch_size = 8 
batch_size *= 3
train_generator = dg.get_train_generator(batch_size)

_EPSILON = K.epsilon()
def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    loss =  tf.convert_to_tensor(0,dtype=tf.float32)
    g = tf.constant(1.0, shape=[1], dtype=tf.float32)
    for i in range(0,batch_size,3):
        try:
            q_embedding = y_pred[i+0]
            p_embedding =  y_pred[i+1]
            n_embedding = y_pred[i+2]
            D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
            loss = (loss + g + D_q_p - D_q_n )            
        except:
            continue
    loss = loss/(batch_size/3)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    return tf.maximum(loss,zero)

deep_rank_model.load_weights('deepranking.h5')
deep_rank_model.compile(loss=_loss_tensor, optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))

train_steps_per_epoch = int((28979)/batch_size)
train_epocs = 4
deep_rank_model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=train_epocs
                        )

model_path = "deepranking_526.h5"
deep_rank_model.save_weights(model_path)


