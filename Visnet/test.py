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
import keras.optimizers
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

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from ImageDataGeneratorCustom import ImageDataGeneratorCustom

def convnet_model_():
    vgg_model = VGG16(weights='imagenet', include_top=False)
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

deep_rank_model.load_weights('deepranking.h5')

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding

test_generator = dg.get_test_generator(3)

count = 0
for j in range(0,1488,3):
    x,y = test_generator.next()
    y_pred = deep_rank_model.predict(x)
    q_embedding = y_pred[0]
    p_embedding =  y_pred[1]
    n_embedding = y_pred[2]
    distance1 = sum([(q_embedding[idx] - p_embedding[idx])**2 for idx in range(len(q_embedding))])
    distance2 = sum([(q_embedding[idx] - n_embedding[idx])**2 for idx in range(len(q_embedding))])
    if distance1 < distance2:
        count+=1
print (count/496.0)

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files]

import matplotlib.pyplot as plt
directory_path = '/home/shared/CS341/Dataprocessing/finaltest/train'
classes = [d for d in os.listdir('/home/shared/CS341/Dataprocessing/finaltest/train') if os.path.isdir(os.path.join('/home/shared/CS341/Dataprocessing/finaltest/train', d))]
all_images = []
for class_ in classes:
    all_images += (list_pictures(os.path.join('/home/shared/CS341/Dataprocessing/finaltest/train',class_)))
triplets = []
query_image = '/home/shared/CS341/Dataprocessing/finaltest/train/cats/cats-1019144.jpg'         
image1 = load_img(query_image)
image1 = img_to_array(image1).astype("float64")
image1 = transform.resize(image1, (224, 224))
image1 *= 1. / 255
image1 = np.expand_dims(image1, axis = 0)
embedding1 = deep_rank_model.predict([image1, image1, image1])[0]
distances = {}
for class_ in classes:
        image_names = list_pictures(os.path.join(directory_path,class_))
        for image_name in image_names: 
            image2 = load_img(image_name)
            image2 = img_to_array(image2).astype("float64")
            image2 = transform.resize(image2, (224, 224))
            image2 *= 1. / 255
            image2 = np.expand_dims(image2, axis = 0)

            embedding2 = deep_rank_model.predict([image2,image2,image2])[0]

            distance = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])**(0.5)
            distances[image_name] = distance
count_= 0
import operator
sorted_d = sorted(distances.items(), key=operator.itemgetter(1))


import numpy as np
import random
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
fig=plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
transform=transforms.Compose([transforms.Scale((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
for i in range(10):
    if i == 0:
        im1 = sorted_d[i][0]
        a1 = Image.open(im1)
        a1 = transform(a1).view(1,3,100,100)
    else:
        im = sorted_d[i][0]
        a = Image.open(im)
        a = transform(a).view(1,3,100,100)
        a1 = torch.cat((a1,a), 0)
    plt.text(i*100+15,-2, "{0:.2f}".format(sorted_d[i][1]), style='italic',fontweight='bold')
img = torchvision.utils.make_grid(a1, nrow=10)

img = img.numpy().transpose((1,2,0))
plt.imshow(img)
