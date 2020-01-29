import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('..')
from Base_Deeplearning_Code.Visualizing_Model.Visualing_Model import visualization_model_class
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from tensorflow import Graph, Session, ConfigProto, GPUOptions

from Shape_Maker import Data_Generator, make_rectangle, make_circle

def prep_network():
    K.clear_session()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    K.set_session(sess)
    return None


image_size = 64

train_generator = Data_Generator(image_size=image_size,batch_size=32, num_examples_per_epoch=150)

prep_network()
num_kernels = 4
kernel_size = (3,3)
model = Sequential([
    Conv2D(num_kernels, kernel_size,
           input_shape=(image_size, image_size, 1),
           padding='same',name='Conv_0',activation='sigmoid'),
    MaxPool2D((image_size)), # Pool into a 1x1x4 image
    Flatten(),
    Dense(2,activation='softmax')
])

model.compile(Adam(lr=1e-1), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,epochs=5)

Visualizing_Class = visualization_model_class(model=model)

Visualizing_Class.define_desired_layers(desired_layer_names=['Conv_0'])

Visualizing_Class.plot_kernels()

Visualizing_Class.predict_on_tensor(make_rectangle(image_size)[None,...,None])

Visualizing_Class.plot_activations()