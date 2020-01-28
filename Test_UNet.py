__author__ = 'Brian M Anderson'
# Created on 10/28/2019

import os, sys
sys.path.append('..')
from Base_Deeplearning_Code.Data_Generators.Generators import Train_Data_Generator2D, os
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import np, dice_coef_3D
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
from Base_Deeplearning_Code.Callbacks.Visualizing_Model_Utils import TensorBoardImage
from Base_Deeplearning_Code.Models.Keras_3D_Models import my_3D_UNet
from Utils import ModelCheckpoint, model_path_maker
import tensorflow as tf
import keras.backend as K

def get_layers_dict(layers=1, filters=16, conv_blocks=1, num_atrous_blocks=4, max_blocks=2, max_filters=np.inf,
                    atrous_rate=1, max_atrous_rate=2, **kwargs):
    activation = {'activation':PReLU,'kwargs':{'alpha_initializer':Constant(0.25),'shared_axes':[1,2,3]}}
    activation = 'relu'
    layers_dict = {}
    conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (1, 1, 1),'activation':activation}}
    strided_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (2, 2, 2), 'activation':activation}}
    atrous_block = lambda x,y,z: {'atrous': {'channels': x, 'atrous_rate': y, 'activations': z}}
    for layer in range(conv_blocks,layers-1):
        encoding = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        atrous_block_dec = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        if layer == 0:
            encoding = [conv_block(filters)] + encoding
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)] = {'Encoding': encoding,
                                              'Pooling':{'Encoding':[strided_block(filters)],'Pool_Size':(2,2,2)},
                                              'Decoding': atrous_block_dec}
        num_atrous_blocks = min([(num_atrous_blocks) * 2,max_blocks])
    num_atrous_blocks = min([(num_atrous_blocks) * 2, max_blocks])
    layers_dict['Base'] = {'Encoding':[atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]}
    return layers_dict
layers_dict = {}
filters = 16
activation = 'relu'
kernel = (3,3)
pooling = (2,2)
from Base_Deeplearning_Code.Models.Keras_3D_Models import my_3D_UNet
from functools import partial
conv_block = lambda x: {'convolution': {'channels': x, 'kernel': kernel, 'strides': (1, 1),'activation':activation}}
strided_block = lambda x: {'convolution': {'channels': x, 'kernel': kernel, 'strides': (2, 2),'activation':activation}}
conv_block = lambda x: {'channels':x}
strided_block = lambda x: {'channels':x,'strides':(2,2)}
layers_dict['Layer_0'] = {'Encoding':[conv_block(filters),conv_block(filters)],
                          'Decoding':[conv_block(filters),conv_block(filters)],
                          'Pooling':{'Encoding':[strided_block(filters)]}}
filters = 32
layers_dict['Layer_1'] = {'Encoding':[conv_block(filters),conv_block(filters)],
                          'Decoding':[conv_block(filters),conv_block(filters)],
                          'Pooling':{'Encoding':[strided_block(filters)]}}
filters = 64
layers_dict['Base'] = {'Encoding':[conv_block(filters), conv_block(filters)]}
new_model = my_3D_UNet(kernel=kernel,layers_dict=layers_dict, pool_size=(2,2),activation=activation,is_2D=True,
                       input_size=3, image_size=512)
data_path = os.path.join('..','Data','Niftii_Arrays')
train_path = os.path.join(data_path,'Train')
validation_path = os.path.join(data_path,'Validation')
test_path = os.path.join(data_path,'Test')
model_path = os.path.join('..','Models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

args = {'batch_size':1,'on_vgg':False, 'mean_val':81,'std_val':30}#'mean_val':81,'std_val':31,
train_generator = Data_Generator(train_path, shuffle=True, **args) # mean_val=81,std_val=30
args_val = {'on_vgg':False,'mean_val':train_generator.mean_val,'std_val':train_generator.std_val,'by_patient':False,
       'shuffle':True,'batch_size':5}#'mean_val':81,'
validation_generator = Data_Generator(validation_path, **args_val)
args_val = {'on_vgg':False,'mean_val':train_generator.mean_val,'std_val':train_generator.std_val,'by_patient':True,
       'shuffle':False}
test_generator = Data_Generator(test_path, **args_val)

layers_dict = {}
conv_block = lambda x: {'Channels': [x], 'Kernel': [(3, 3)]}
pool = (4,4)
filters = 16
layers_dict['Layer_0'] = {'Encoding':[conv_block(filters), conv_block(filters)],'Pooling':pool,
                         'Decoding':[conv_block(filters),conv_block(filters)]}
pool = (2,2)
filters = 32
layers_dict['Layer_1'] = {'Encoding':[conv_block(filters), conv_block(filters)],'Pooling':pool,
                         'Decoding':[conv_block(filters),conv_block(filters)]}
filters = 64
layers_dict['Base'] = {'Encoding':[conv_block(filters), conv_block(filters)]}

model_name = 'My_New_Model'
other_aspects = [model_name,'3_Layers','16_filters'] # Just a list of defining things
model_path_out = model_path_maker(model_path,other_aspects)
checkpoint = ModelCheckpoint(os.path.join(model_path_out,'best-mode.hdf5'), monitor='val_dice_coef_3D', verbose=1, save_best_only=True,
                              save_weights_only=False, period=5, mode='max')
# TensorboardImage lets us view the predictions of our model
tensorboard = TensorBoardImage(log_dir=model_path_out, batch_size=1,update_freq='epoch',
                               data_generator=validation_generator)
callbacks = [checkpoint, tensorboard]
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
K.set_session(sess)
new_model = my_UNet(layers_dict=layers_dict, image_size=512, num_channels=train_generator.channels).created_model
new_model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy', dice_coef_3D])
new_model.fit_generator(train_generator,epochs=30, workers=20, max_queue_size=50, validation_data=validation_generator,
                       callbacks=callbacks, steps_per_epoch=100)
K.clear_session()