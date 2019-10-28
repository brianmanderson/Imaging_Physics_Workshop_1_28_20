__author__ = 'Brian M Anderson'
# Created on 10/28/2019

from keras.optimizers import Adam
from UNet_Maker import my_UNet
from Liver_Generator import Data_Generator, os, plot_scroll_Image, dice_coef_3D
from Utils import ModelCheckpoint, TensorBoardImage, model_path_maker

data_path = os.path.join('..','Data','Niftii_Arrays')
train_path = os.path.join(data_path,'Train')
validation_path = os.path.join(data_path,'Validation')
test_path = os.path.join(data_path,'Test')
model_path = os.path.join('..','Models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

args = {'batch_size':1,'on_vgg':True}#'mean_val':81,'std_val':31,
train_generator = Data_Generator(train_path, shuffle=True, **args) # mean_val=81,std_val=30
args_val = {'on_vgg':True,'mean_val':train_generator.mean_val,'std_val':train_generator.std_val,'by_patient':False,
       'shuffle':True,'batch_size':5}#'mean_val':81,'
validation_generator = Data_Generator(validation_path, **args_val)
args_val = {'on_vgg':True,'mean_val':train_generator.mean_val,'std_val':train_generator.std_val,'by_patient':True,
       'shuffle':False}
test_generator = Data_Generator(test_path, **args_val)

layers_dict = {}
conv_block = lambda x: {'Channels': [x], 'Kernel': [(3, 3)]}
pool = (4,4)
filters = 16
layers_dict['Layer_0'] = {'Encoding':[conv_block(filters), conv_block(filters)],'Pooling':(4,4),
                         'Decoding':[conv_block(filters),conv_block(filters)]}
pool = (2,2)
filters = 32
layers_dict['Layer_1'] = {'Encoding':[conv_block(filters), conv_block(filters)],'Pooling':pool,
                         'Decoding':[conv_block(filters),conv_block(filters)]}
filters = 64
layers_dict['Base'] = {'Encoding':[conv_block(filters), conv_block(filters)]}

new_model = my_UNet(layers_dict=layers_dict,image_size=512,num_channels=train_generator.channels).created_model
new_model.compile(Adam(lr=1e-6),loss='categorical_crossentropy', metrics=['accuracy',dice_coef_3D])
model_name = 'My_New_Model'
other_aspects = [model_name,'3_Layers','16_filters'] # Just a list of defining things
model_path_out = model_path_maker(model_path,other_aspects)
checkpoint = ModelCheckpoint(model_path_out, monitor='val_dice_coef_3D', verbose=1, save_best_only=True,
                              save_weights_only=False, period=5, mode='max')
# TensorboardImage lets us view the predictions of our model
tensorboard = TensorBoardImage(log_dir=model_path_out, batch_size=1,update_freq='epoch',
                               data_generator=validation_generator)
callbacks = [checkpoint, tensorboard]
new_model.fit_generator(train_generator,epochs=5, workers=20, max_queue_size=50, validation_data=validation_generator,
                       callbacks=callbacks, steps_per_epoch=10)