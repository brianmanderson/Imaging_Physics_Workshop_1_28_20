import os, sys
sys.path.append('..')
from Base_Deeplearning_Code.Data_Generators.Generators import Data_Generator_class, os
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import np, dice_coef_3D
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
from Base_Deeplearning_Code.Callbacks.Visualizing_Model_Utils import TensorBoardImage
from Utils import ModelCheckpoint, model_path_maker
# import tensorflow as tf
# import keras.backend as K
# from Base_Deeplearning_Code.Models.Keras_3D_Models import my_3D_UNet
# from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback import CyclicLR
# from functools import partial
# from keras.optimizers import Adam


base = '.'
for i in range(3):
    if 'Data' not in os.listdir(base):
        base = os.path.join(base,'..')
    else:
        break

data_path = os.path.join(base,'Data','Niftii_Arrays')
train_path = [os.path.join(data_path,'Train')]
validation_path = os.path.join(data_path,'Validation')
test_path = os.path.join(data_path,'Test')
model_path = os.path.join(base,'Models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

image_processors_train = [Ensure_Image_Proportions(512,512),Repeat_Channel(repeats=3),
                          Normalize_Images(mean_val=78,std_val=29),
                          Add_Noise_To_Images(variation=np.round(np.arange(start=0, stop=0.3, step=0.1),2)),
                          Threshold_Images(lower_bound=-3.55,upper_bound=3.55),
                          Annotations_To_Categorical(2)]
image_processors_test = [Ensure_Image_Proportions(512,512),Normalize_Images(mean_val=78,std_val=29),
                         Repeat_Channel(repeats=3), Threshold_Images(lower_bound=-3.55,upper_bound=3.55),
                         Annotations_To_Categorical(2)]
train_generator = Data_Generator_class(by_patient=False,data_paths=train_path,shuffle=True,batch_size=5,
                                       image_processors=image_processors_train)
validation_generator = Data_Generator_class(by_patient=True, whole_patient=False,data_paths=validation_path,shuffle=False,
                                      image_processors=image_processors_train, batch_size=50)
x,y = train_generator.__getitem__(0)
train_generator = Train_Data_Generator2D(shuffle=True,data_paths=train_path,batch_size=5,image_processors=image_processors_train)
validation_generator = Train_Data_Generator2D(shuffle=True,data_paths=validation_path,batch_size=5,image_processors=image_processors_test)


activation = 'relu'
kernel = (3,3)
pool_size = (2,2)

conv_block = lambda x: {'channels': x}
strided_block = lambda x: {'channels': x, 'strides': (2, 2)}

layers_dict = {}
filters = 16
layers_dict['Layer_0'] = {'Encoding':[conv_block(filters),conv_block(filters)],
                          'Decoding':[conv_block(filters),conv_block(filters)],
                          'Pooling':{}}
filters = 32
layers_dict['Layer_1'] = {'Encoding':[conv_block(filters),conv_block(filters)],
                          'Decoding':[conv_block(filters),conv_block(filters)],
                          'Pooling':{}}
filters = 64
layers_dict['Base'] = {'Encoding':[conv_block(filters), conv_block(filters)]}

K.clear_session()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
K.set_session(sess)
new_model = my_3D_UNet(kernel=kernel,layers_dict=layers_dict, pool_size=pool_size,activation=activation,is_2D=True,
                       input_size=3, image_size=512).created_model

min_lr = 1e-5
max_lr = 1e-3
new_model.compile(Adam(lr=min_lr),loss='categorical_crossentropy', metrics=['accuracy',dice_coef_3D])

model_name = 'My_New_Model'
other_aspects = [model_name,'{}_Layers'.format(3),'{}_Conv_Blocks'.format(2),
                 '{}_Filters'.format(16),'{}_MinLR_{}_MaxLR'.format(min_lr,max_lr)] # Just a list of defining things
model_path_out = model_path_maker(model_path,other_aspects)

steps_per_epoch = 100
step_size_factor = 10

checkpoint = ModelCheckpoint(os.path.join(model_path_out,'best-model.hdf5'), monitor='val_dice_coef_3D', verbose=1, save_best_only=True,
                              save_weights_only=False, period=5, mode='max')
# TensorboardImage lets us view the predictions of our model
tensorboard = TensorBoardImage(log_dir=model_path_out, batch_size=1, num_images=3,update_freq='epoch',
                               data_generator=validation_generator, image_frequency=1)
lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=steps_per_epoch * step_size_factor, mode='triangular2')
callbacks = [lrate, checkpoint, tensorboard]

x,y = train_generator.__getitem__(0)
new_model.fit_generator(train_generator,epochs=50, workers=50, max_queue_size=200, validation_data=validation_generator,
                        callbacks=callbacks, steps_per_epoch=steps_per_epoch)
