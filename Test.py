__author__ = 'Brian M Anderson'
# Created on 10/17/2019

from Liver_Generator import Data_Generator, os, plot_scroll_Image
from Easy_VGG16_UNet.Keras_Fine_Tune_VGG_16_Liver import VGG_16
from Visualizing_Model.Visualing_Model import visualization_model_class

data_path = os.path.join('..','Data','Niftii_Arrays')
train_path = os.path.join(data_path,'Train')
validation = os.path.join(data_path,'Validation')
train_generator = Data_Generator([validation], batch_size=10, on_vgg=True, shuffle=True, by_patient=True) # mean_val=75,std_val=25
train_generator.shuffle_files()
x,y = train_generator.__getitem__(0)
network = {'Layer_0': {'Encoding': [64, 64], 'Decoding': [64, 32]},
           'Layer_1': {'Encoding': [128, 128], 'Decoding': [128]},
           'Layer_2': {'Encoding': [256, 256, 256], 'Decoding': [256]},
           'Layer_3': {'Encoding': [512, 512, 512], 'Decoding': [512]},
           'Layer_4': {'Encoding': [512, 512, 512]}}
VGG_model = VGG_16(network=network, activation='relu',filter_size=(3,3))
VGG_model.make_model()
VGG_model.load_weights()
new_model = VGG_model.created_model
Visualizing_Class = visualization_model_class(model=new_model, save_images=True)
Visualizing_Class.define_desired_layers()
Visualizing_Class.predict_on_tensor(x)
Visualizing_Class.plot_activations()
# x,y = train_generator.__getitem__(0)
xxx = 1