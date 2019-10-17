__author__ = 'Brian M Anderson'
# Created on 10/17/2019

from Liver_Generator import Data_Generator, os, plot_scroll_Image

data_path = os.path.join('..','Data','Numpy_Arrays')
train_path = os.path.join(data_path,'Train')
validation = os.path.join(data_path,'Validation')
train_generator = Data_Generator(train_path, batch_size=10, on_vgg=True, shuffle=False) # mean_val=81,std_val=30
# x,y = train_generator.__getitem__(0)
train_generator.get_mean_std_val() # This will calculate the mean and std for you
xxx = 1