__author__ = 'Brian M Anderson'
# Created on 10/16/2019

from keras.utils import Sequence
import nibabel as nib
import numpy as np
import os
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import plot_scroll_Image
from keras.utils import np_utils
import keras.backend as K


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[...,1:] * y_pred[...,1:])
    union = K.sum(y_true[...,1:]) + K.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)


class Data_Generator(Sequence):
    def __init__(self, data_path, batch_size=10, shuffle=False, mean_val=0, std_val=1, channels=1, on_vgg=False):
        self.on_vgg = on_vgg
        self.vgg_mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        if on_vgg:
            channels = 3
        self.data_path = data_path
        self.channels = channels
        self.mean_val = mean_val
        self.std_val = std_val
        self.image_dictionary = {}
        self.load_list = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_list = np.asarray([i for i in os.listdir(data_path) if i.find('.nii.gz') != -1])
        self.file_list.sort()
        self.distribute_images()

    def load_image(self, file):
        data = nib.load(os.path.join(self.data_path,file))
        data = data.get_fdata() # Data should be of shape [2, # images, # images, 1]
        images = np.stack([data[0][...,0] for _ in range(self.channels)],axis=-1)
        annotation = data[1]
        annotation = np_utils.to_categorical(annotation,2)
        self.image_dictionary[file] = [images[None,...],annotation[None,...]]

    def shuffle_files(self):
        np.random.shuffle(self.file_list)

    def distribute_images(self):
        if self.shuffle:
            self.shuffle_files()
        self.load_list = []
        ii = 0
        batch = []
        for i in range(len(self.file_list)):
            if ii < self.batch_size:
                ii += 1
                batch.append(self.file_list[i])
            else:
                if len(batch) == self.batch_size:
                    self.load_list.append(batch)
                    batch = []
                    ii = 0

    def __getitem__(self, item):
        out_images, out_annotations = np.zeros([self.batch_size,512,512,self.channels]), np.zeros([self.batch_size,512,512,2])
        for i, file in enumerate(self.load_list[item]):
            if file not in self.image_dictionary:
                self.load_image(file)
            out_images[i], out_annotations[i] = self.image_dictionary[file]
        if self.mean_val != 0 or self.std_val != 1:
            out_images = (out_images - self.mean_val)/self.std_val
            out_images[out_images<-5] = -5
            out_images[out_images>5] = 5
        if self.on_vgg:
            out_images = (out_images+5)/10 * 255 # bring to 0-255 range
            out_images -= self.vgg_mean # normalize across vgg
        return out_images, out_annotations

    def __len__(self):
        return len(self.load_list)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_files()

    def get_mean_std_val(self):
        self.mean_val, self.std_val = 0, 1
        reset_vgg = self.on_vgg
        self.on_vgg = False
        for i in range(self.__len__()):
            print('Calculating mean and std...{}% done'.format(str(round(i/self.__len__()*100))))
            x, y = self.__getitem__(i)
            data = x[y[..., 0] == 0][..., 0]
            if i == 0:
                output = data
            else:
                output = np.append(output, data, axis=0)
        self.mean_val = round(np.mean(output, axis=0))
        self.std_val = round(np.std(output,axis=0))
        self.on_vgg = reset_vgg
        print('Mean is {}, and std is {}'.format(str(self.mean_val),str(self.std_val)))



def main():
    pass

if __name__ == '__main__':
    xxx = 1