__author__ = 'Brian M Anderson'
# Created on 10/16/2019

from keras.utils import Sequence
import nibabel as nib
import numpy as np
import os
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import plot_scroll_Image
from keras.utils import np_utils

class Data_Generator(Sequence):
    def __init__(self, data_path, batch_size=10, shuffle=True, mean_val=0, std_val=1):
        self.mean_val = mean_val
        self.std_val = std_val
        self.image_dictionary = {}
        self.load_list = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_all_images(data_path)
        self.file_list = np.asarray(list(self.image_dictionary.keys()))
        self.distribute_images()

    def load_all_images(self, data_path):
        files = [i for i in os.listdir(data_path) if i.find('.nii.gz') != -1]
        for ii, file in enumerate(files):
            data = nib.load(os.path.join(data_path,file))
            data = data.get_fdata() # Data should be of shape [2, # images, # images, 1]
            images = data[0]
            annotation = data[1]
            annotation = np_utils.to_categorical(annotation,2)
            for i in range(images.shape[0]):
                self.image_dictionary[file+'_' + str(i)] = [images[i][None,...],annotation[i][None,...]]
            print('Loading {}% done'.format(str(round((ii+1)/len(files)*100))))


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
        out_images, out_annotations = np.zeros([self.batch_size,512,512,1]), np.zeros([self.batch_size,512,512,2])
        for i, file in enumerate(self.load_list[item]):
            out_images[i], out_annotations[i] = self.image_dictionary[file]
        if self.mean_val != 0 or self.std_val != 1:
            out_images = (out_images - self.mean_val)/self.std_val
            out_images[out_images<-5] = -5
            out_images[out_images>5] = 5
        return out_images, out_annotations

    def __len__(self):
        return len(self.load_list)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_files()

    def get_mean_std_val(self):
        self.mean_val, self.std_val = 0, 1
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
        print('Mean is {}, and std is {}'.format(str(self.mean_val),str(self.std_val)))



def main():
    pass

if __name__ == '__main__':
    xxx = 1