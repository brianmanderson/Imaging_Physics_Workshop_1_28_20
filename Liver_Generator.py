__author__ = 'Brian M Anderson'
# Created on 10/16/2019

from keras.utils import Sequence
import SimpleITK as sitk
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
    def __init__(self, data_paths, batch_size=10, shuffle=False, mean_val=0, std_val=1, channels=1, on_vgg=False,
                 classes=2, by_patient=False):
        self.by_patient = by_patient
        if type(data_paths) != list:
            data_paths = [data_paths]
        self.on_vgg = on_vgg
        self.vgg_mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        self.classes = classes
        if on_vgg:
            channels = 3
        self.channels = channels
        self.mean_val = mean_val
        self.std_val = std_val
        self.image_dictionary = {}
        self.load_list = []
        self.patient_dict = {}
        self.file_list = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter_fuction = lambda x: int(x.split('_')[-2].split('image')[0])
        for data_path in data_paths:
            self.load_all_data(data_path)
        if by_patient:
            self.file_list = list(self.patient_dict.keys())
        self.file_list = np.asarray(self.file_list)
        self.distribute_images()

    def load_all_data(self, data_path):
        file_list = os.listdir(data_path)
        file_list = [i for i in file_list if i.find('_image.nii.gz') != -1]
        for file in file_list:
            load_path = os.path.join(data_path,file)
            split_up = file.split('_')
            desc = ''.join(split_up[:-2])
            if desc not in self.patient_dict:
                self.patient_dict[desc] = [load_path]
            else:
                self.patient_dict[desc].append(load_path)

        for key in self.patient_dict.keys(): # Sort the images properly
            self.patient_dict[key].sort(key=self.filter_fuction)
            self.file_list += self.patient_dict[key]
    def load_images(self, file_paths):
        temp_images, temp_annotations = [], []
        for load_path in file_paths:
            image_handle = sitk.ReadImage(load_path)
            annotation_handle = sitk.ReadImage(load_path.replace('_image.','_annotation.'))
            images = sitk.GetArrayFromImage(image_handle)
            images = np.stack([images for _ in range(self.channels)], axis=-1)
            annotations = sitk.GetArrayFromImage(annotation_handle)
            annotations = np_utils.to_categorical(annotations,self.classes)
            temp_images.append(images)
            temp_annotations.append(annotations)
        return np.asarray(temp_images), np.asarray(temp_annotations)

    def shuffle_files(self):
        np.random.shuffle(self.file_list)

    def distribute_images(self):
        if self.shuffle:
            self.shuffle_files()
        self.load_list = []
        if self.by_patient:
            for desc in self.file_list:
                self.load_list.append(self.patient_dict[desc])
        else:
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
        out_images, out_annotations = self.load_images(self.load_list[item])
        if self.mean_val != 0 or self.std_val != 1:
            out_images = (out_images - self.mean_val)/self.std_val
            out_images[out_images<-5] = -5
            out_images[out_images>5] = 5
        if self.on_vgg:
            if self.mean_val != 0 and self.std_val != 1:
                out_images = (out_images+5)/10 * 255 # bring to 0-255 range
            else:
                out_images = (out_images + 1000) / 2000 * 255  # bring to 0-255 range
                out_images[out_images<0] = 0
                out_images[out_images>255] = 255
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