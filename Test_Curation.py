__author__ = 'Brian M Anderson'
# Created on 10/22/2019
import os
import SimpleITK as sitk
import numpy as np
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import DicomImagestoData, plot_scroll_Image

def write_data(data_path, out_path, associations):
    for patient in os.listdir(data_path):
        print(patient)
        patient_data_path = os.path.join(data_path,patient)
        out_file = os.path.join(out_path,patient+'.txt')
        if not os.path.exists(out_file) or True:
            Dicom_Reader = DicomImagestoData(path=patient_data_path,get_images_mask=False,associations=associations)
            image_handle = Dicom_Reader.dicom_handle
            Dicom_Reader.get_mask(['Liver']) # Tell the class to load up the mask with contour name 'Liver'
            mask_handle = Dicom_Reader.mask_handle
            out_write_image = os.path.join(out_path, patient + '_image.nii.gz')
            sitk.WriteImage(image_handle,out_write_image)
            sitk.WriteImage(mask_handle,out_write_image.replace('_image.','_annotation.'))
            fid = open(out_file,'w+')
            fid.close()

associations = {'Liver_BMA_Program_4':'Liver',
                'bma_liver':'Liver',
                'best_liver':'Liver',
                'tried_liver':'Liver'}
input_path = os.path.join('..','Data','Whole_Patients')
output_path = os.path.join('..','Data','Niftii_Arrays')
write_data(input_path,output_path, associations)