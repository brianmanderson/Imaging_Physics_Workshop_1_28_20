__author__ = 'Brian M Anderson'
# Created on 10/22/2019
import os
import SimpleITK as sitk
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack
from Dicom_RT_and_Images_to_Mask.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Distribute_Patients import Separate_files
from Make_Single_Images.Make_Single_Images_Class import run_main

def write_data(data_path, out_path, Dicom_Reader):
    desc = 'TCIA_Liver_Patients'
    Dicom_Reader.set_description(desc)
    iteration = 0
    for patient in os.listdir(data_path):
        print(patient)
        patient_data_path = os.path.join(data_path,patient)
        out_file = os.path.join(patient_data_path, desc + '_Iteration_' + str(iteration) + '.txt')
        if not os.path.exists(out_file):
            Dicom_Reader.Make_Contour_From_directory(patient_data_path)
            Dicom_Reader.set_iteration(iteration)
            Dicom_Reader.write_images_annotations(out_path)
        iteration += 1
    return None

associations = {'Liver_BMA_Program_4':'Liver',
                'bma_liver':'Liver',
                'best_liver':'Liver',
                'tried_liver':'Liver'}
data_path = os.path.join('..','Data','Whole_Patients')
output_path = os.path.join('..','Data','Niftii_Arrays')
Dicom_Reader = Dicom_to_Imagestack(get_images_mask=False)
# Dicom_Reader.down_folder(data_path)
# all_rois = Dicom_Reader.all_rois
Dicom_Reader.set_associations(associations)
Dicom_Reader.set_get_images_and_mask(True)
Dicom_Reader.set_contour_names(['Liver'])
Dicom_Reader.Make_Contour_From_directory(os.path.join(data_path,'ABD_LYMPH_036'))

xxx = 1
# write_data(data_path,output_path, Dicom_Reader)
# Separate_files(output_path)
run_main(output_path,extension=5)