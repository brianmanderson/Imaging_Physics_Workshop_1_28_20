import os
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import DicomImagestoData, plot_scroll_Image
import SimpleITK as sitk

def write_data(data_path, out_path, associations):
    for patient in os.listdir(data_path):
        print(patient)
        patient_data_path = os.path.join(data_path,patient)
        out_file = os.path.join(out_path,patient+'.txt')
        if not os.path.exists(out_file):
            Dicom_Reader = DicomImagestoData(path=patient_data_path,get_images_mask=True,associations=associations)
            image_handle = Dicom_Reader.dicom_handle
            Dicom_Reader.get_mask(['Liver']) # Tell the class to load up the mask with contour name 'Liver'
            mask_handle = Dicom_Reader.mask_handle
            num_images = image_handle.GetSize()[-1]
            for i in range(num_images):
                out_write_image = os.path.join(out_path, patient + '_' + str(i) + '_image.nii.gz')
                sitk.WriteImage(image_handle[:,:,i],out_write_image)
                sitk.WriteImage(mask_handle[:,:,i],out_write_image.replace('_image.','_annotation.'))
            fid = open(out_file,'w+')
            fid.close()
    return None


input_path = os.path.join('..','Data','Whole_Patients')
output_path = os.path.join('..','Data','Niftii_Arrays_2')
associations = {'Liver_BMA_Program_4':'Liver',
                'bma_liver':'Liver',
                'best_liver':'Liver',
                'tried_liver':'Liver'}
write_data(input_path,output_path,associations=associations)