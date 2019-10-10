import os
import numpy as np
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import DicomImagestoData

base_path = r'K:\Morfeus\BMAnderson\NCC_AAPM\Data\Whole_Patients'

Contour_Names = ['Liver']

associations = {'Liver_BMA_Program_4':'Liver','bma_liver':'Liver','best_liver':'Liver','tried_liver':'Liver'}
for folder in os.listdir(base_path):
    print(folder)
    path = os.path.join(base_path,folder)
    DicomImage = DicomImagestoData(path=path, associations=associations)
    DicomImage.get_mask(Contour_Names)
    mask = DicomImage.mask
    max_vals = np.max(mask,axis=(1,2,3))
    z_images = np.where(max_vals==1)[0]
    for i in range(len(DicomImage.lstFilesDCM)):
        if i not in z_images:
            os.remove(DicomImage.lstFilesDCM[i])
    xxx = 1