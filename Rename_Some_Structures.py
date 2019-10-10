import os
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import DicomImagestoData

base_path = r'K:\Morfeus\BMAnderson\NCC_AAPM\Data\Whole_Patients'

all_assocations = [{'Liver_BMA_Program_4': 'Liver'},{'Liver_BMA_Program_4': 'bma_liver'},{'Liver_BMA_Program_4': 'best_liver'},
                   {'Liver_BMA_Program_4': 'tried_liver'}]
for i, folder in enumerate(os.listdir(base_path)):
    print(folder)
    path = os.path.join(base_path,folder)
    DicomImage = DicomImagestoData(path=path, associations=all_assocations[i], get_images_mask=False)
    DicomImage.rewrite_RT(DicomImage.lstRSFile)