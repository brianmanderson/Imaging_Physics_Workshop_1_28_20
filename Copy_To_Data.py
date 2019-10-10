import os
from Copy_Folders.Copy_Folder_To_Another import down_copy

def down_folder(input_path, output=None):
    files = None
    dirs = None
    for root, dirs, files in os.walk(input_path):
        break
    if files:
        return root
    else:
        for dir in dirs:
            output = down_folder(os.path.join(root,dir))
        return output


out_path_base = os.path.join('..','Data')

paths = [r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\TCIA_For_NCAAPM\Patients\CT Lymph Nodes',r'K:\Morfeus\Auto_Contour_Sites\Liver_Auto_Contour\Output']
for path in paths:
    file_list = [i for i in os.listdir(path) if i.find('LYMPH') != -1]
    i = 0
    for file in file_list:
        if not os.path.exists(os.path.join(out_path_base,file)):
            os.makedirs(os.path.join(out_path_base,file))
        output = down_folder(os.path.join(path,file))
        print(file)
        down_copy(input_path=output,output_path=os.path.join(out_path_base,file), file_criteria=lambda x: x.find('.dcm') != -1)
        i += 1
        if i > 19 and path.find('TCIA') != -1:
            break
