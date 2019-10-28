from nbmerge import merge_notebooks, write_notebook
import io, os

base_path = r'K:\Morfeus\BMAnderson\NCC_AAPM\Code'
file_names = ['Download_Data.ipynb','DeepBox.ipynb','Data_Curation.ipynb','Liver_Model.ipynb']
file_paths = [os.path.join(base_path,i) for i in file_names]
merged = merge_notebooks(base_dir=base_path,file_paths=file_paths, verbose=True)
with io.open(os.path.join(base_path,'Click_Me.ipynb'), 'w', encoding='utf8') as fp:
    write_notebook(merged, fp)

