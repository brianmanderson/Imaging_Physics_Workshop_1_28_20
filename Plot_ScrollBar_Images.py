__author__ = 'Brian M Anderson'
# Created on 1/29/2020

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from ipywidgets import interactive, IntSlider


class scroll_bar_class(object):
    def __init__(self,numpy_images):
        images = np.squeeze(numpy_images)
        if images.shape[-1] == 3:
            images = images[...,1]
        if len(images.shape) == 4:
            images = np.argmax(images,axis=-1)
        self.selected_images = sitk.GetImageFromArray(images,isVector=False)
        self.size = self.selected_images.GetSize()
    def custom_myshow1(self,img, title=None, margin=0.05, dpi=80 ):
        nda = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()
        if nda.ndim == 3:
            # fastest dim, either component or x
            c = nda.shape[-1]

            # the the number of components is 3 or 4 consider it an RGB image
            if not c in (3,4):
                nda = nda[nda.shape[0]//2,:,:]

        elif nda.ndim == 4:
            c = nda.shape[-1]

            if not c in (3,4):
                raise Runtime("Unable to show 3D-vector Image")

            # take a z-slice
            nda = nda[nda.shape[0]//2,:,:,:]

        ysize = nda.shape[1]
        xsize = nda.shape[1]

        # Make a figure big enough to accomodate an axis of xpixels by ypixels
        # as well as the ticklabels, etc...

        plt.close('all')
        plt.clf()

        figsize = ((1 + margin) * ysize / dpi)*2, (1 + margin) * xsize / dpi
        fig, ax = plt.subplots(1, 1, figsize= figsize)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        t = ax.imshow(nda,cmap='gray',interpolation=None)
        ax.set_title('use scroll bar to navigate images')
        plt.show()

    def update(self,Z, view='2D'):
        if view == '2D':
            slices =[self.selected_images[:,:,Z]]
            dpi = 50
        self.custom_myshow1(sitk.Tile(slices, [3,1]),dpi=dpi, margin = 0.05)


def plot_Image_Scroll_Bar_Image(x):
    k = scroll_bar_class(x)
    interactive_plot = interactive(k.update,Z=IntSlider(min=0,max=x.shape[0]-1))
    output = interactive_plot.children[-1]
    output.layout.height = '600px'
    return interactive_plot

if __name__ == '__main__':
    pass
