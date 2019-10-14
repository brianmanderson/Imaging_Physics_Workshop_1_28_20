from keras.utils import Sequence
import numpy as np
from skimage.draw import rectangle, circle
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, Input


def make_rectangle(image_size_x=100, image_size_y=100):
    image = np.zeros((image_size_x,image_size_y))
    origin_x, origin_y = np.random.randint(image.shape[0]), np.random.randint(image.shape[1])
    rows, cols = np.random.randint(1,image.shape[0]), np.random.randint(1,image.shape[1])
    rr, cc = rectangle((origin_x,origin_y),extent=(rows,cols),shape=image.shape)
    image[rr,cc] = 1
    return image


def make_circle(image_size_x=100, image_size_y=100):
    image = np.zeros((image_size_x,image_size_y))
    radius = np.random.randint(1,image.shape[1]//2)
    origin_x, origin_y = np.random.randint(radius,image.shape[0]-radius), np.random.randint(radius,image.shape[1]-radius)
    rr, cc = circle(origin_x,origin_y,radius,shape=image.shape)
    image[rr,cc] = 1
    return image


class Data_Generator(Sequence):
    def __init__(self, image_size=100, batch_size=10):
        self.image_size = image_size
        self.batch_size = batch_size

    def return_rectangle(self):
        return make_rectangle(self.image_size,self.image_size)

    def return_circle(self):
        return make_circle(self.image_size,self.image_size)

    def __getitem__(self, item):
        output = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
        y = np.zeros((self.batch_size,2))
        for i in range(self.batch_size):
            if np.random.randint(2) == 1:
                output[i] = self.return_rectangle()[...,None]
                y[i,0] = 1
            else:
                output[i] = self.return_circle()[...,None]
                y[i,1] = 1
        return output, y

if __name__ == '__main__':
    # image_input_primary = x = Input(shape=(100, 100, 1), name='UNet_Input')
    # x = Conv2D(6, (3,3), padding='same')(x)
    # x = MaxPool2D(52)(x)
    #
    # model = Sequential([
    #     Conv2D(6, (3,3), input_shape=(100, 100, 1), padding='same'), # Make 6 kernels
    #     MaxPool2D(52), # Pool into a 1x1x6 image
    #     Flatten(),
    #     Dense(2),
    #     Activation('softmax')
    # ])
    xxx = 1