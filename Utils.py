import matplotlib.pyplot as plt
import os
# third-party imports
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend as K
import io
from PIL import Image
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Activation, Input, UpSampling2D, Concatenate, BatchNormalization, MaxPooling2D
from keras.layers import LeakyReLU


class UNet_Core():
    def __init__(self, activation='elu', alpha=0.2, input_size=(512,512,3), visualize=False,batch_normalization=True):
        self.filters = (3,3)
        self.activation = activation
        self.alpha = alpha
        self.pool_size = (2,2)
        self.input_size = input_size
        self.visualize=visualize
        self.batch_normalization = batch_normalization

    def conv_block(self,output_size,x, strides=1):
        x = Conv2D(output_size, self.filters, activation=None, padding='same',
                   name=self.desc, strides=strides)(x)
        if self.activation != 'LeakyReLU':
            x = LeakyReLU(self.alpha)(x)
        else:
            x = Activation(self.activation)(x)
        self.layer += 1
        if not self.visualize and self.batch_normalization:
            x = BatchNormalization()(x)
        return x

    def get_unet(self, layers_dict):
        self.layers_names = []
        layers = 0
        for name in layers_dict:
            if name.find('Base') != 0:
                layers += 1
        for i in range(layers):
            self.layers_names.append('Layer_' + str(i))
        self.layers_names.append('Base')
        x = input_image = Input(shape=self.input_size, name='Input')
        self.layer = 0
        layer_vals = {}
        self.desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = layers_dict[layer]['Encoding']
            for i in range(len(all_filters)):
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i) if strides == 1 else layer + '_Strided_Conv' + str(i)
                if strides == 2 and layer_index not in layer_vals:
                    layer_vals[layer_index] = x
                x = self.conv_block(all_filters[i], x, strides=strides)
                layer_vals[layer_index] = x
            layer_index += 1
            x = MaxPooling2D(name=layer + '_Max_Pooling')(x)
        if 'Base' in layers_dict:
            strides = 1
            all_filters = layers_dict['Base']['Encoding']
            for i in range(len(all_filters)):
                self.desc = 'Base_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, strides=strides)
        self.desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            layer_index -= 1
            all_filters = layers_dict[layer]['Decoding']
            x = UpSampling2D(name='Upsampling' + str(self.layer) + '_UNet')(x)
            x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[layer_index]])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x)
        x = Conv2D(2, self.filters, activation='softmax', padding='same',
                   name='output')(x)
        model = Model(inputs=input_image, outputs=x)
        self.created_model = model


class new_model(object):
    def __init__(self, layers, image_size=(256,256, 1), indexing='ij',batch_normalization=False,visualize=False):
        self.indexing = indexing
        UNet_Core_class = UNet_Core(input_size=image_size, batch_normalization=batch_normalization, visualize=visualize)
        UNet_Core_class.get_unet(layers)
        self.model = UNet_Core_class.created_model


def model_path_maker(model_path,*args):
    out_list = []
    for arg in args:
        if type(arg) is list:
            out_list += arg
        else:
            out_list.append(arg)
    for arg in out_list:
        model_path = os.path.join(model_path,arg)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    tensor = np.squeeze(tensor)
    height, width = tensor.shape
    tensor = (tensor-np.min(tensor))/(np.max(tensor)-np.min(tensor))*255
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=1,
                         encoded_image_string=image_string)


class TensorBoardImage(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch', tag='', data_generator=None):
        super().__init__(log_dir=log_dir,
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch')
        self.tag = tag
        self.log_dir = log_dir
        self.data_generator = data_generator
        if self.data_generator:
            x,_ = self.data_generator.__getitem__(0)
            _, self.rows, self.cols, _ = x.shape
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.embeddings_data = embeddings_data
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def make_image(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        tensor = np.squeeze(tensor)
        height, width = tensor.shape
        tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor)) * 255
        image = Image.fromarray(tensor.astype('uint8'))
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=1,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.data_generator:
            self.add_images(epoch)
        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {_input: embeddings_data[idx][batch]
                                     for idx, _input in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def add_images(self, epoch):
        # Load image
        self.data_generator.shuffle()
        num_images = min([3,len(self.data_generator)])
        out_image, out_mask, out_pred = np.zeros([self.rows, int(self.cols*num_images+50*(num_images-1))]), \
                                        np.zeros([self.rows, int(self.cols * num_images + 50 * (num_images - 1))]), \
                                        np.zeros([self.rows, int(self.cols * num_images + 50 * (num_images - 1))])
        step = self.cols
        for i in range(num_images):
            start = int(50*i)
            x,y = self.data_generator.__getitem__(i)
            pred = self.model.predict(x)
            out_image[:,step*i + start:step*(i+1)+start] = x[0,...,-1]
            out_mask[:, step * i + start:step * (i + 1) + start] = y[0, ..., -1]
            out_pred[:, step * i + start:step * (i + 1) + start] = pred[0, ..., -1]
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'_image', image=self.make_image(out_image))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'_mask', image=self.make_image(out_mask))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'_prediction', image=self.make_image(out_pred))])
        self.writer.add_summary(summary, epoch)
        return None


def normalize(X, lower, upper):
    X[X<lower] = lower
    X[X > upper] = upper
    X = (X - lower)/(upper - lower)
    return X


def plot_scroll_Image(x):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, x.astype('float32'))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def main():
    pass


if __name__ == '__main__':
    main()
