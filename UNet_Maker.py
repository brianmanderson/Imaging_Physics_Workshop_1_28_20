__author__ = 'Brian M Anderson'
# Created on 10/28/2019
from keras.backend import variable
from keras.layers import *
import keras.backend as K
import numpy as np
from keras.models import Model


ExpandDimension = lambda axis: Lambda(lambda x: K.expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: K.squeeze(x, axis))
Subtract_new = lambda y: Lambda(lambda x: Subtract()([x,y]))
Multipy_new = lambda y: Lambda(lambda x: Multiply()([x,y]))

class Unet(object):

    def __init__(self, save_memory=False):
        self.previous_conv = None
        self.save_memory = save_memory

    def define_res_block(self, do_res_block=False):
        self.do_res_block = do_res_block

    def define_unet_dict(self, layers_dict):
        self.layers_dict = layers_dict
        self.layers_names = []
        layers = 0
        for name in layers_dict:
            if name.find('Layer') == 0:
                layers += 1
        for i in range(layers):
            self.layers_names.append('Layer_' + str(i))
        if 'Base' in layers_dict:
            self.layers_names.append('Base')
        return None

    def define_fc_dict(self, layers_dict_FC):
        self.layers_dict_FC = layers_dict_FC
        self.layer_names_fc = []
        layers = 0
        for name in layers_dict_FC:
            if name.find('Final') != 0:
                layers += 1
        for i in range(layers):
            self.layer_names_fc.append('Layer_' + str(i))
        if 'Final' in layers_dict_FC:
            self.layer_names_fc.append('Final')
        return None

    def FC_Block(self, output_size, x, dropout=0.0, name=''):
        for i in range(1):
            x = Dense(output_size, activation=self.activation,
                      name=name)(x)
            if dropout != 0.0:
                x = Dropout(dropout)(x)
        return x

    def run_FC_block(self, x, all_connections_list, name=''):
        variable_dropout = None
        if 'Connections' in all_connections_list:
            all_connections = all_connections_list['Connections']
            if 'Dropout' in all_connections_list:
                variable_dropout = all_connections_list['Dropout']
        else:
            all_connections = all_connections_list
        for i in range(len(all_connections)):
            if variable_dropout:
                self.drop_out = variable_dropout[i]
            x = self.FC_Block(all_connections[i], x, dropout=self.drop_out,
                              name=name + '_' + str(i))
        return x

    def do_FC_block(self, x):
        self.layer = 0
        self.desc = 'Encoder_FC'
        layer_order = []
        for layer in self.layer_names_fc:
            print(layer)
            if layer == 'Final':
                continue
            layer_order.append(layer)
            all_connections_list = self.layers_dict_FC[layer]['Encoding']
            x = self.run_FC_block(x, all_connections_list, name=self.desc + '_' + layer)
        layer_order.reverse()

        self.desc = 'Decoding_FC'
        for layer in layer_order:
            all_connections = self.layers_dict_FC[layer]['Decoding']
            x = self.run_FC_block(x, all_connections, name=self.desc + '_' + layer)
        if 'Final' in self.layers_dict_FC:
            self.desc = 'Final'
            all_connections = self.layers_dict_FC['Final']['Encoding']
            x = self.run_FC_block(x, all_connections, name=self.desc)
        return x

    def define_2D_or_3D(self, is_2D=False):
        self.is_2D = is_2D
        if is_2D:
            self.conv = Conv2D
            self.pool = MaxPooling2D
            self.up_sample = UpSampling2D
        else:
            self.conv = Conv3D
            self.pool = MaxPooling3D
            self.up_sample = UpSampling3D

    def define_batch_norm(self, batch_norm=False):
        self.batch_norm = batch_norm

    def define_filters(self, filters):
        self.filters = filters
        if len(filters) == 2:
            self.define_2D_or_3D(True)
        else:
            self.define_2D_or_3D()

    def define_activation(self, activation):
        self.activation = activation

    def define_pool_size(self, pool_size):
        self.pool_size = pool_size

    def define_padding(self, padding='same'):
        self.padding = padding

    def conv_block(self, output_size, x, name, strides=1, dialation_rate=1, activate=True, filters=None):
        if not filters:
            filters = self.filters
        if len(filters) + 1 == len(x.shape):
            self.define_2D_or_3D(is_2D=False)
            x = ExpandDimension(0)(x)
        elif len(filters) + 2 < len(x.shape):
            self.define_2D_or_3D(True)
            x = SqueezeDimension(0)(x)
        if not self.save_memory or max(filters) == 1:
            x = self.conv(output_size, filters, activation=None, padding=self.padding,
                          name=name, strides=strides, dilation_rate=dialation_rate)(x)
        else:
            for i in range(len(filters)):
                filter = np.ones(len(filters)).astype('int')
                filter[i] = filters[i]
                x = self.conv(output_size, filter, activation=None, padding=self.padding, name=name + '_' + str(i),
                              strides=strides, dilation_rate=dialation_rate)(x)  # Turn a 3x3 into a 3x1 with a 1x3
        if self.batch_norm:
            x = BatchNormalization()(x)
        if activate:
            x = Activation(self.activation, name=name + '_activation')(x)
        return x

    def residual_block(self, output_size, x, name, blocks=0):
        # This used to be input_val is the convolution
        if x.shape[-1] != output_size:
            x = self.conv_block(output_size, x=x, name=name + '_' + 'rescale_input', activate=False,
                                filters=self.filters)
            x = input_val = Activation(self.activation)(x)
        else:
            input_val = x

        for i in range(blocks):
            x = self.conv_block(output_size, x, name=name + '_' + str(i))
        x = self.conv(output_size, self.filters, activation=None, padding=self.padding, name=name)(x)
        x = Add(name=name + '_add')([x, input_val])
        x = Activation(self.activation)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def atrous_block(self, output_size, x, name,
                     rate_blocks=5):  # https://arxiv.org/pdf/1901.09203.pdf, follow formula of k^(n-1)
        # where n is the convolution layer number, this is for k = 3, 5 gives a field of 243x243
        rates = []
        get_new = True
        if x.shape[-1] == output_size:
            input_val = x
            get_new = False
        #     x = input_val = self.conv_block(output_size, x=x, name=name + 'Atrous_' + 'rescale_input', activate=True,
        #                                     filters=self.filters)
        # else:
        #     input_val = x
        for rate_block in range(rate_blocks):
            rate = []
            for i in range(len(self.filters)):
                rate.append(self.filters[i] ** (rate_block))  # not plus 1 minus 1, since we are 0 indexed
            # if len(rate) == 3 and rate[0] > 9:
            #     rate[0] = 9
            rates.append(rate)
        for i, rate in enumerate(rates):
            temp_name = name + 'Atrous_' + str(rate[-1])
            x = self.conv_block(output_size=output_size, x=x, name=temp_name, dialation_rate=rate, activate=False,
                                filters=self.filters)
            # x = self.conv(output_size,self.filters, activation=None,padding=self.padding, name=temp_name, dilation_rate=rate)(x)
            if i == len(rates) - 1:
                x = Add(name=name + '_add')([x, input_val])
            x = Activation(self.activation, name=temp_name + '_activation')(x)
            if i == 0 and get_new:
                input_val = x
            if self.batch_norm:
                x = BatchNormalization()(x)
        return x

    def strided_conv_block(self, output_size, x, name, strides=(2, 2, 2)):
        x = Conv3DTranspose(output_size, self.filters, strides=strides, padding=self.padding,
                            name=name)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(self.activation, name=name + '_activation')(x)
        return x

    def define_pooling_type(self, name='Max'):
        self.pooling_name = name

    def pooling_down_block(self, x, desc):
        if not self.is_2D:
            if self.pooling_name == 'Max':
                x = MaxPooling3D(pool_size=self.pool_size, name=desc)(x)
            elif self.pooling_name == 'Average':
                x = AveragePooling3D(pool_size=self.pool_size, name=desc)(x)
        else:
            if self.pooling_name == 'Max':
                x = MaxPooling2D(pool_size=self.pool_size, name=desc)(x)
            elif self.pooling_name == 'Average':
                x = AveragePooling2D(pool_size=self.pool_size, name=desc)(x)
        return x

    def shared_conv_block(self, x, y, output_size, name, strides=1):
        layer = Conv3D(output_size, self.filters, activation=None, padding=self.padding, name=name, strides=strides)
        x = layer(x)
        x = Activation(self.activation, name=name + '_activation')(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        y = layer(y)
        y = Activation(self.activation, name=name + '_activation')(y)
        if self.batch_norm:
            y = BatchNormalization()(y)
        return x, y

    def do_conv_block_enc(self, x):
        self.layer = 0
        layer_vals = {}
        desc = 'Encoder'
        self.layer_index = 0
        self.layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            self.layer_order.append(layer)
            all_filters = self.layers_dict[layer]['Encoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            layer_vals[self.layer_index] = x
            if 'Pooling' in self.layers_dict[layer]:
                self.define_pool_size(self.layers_dict[layer]['Pooling'])
            if len(self.layers_names) > 1:
                x = self.pooling_down_block(x, layer + '_Pooling')
            self.layer_index += 1
        self.concat = False
        if 'Base' in self.layers_dict:
            self.concat = True
            all_filters = self.layers_dict['Base']['Encoding']
            x = self.run_filter_dict(x, all_filters, 'Base_', '')
        return x, layer_vals

    def do_conv_block_decode(self, x, layer_vals=None):
        desc = 'Decoder'
        self.layer = 0
        self.layer_order.reverse()
        for layer in self.layer_order:
            if 'Decoding' not in self.layers_dict[layer]:
                continue
            print(layer)
            self.layer_index -= 1
            if 'Pooling' in self.layers_dict[layer]:
                self.define_pool_size(self.layers_dict[layer]['Pooling'])
            if self.concat:
                x = self.up_sample(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[self.layer_index]])
            all_filters = self.layers_dict[layer]['Decoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer += 1
        return x

    def dict_conv_block(self, x, filter_dict, desc):
        filter_vals = None
        res_blocks = None
        atrous_blocks = None
        activations = None
        if 'Res_block' in filter_dict:
            res_blocks = filter_dict['Res_block']
        if 'Kernel' in filter_dict:
            filter_vals = filter_dict['Kernel']
        if 'Atrous_block' in filter_dict:
            atrous_blocks = filter_dict['Atrous_block']
        if 'Activation' in filter_dict:
            activations = filter_dict['Activation']
        if 'Channels' in filter_dict:
            all_filters = filter_dict['Channels']
        elif 'FC' in filter_dict:
            if len(x.shape) != 2:
                x = Flatten()(x)
            for i, rate in enumerate(filter_dict['FC']):
                if activations:
                    self.define_activation(activations[i])
                x = self.FC_Block(rate, x, dropout=filter_dict['Dropout'][i], name=desc + '_FC_' + str(i))
            return x
        else:
            all_filters = filter_dict
        rescale = False
        for i in range(len(all_filters)):
            if activations:
                self.define_activation(activations[i])
            if filter_vals:
                self.define_filters(filter_vals[i])
                if len(filter_vals[i]) + 1 == len(x.shape):
                    self.define_2D_or_3D(is_2D=False)
                    x = ExpandDimension(0)(x)
                    rescale = True
                elif len(filter_vals[i]) + 1 > len(x.shape):
                    self.define_2D_or_3D(True)
                    x = SqueezeDimension(0)(x)
            strides = 1
            if rescale:
                self.desc = desc + '3D_' + str(i)
            else:
                self.desc = desc + str(i)

            if res_blocks:
                rate = res_blocks[i] if res_blocks else 0
                x = self.residual_block(all_filters[i], x=x, name=self.desc, blocks=rate)
            elif atrous_blocks:
                x = self.atrous_block(all_filters[i], x=x, name=self.desc, rate_blocks=atrous_blocks[i])
            else:
                x = self.conv_block(all_filters[i], x=x, strides=strides, name=self.desc)
        return x

    def run_filter_dict(self, x, layer_dict, layer, desc):
        if type(layer_dict) == list:
            for i, filters in enumerate(layer_dict):
                x = self.dict_conv_block(x, filters, layer + '_' + desc + '_' + str(i))
        else:
            x = self.dict_conv_block(x, layer_dict, layer + '_' + desc + '_')
        return x

    def run_unet(self, x):
        self.layer = 0
        self.layer_vals = {}
        desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = self.layers_dict[layer]['Encoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer_vals[layer_index] = x
            if 'Pooling' not in self.layers_dict[layer] or (
                    'Pooling' in self.layers_dict[layer] and self.layers_dict[layer]['Pooling'] is not None):
                if 'Pooling' in self.layers_dict[layer]:
                    self.define_pool_size(self.layers_dict[layer]['Pooling'])
                if 'Pooling_Type' in self.layers_dict[layer]:
                    self.define_pooling_type(self.layers_dict[layer]['Pooling_Type'])
                if len(self.layers_names) > 1:
                    x = self.pooling_down_block(x, layer + '_Pooling')
            layer_index += 1
        concat = False
        if 'Base' in self.layers_dict:
            concat = True
            all_filters = self.layers_dict['Base']['Encoding']
            x = self.run_filter_dict(x, all_filters, 'Base_', '')
        desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            if 'Decoding' not in self.layers_dict[layer]:
                continue
            print(layer)
            layer_index -= 1
            if 'Pooling' in self.layers_dict[layer]:
                self.define_pool_size(self.layers_dict[layer]['Pooling'])
            if concat:
                x = self.up_sample(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, self.layer_vals[layer_index]])
            all_filters = self.layers_dict[layer]['Decoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer += 1
        return x

class base_UNet(Unet):
    def __init__(self, filter_vals=(3, 3, 3), layers_dict=None, pool_size=(2, 2, 2), activation='elu', pool_type='Max',
                 batch_norm=False, is_2D=False, save_memory=False):
        super().__init__(save_memory=save_memory)
        self.layer_vals = {}
        self.define_2D_or_3D(is_2D)
        self.define_unet_dict(layers_dict)
        self.define_pool_size(pool_size)
        self.define_batch_norm(batch_norm)
        self.define_filters(filter_vals)
        self.define_activation(activation)
        self.define_padding('same')
        self.define_pooling_type(pool_type)

    def get_unet(self, layers_dict):
        pass

class my_UNet(base_UNet):

    def __init__(self, filter_vals=(3,3),layers_dict=None, pool_size=(2,2), activation='elu',pool_type='Max',
                 z_images=None,batch_norm=False, out_classes=2,image_size=512, num_channels=3, is_2D=True):
        self.image_size = image_size
        self.num_channels = num_channels
        self.z_images = z_images
        self.previous_conv = None
        if not layers_dict:
            print('Need to pass in a dictionary')
        self.is_2D = is_2D
        super().__init__(filter_vals=filter_vals, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm, is_2D=is_2D)
        self.striding_not_pooling = False
        self.out_classes = out_classes
        self.mask_input = False
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        if self.is_2D:
            image_input_primary = x = Input(shape=(self.image_size, self.image_size, self.num_channels), name='UNet_Input')
            output_kernel = (1,1)
        else:
            image_input_primary = x = Input(shape=(self.z_images, self.image_size, self.image_size, self.num_channels), name='UNet_Input')
            output_kernel = (1,1,1)
        x = self.run_unet(x)
        self.save_memory = False
        self.define_filters(output_kernel)
        x = self.conv_block(self.out_classes, x, name='output', activate=False)
        x = Activation('softmax')(x)
        inputs = image_input_primary
        model = Model(inputs=inputs, outputs=x)
        self.created_model = model


def main():
    pass

if __name__ == '__main__':
    main()
