from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] += lam * self.params[n]


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size = input_size[0]
        channels = self.number_filters
        w_numerator = input_size[2] - self.kernel_size + 2*self.padding
        h_numerator = input_size[1] - self.kernel_size + 2*self.padding
        width = int(np.floor(w_numerator / self.stride) + 1)
        height = int(np.floor(h_numerator / self.stride) + 1)
        output_shape = [batch_size, height, width, channels]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        padded_img = np.pad(img, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), 'constant')
        input_matrix = np.zeros((img.shape[0], output_height*output_width, self.kernel_size*self.kernel_size*img.shape[-1]))
        for i in range(output_height):
            for j in range(output_width):
                input_matrix[:, i*output_width+j, :] = padded_img[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :].reshape(img.shape[0], -1)
                
        filter_matrix = self.params[self.w_name].reshape(-1, self.number_filters)
        conv_matrix = input_matrix.dot(filter_matrix) + self.params[self.b_name]
        output = conv_matrix.reshape(-1, output_height, output_width, self.number_filters)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        batch_size, input_height, input_width, input_channels = img.shape
        _, output_height, output_width, number_filters = self.get_output_size(img.shape)
        kernel_height, kernel_width, _, _ = self.params[self.w_name].shape

        filter_matrix = self.params[self.w_name].reshape(-1, number_filters)

        padded_img = np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')

        dinput_matrix = np.zeros((batch_size, input_height + 2*self.padding, input_width + 2*self.padding, input_channels))
        dfilter_matrix = np.zeros((kernel_height, kernel_width, input_channels, number_filters))
        dbias = np.zeros(number_filters)

        dconv_matrix = dprev
        for i in range(output_height):
            for j in range(output_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + kernel_height
                end_j = start_j + kernel_width
                input_matrix = padded_img[:, start_i:end_i, start_j:end_j, :]
                input_matrix = input_matrix.reshape(batch_size, -1)
                dfilter_matrix += (input_matrix.T.dot(dconv_matrix[:, i, j, :])).reshape(kernel_height, kernel_width, input_channels, number_filters)
                dinput_matrix[:, start_i:end_i, start_j:end_j, :] += (dconv_matrix[:, i, j, :].dot(filter_matrix.T)).reshape(batch_size, kernel_height, kernel_width, input_channels)
        dbias += np.sum(dconv_matrix, axis=(0, 1, 2))

        dimg = dinput_matrix[:, self.padding:-self.padding, self.padding:-self.padding, :]

        self.grads[self.w_name] = dfilter_matrix
        self.grads[self.b_name] = dbias
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        N, H, W, C = img.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        stride = self.stride

        out_h = int(1 + (H - pool_height) / stride)
        out_w = int(1 + (W - pool_width) / stride)

        output = np.zeros((N, out_h, out_w, C))

        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    vert_start, vert_end = i * stride, i * stride + pool_height
                    horiz_start, horiz_end = j * stride, j * stride + pool_width
                    pool_region = img[n, vert_start:vert_end, horiz_start:horiz_end, :]

                    output[n, i, j, :] = np.amax(pool_region, axis=(0, 1))

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        for n in range(dprev.shape[0]):
            for i in range(h_out):
                for j in range(w_out):
                    for c in range(dprev.shape[3]):
                        vert_start, vert_end = i * self.stride, i * self.stride + h_pool
                        horiz_start, horiz_end = j * self.stride, j * self.stride + w_pool
                        pool_region = img[n, vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = pool_region == np.max(pool_region)
                        dimg[n, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dprev[n, i, j, c]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
