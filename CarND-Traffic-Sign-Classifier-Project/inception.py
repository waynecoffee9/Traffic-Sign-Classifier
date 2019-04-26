# Class incepNet contains scaled down inception neural network v4
# Reference: https://arxiv.org/pdf/1602.07261.pdf
import cv2
import numpy as np
import tensorflow as tf
from math import ceil

class incepNet():
    def __init__(self):
        self.all_weights = []
        self.keep_prob = tf.placeholder(tf.float32)
        self.layer_features = []

    def conv_layer(self, features, in_size, param, pad, layer_name, mu=0, sigma=0.1):
        # param = ([filter_h, filter_w, input_depth, output_depth],
        #          [conv_stride_batch, conv_stride_h, conv_stride_w, conv_stride_ch])
        bias_name = 'bias_' + layer_name
        weight_name = 'weight_' + layer_name
        conv_W = tf.Variable(tf.truncated_normal(param[0], mean=mu, stddev=sigma), name=weight_name)
        conv_b = tf.Variable(tf.truncated_normal([param[0][3]], mean=mu, stddev=sigma), name=bias_name)
        self.all_weights.append(conv_W)
        conv = tf.nn.conv2d(features, conv_W, strides=param[1], padding=pad)
        conv = tf.nn.bias_add(conv, conv_b)
        conv = tf.nn.relu(conv)
        self.layer_features.append(conv)
        conv = tf.nn.dropout(conv, self.keep_prob)
        conv_size = [None]*3
        if pad == 'SAME':
            conv_size[0] = ceil(float(in_size[0])/float(param[1][1]))
            conv_size[1] = ceil(float(in_size[1])/float(param[1][2]))
        elif pad == 'VALID':
            conv_size[0] = ceil(float(in_size[0]-param[0][0]+1)/float(param[1][1]))
            conv_size[1] = ceil(float(in_size[1]-param[0][1]+1)/float(param[1][2]))
        conv_size[2] = param[0][3]
        return conv, conv_size

    def max_pooling(self, features, feature_size, param, pad):
        # param = ([pool_ksize_batch, pool_ksize_h, pool_ksize_w, pool_ksize_ch]
        #          [pool_stride_batch, pool_stride_h, pool_stride_w, pool_stride_ch])
        pooling = tf.nn.max_pool(features, ksize=param[0], strides = param[1], padding=pad)
        pool_size = [None]*3
        if pad == 'SAME':
            pool_size[0] = ceil(float(feature_size[0])/float(param[1][1]))
            pool_size[1] = ceil(float(feature_size[1])/float(param[1][2]))
        elif pad == 'VALID':
            pool_size[0] = ceil(float(feature_size[0]-param[0][1]+1)/float(param[1][1]))
            pool_size[1] = ceil(float(feature_size[1]-param[0][2]+1)/float(param[1][2]))
        pool_size[2] = feature_size[2]
        return pooling, pool_size

    def avg_pooling(self, features, feature_size, param, pad):
        # param = ([pool_ksize_batch, pool_ksize_h, pool_ksize_w, pool_ksize_ch]
        #          [pool_stride_batch, pool_stride_h, pool_stride_w, pool_stride_ch])
        pooling = tf.nn.avg_pool(features, ksize=param[0], strides = param[1], padding=pad)
        pool_size = [None]*3
        if pad == 'SAME':
            pool_size[0] = ceil(float(feature_size[0])/float(param[1][1]))
            pool_size[1] = ceil(float(feature_size[1])/float(param[1][2]))
        elif pad == 'VALID':
            pool_size[0] = ceil(float(feature_size[0]-param[0][1]+1)/float(param[1][1]))
            pool_size[1] = ceil(float(feature_size[1]-param[0][2]+1)/float(param[1][2]))
        pool_size[2] = feature_size[2]
        return pooling, pool_size

    def stem(self, features, in_size):

        param = ([3,3,in_size[2],32], [1,1,1,1])
        features, in_size = self.conv_layer(features, in_size, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))
        param = ([3,3,in_size[2],32], [1,1,1,1])
        features, in_size = self.conv_layer(features, in_size, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))

        param = ([3,3,in_size[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))
        param = ([1,3,3,1], [1,1,1,1])
        features_2, in_size_2 = self.max_pooling(features, in_size, param, 'SAME')

        features = tf.concat([features_1, features_2], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2]

        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))
        param = ([7,1,in_size_1[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))
        param = ([1,7,in_size_1[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_1[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_2[2],64], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features_2, in_size_2, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))

        features = tf.concat([features_1, features_2], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2]

        param = ([1,2,2,1], [1,2,2,1])
        features_1, in_size_1 = self.max_pooling(features, in_size, param, 'SAME')
        param = ([3,3,in_size[2],96], [1,2,2,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'stem_' + str(len(self.all_weights)+1))

        features = tf.concat([features_1, features_2], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2]

        return features, in_size

    def inception_A(self, features, in_size):

        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_1[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_1[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_2[2],64], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features_2, in_size_2, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],64], [1,1,1,1])
        features_3, in_size_3 = self.conv_layer(features, in_size, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))

        param = ([1,2,2,1], [1,1,1,1])
        features_4, in_size_4 = self.avg_pooling(features, in_size, param, 'SAME')
        param = ([1,1,in_size_4[2],64], [1,1,1,1])
        features_4, in_size_4 = self.conv_layer(features_4, in_size_4, param, 'SAME', 'incep_A' + str(len(self.all_weights)+1))

        features = tf.concat([features_1, features_2, features_3, features_4], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2] + in_size_3[2] + in_size_4[2]

        return features, in_size

    def reduction_A(self, features, in_size):
        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'reduc_A' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_1[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'reduc_A' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_1[2],64], [1,2,2,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'reduc_A' + str(len(self.all_weights)+1))

        param = ([3,3,in_size[2],64], [1,2,2,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'reduc_A' + str(len(self.all_weights)+1))

        param = ([1,2,2,1], [1,2,2,1])
        features_3, in_size_3 = self.max_pooling(features, in_size, param, 'SAME')

        features = tf.concat([features_1, features_2, features_3], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2] + in_size_3[2]

        return features, in_size

    def inception_B(self, features, in_size):
        param = ([1,1,in_size[2],36], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))
        param = ([1,7,in_size_1[2],36], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))
        param = ([7,1,in_size_1[2],36], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))
        param = ([1,7,in_size_1[2],36], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))
        param = ([7,1,in_size_1[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],36], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))
        param = ([1,7,in_size_2[2],36], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features_2, in_size_2, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))
        param = ([1,7,in_size_2[2],48], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features_2, in_size_2, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],64], [1,1,1,1])
        features_3, in_size_3 = self.conv_layer(features, in_size, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))

        param = ([1,2,2,1], [1,1,1,1])
        features_4, in_size_4 = self.avg_pooling(features, in_size, param, 'SAME')
        param = ([1,1,in_size_4[2],24], [1,1,1,1])
        features_4, in_size_4 = self.conv_layer(features_4, in_size_4, param, 'SAME', 'incep_B' + str(len(self.all_weights)+1))

        features = tf.concat([features_1, features_2, features_3, features_4], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2] + in_size_3[2] + in_size_4[2]

        return features, in_size

    def reduction_B(self, features, in_size):
        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'reduc_B' + str(len(self.all_weights)+1))
        param = ([1,7,in_size_1[2],48], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'reduc_B' + str(len(self.all_weights)+1))
        param = ([7,1,in_size_1[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'reduc_B' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_1[2],64], [1,2,2,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'VALID', 'reduc_B' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],36], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'reduc_B' + str(len(self.all_weights)+1))
        param = ([3,3,in_size_2[2],36], [1,2,2,1])
        features_2, in_size_2 = self.conv_layer(features_2, in_size_2, param, 'VALID', 'reduc_B' + str(len(self.all_weights)+1))

        param = ([1,3,3,1], [1,2,2,1])
        features_3, in_size_3 = self.max_pooling(features, in_size, param, 'VALID')

        features = tf.concat([features_1, features_2, features_3], 3)
        in_size = in_size_1
        in_size[2] = in_size_1[2] + in_size_2[2] + in_size_3[2]

        return features, in_size

    def inception_C(self, features, in_size):
        param = ([1,1,in_size[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features, in_size, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))
        param = ([1,3,in_size_1[2],64], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))
        param = ([3,1,in_size_1[2],72], [1,1,1,1])
        features_1, in_size_1 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))
        param = ([1,3,in_size_1[2],48], [1,1,1,1])
        features_11, in_size_11 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))
        param = ([3,1,in_size_1[2],48], [1,1,1,1])
        features_12, in_size_12 = self.conv_layer(features_1, in_size_1, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],64], [1,1,1,1])
        features_2, in_size_2 = self.conv_layer(features, in_size, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))
        param = ([1,3,in_size_2[2],48], [1,1,1,1])
        features_21, in_size_21 = self.conv_layer(features_2, in_size_2, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))
        param = ([3,1,in_size_2[2],48], [1,1,1,1])
        features_22, in_size_22 = self.conv_layer(features_2, in_size_2, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))

        param = ([1,1,in_size[2],48], [1,1,1,1])
        features_3, in_size_3 = self.conv_layer(features, in_size, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))

        param = ([1,2,2,1], [1,1,1,1])
        features_4, in_size_4 = self.avg_pooling(features, in_size, param, 'SAME')
        param = ([1,1,in_size_4[2],48], [1,1,1,1])
        features_4, in_size_4 = self.conv_layer(features_4, in_size_4, param, 'SAME', 'incep_C' + str(len(self.all_weights)+1))

        features = tf.concat([features_11, features_12, features_21, features_22, features_3, features_4], 3)
        in_size = in_size_1
        in_size[2] = in_size_11[2] + in_size_12[2] + in_size_21[2] + in_size_22[2] + in_size_3[2] + in_size_4[2]

        return features, in_size



