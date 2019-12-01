#!/usr/bin/python3

import time
import random
import os
import skimage.transform
import tensorflow as tf
from tensorflow import nn, layers
import numpy as np
from matplotlib.image import imread
import glob
import csv
from collections import defaultdict
from skimage import transform

# Set random seem for reproducibility
manualSeed = 999
np.random.seed(manualSeed)

image_size = (224, 224, 3)

# making a dictionary for vgg19 values
data_dict = np.load('vgg19.npy', encoding='latin1').item()

columns = defaultdict(list)

path_test = '../test/*/*.jpg'
files_test = glob.glob(path_test)
n_test = len(files_test)


names =  []
for i in range(len(files_test)):
    l2 = list(files_test[i].split('\\')[1])
    l3 = list(os.path.split(os.path.abspath(files_test[i])))
    l4 = list(l3[1].split('_')[0:-1])
    names.append(str(''.join(l2) + '/' + l4[0]))


def batch_generator(e_images, which_batch):
    if (which_batch) * batch_size > len(e_images):
        n_images = len(e_images) - (which_batch - 1) * batch_size
    else:
        n_images = batch_size

    print("Batch_size", n_images)

    b_images = np.zeros((n_images, 224, 224, 3))

    for j in range(n_images):
        b_images[j, :, :, :] = skimage.transform.resize(imread(e_images[(which_batch - 1) * batch_size + j]),
                                                        (224, 224))
    return b_images


batch_size = 100
num_batches = int(round(n_test / batch_size))


def noise(size):
    return np.random.normal(size=size)


def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # vgg19_convolutional layers
        with tf.variable_scope("conv1"):
            conv1_1 = conv_layer(x, "conv1_1")
            conv1_2 = conv_layer(conv1_1, "conv1_2")
            pool1 = max_pool(conv1_2, "pool1")

        with tf.variable_scope("conv2"):
            conv2_1 = conv_layer(pool1, "conv2_1")
            conv2_2 = conv_layer(conv2_1, "conv2_2")
            pool2 = max_pool(conv2_2, "pool2")

        with tf.variable_scope("conv3"):
            conv3_1 = conv_layer(pool2, "conv3_1")
            conv3_2 = conv_layer(conv3_1, "conv3_2")
            conv3_3 = conv_layer(conv3_2, "conv3_3")
            conv3_4 = conv_layer(conv3_3, "conv3_4")
            pool3 = max_pool(conv3_4, "pool3")

        with tf.variable_scope("conv4"):
            conv4_1 = conv_layer(pool3, "conv4_1")
            conv4_2 = conv_layer(conv4_1, "conv4_2")
            conv4_3 = conv_layer(conv4_2, "conv4_3")
            conv4_4 = conv_layer(conv4_3, "conv4_4")
            pool4 = max_pool(conv4_4, "pool4")

        with tf.variable_scope("conv5"):
            conv5_1 = conv_layer(pool4, "conv5_1")
            conv5_2 = conv_layer(conv5_1, "conv5_2")
            conv5_3 = conv_layer(conv5_2, "conv5_3")
            conv5_4 = conv_layer(conv5_3, "conv5_4")
            # pool5 = max_pool(conv5_4, 'pool5')
        with tf.variable_scope("linear"):
            linear = layers.flatten(conv5_4)
            y_ = layers.dense(linear, 4, use_bias=False, kernel_initializer=tf.initializers.random_normal(0.0, 0.1))
    return y_


def fc_layer(bottom, name):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name)
        biases = get_bias(name)
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, name):
    with tf.variable_scope(name):
        filt = get_conv_filter(name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu


def get_conv_filter(name):
    return tf.constant(data_dict[name][0], name="filter")


def get_fc_weight(self, name):
    return tf.constant(data_dict[name][0], name="weights")


def get_bias(name):
    return tf.constant(data_dict[name][1], name="biases")


X = tf.placeholder(tf.float32, shape=(None,) + image_size)
y_ = discriminator(X)
hard_y_ = tf.argmax(y_,1)

train_vars = tf.trainable_variables()
D_vars = [var for var in train_vars if 'discriminator' in var.name]

print("Discriminator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in D_vars])))

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()

saver = tf.train.Saver()

saver.restore(session, "model_38.ckpt")

labels = np.zeros((n_test,1))

for i in range(num_batches):
    # randomize the files
    d_indices = list(range(n_test))

    e_images = [files_test[i] for i in d_indices]

    b_images = batch_generator(e_images, i+1)
    if (i+1) * batch_size > len(e_images):
        n_images = len(e_images) - (which_batch - 1) * batch_size
    else:
        n_images = batch_size

    # Train Discriminator
    feed_dict = {X: b_images}

    b_labels= session.run([hard_y_], feed_dict=feed_dict)


    labels[i*batch_size : i*batch_size + n_images,0] =b_labels[0]


labels = labels.astype(int)

names = np.reshape(np.array(names),(n_test,1))
final = np.column_stack((names,labels))
np.savetxt('submission2.csv', final, delimiter=',', header = "guid/image,label",fmt='%s',comments ='')
