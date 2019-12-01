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

#making a dictionary for vgg19 values
data_dict = np.load('vgg19.npy', encoding='latin1').item()

columns = defaultdict(list)

with open('../labels.csv') as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items():  # go over each column name and value
            columns[k].append(v)

path_train = '../trainval/*/*.jpg'
files_train = glob.glob(path_train)
print(type(files_train))
n_train = len(files_train)
print(n_train)

path_test = '../deploy/test/*/*.jpg'
files_test = glob.glob(path_test)
n_test = len(files_test)

batch_size =  100

labels = np.zeros(n_train)

for i in range(len(files_train)):
    l2 = list(files_train[i].split('/')[0:-1])
    l3 = list(os.path.split(os.path.abspath(files_train[i])))
    l4 = list(l3[1].split('_')[0:-1])
    list_label = columns["label"]
    labels[i] = list_label[i]


def batch_generator(e_images,e_labels,which_batch):

    if (which_batch)*batch_size > len(e_images):
        n_images = len(e_images) - (which_batch-1)*batch_size
    else :
        n_images = batch_size

    print("Batch_size",n_images)

    b_images = np.zeros((n_images,)+image_size)

    for j in range(n_images) :
        b_images[j,:,:,:] = skimage.transform.resize(imread(e_images[(which_batch-1)*batch_size + j]),image_size)

    b_labels = np.array(e_labels[(which_batch-1)*batch_size : (which_batch-1)*batch_size + n_images])

    return b_images, b_labels

if n_train%batch_size ==0:
    num_batches = int(n_train/batch_size)
else :
    num_batches = int(n_train/batch_size)+1

def noise(size):
    return np.random.normal(size=size)

def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        #vgg19_convolutional layers
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

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
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


X = tf.placeholder(tf.float32, shape=(None, ) + image_size)
X_labels = tf.placeholder(tf.uint8, shape=(None,))
print("shape of X_labels",X_labels.shape)
y = tf.one_hot(X_labels,4)
print("shape of y", y.shape)

y_ = discriminator(X)

#node for accuracy
correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#node for loss
D_loss = tf.reduce_mean(
    nn.softmax_cross_entropy_with_logits(
        logits=y_, labels = y))

train_vars = tf.trainable_variables()

D_vars = [var for var in train_vars if 'discriminator' in var.name]

print("Discriminator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in D_vars])))

learning_rate = tf.placeholder(tf.float32, shape=[])
D_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)


# Total number of epochs to train
num_epochs = 100

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()

saver = tf.train.Saver()

# Initial SR training by itself
batch_start_time = time.time()

loss_batch = np.zeros((num_epochs*num_batches,1))
accuracy_batch= np.zeros((num_epochs*num_batches,1))

loss_epoch = np.zeros((num_epochs,1))
accuracy_epoch = np.zeros((num_epochs,1))


for epoch in range(1,num_epochs+1):
    d_Loss = 0
    d_Accuracy = 0
    lr = 1e-4

    if epoch > 1:
        saver.save(session, "./model_{}.ckpt".format(epoch))

    #randomize the files
    d_indices = list(range(n_train))
    random.shuffle(d_indices)
    e_images = [files_train[i] for i in d_indices]
    e_labels = np.array([labels[i] for i in d_indices])

    for i in range(num_batches):
        #get the b_images and b_labels
        n_batch = i+1
        b_images,b_labels = batch_generator(e_images,e_labels,n_batch)

        #Train Discriminator
        feed_dict = {X: b_images, X_labels: b_labels, learning_rate: lr}
        _,Loss, Accuracy = session.run([D_opt, D_loss, accuracy], feed_dict=feed_dict)
        loss_batch[(epoch-1)*num_batches + n_batch - 1,:]  = Loss
        accuracy_batch[(epoch-1)*num_batches + n_batch - 1,:] = Accuracy
        np.savetxt("epoch_batch_data.txt",np.column_stack((loss_batch,accuracy_batch)))
        # Display Progress every few batches
        if n_batch % 2 == 0:
            #print the time taken by the bathces
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time

            print("Batches took {:.3f} ms".format(elapsed * 1000))
            print("epoch:",epoch,"/",num_epochs,"n_batches:",n_batch,"/",num_batches, "Loss:", Loss,
                  "Accuracy:" ,Accuracy)
    #print stats for entire epoch
    for i in range(num_batches):
        #get the b_images and b_labels
        n_batch = i+1
        b_images,b_labels = batch_generator(e_images,e_labels,n_batch)
        if (n_batch)*batch_size > len(e_images):
            n_images = len(e_images) - (n_batch-1)*batch_size
        else :
            n_images = batch_size

        #Train Discriminator
        feed_dict = {X: b_images, X_labels: b_labels, learning_rate: lr}
        Loss, Accuracy = session.run([D_loss, accuracy], feed_dict=feed_dict)
        d_Loss += Loss*n_images
        d_Accuracy += Accuracy * n_images
    loss_epoch[(epoch-1),:]  = d_Loss/n_train
    accuracy_epoch[(epoch-1),:] = d_Accuracy/n_train
    np.savetxt("epoch_epoch_data.txt",np.column_stack((loss_epoch,accuracy_epoch)))
    print("-----------------------------------------------\nepoch:",epoch,"/",num_epochs,"n_batches:","Loss:", d_Loss/n_train,
                  "Accuracy:" ,d_Accuracy/n_train,"\n-------------------------------------------------")



saver.save(session, "./model_full.ckpt")
