#--*-- coding:utf8 --*-- 
'''
Program:
    This code show us how to predict the classifier of image through vgg19.
'''
import os 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np 

import scipy.io 
import scipy.misc

from imagenet_classes import class_names 

def _conv2d(x, weights, bias):

    conv = tf.nn.conv2d(x, weights, strides = [1, 1, 1, 1], padding = 'SAME')

    return tf.nn.bias_add(conv, bias)

def _pool(x, method='max'):

    if method == 'max':

        result = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    elif method == 'avg':

        result = tf.nn.avg_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    else:

        print "input the method is error."

    return result 

def cnn_vgg19(x, layers):

    layers_name = (
    'conv1_1', 'conv1_2', 'pool1',
    'conv2_1', 'conv2_2', 'pool2', 
    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5',
    'fc1', 'fc2' , 'fc3', 
    'softmax'
    )

    cnn = {}

    l_idx = 0

    cnn['src_img'] = x
    
    for idx,item in enumerate(layers):
        
        item_tmp = item[0][0][0][0]

        if str(item_tmp)[:4] == 'relu':
            
            continue

        elif str(item_tmp)[:4] == 'pool':

            x = _pool(x)

        elif str(item_tmp)[:4] == 'soft':

            x = tf.nn.softmax(x)

        elif idx == 37:

            prod_shape = np.prod(x.shape)

            x = tf.reshape(x, [-1, prod_shape])

            weights, bias = item_tmp[0], item_tmp[1] 

            weights = weights.reshape([-1, 4096])

            bias = bias.reshape([-1])

            x = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), bias))

        elif idx == 39:

            weights, bias = item_tmp[0], item_tmp[1] 

            weights = weights.reshape([-1, 4096])

            bias = bias.reshape([-1])

            x = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), bias))

        elif idx == 41:

            weights, bias = item_tmp[0], item_tmp[1] 

            weights = weights.reshape([-1, 1000])

            bias = bias.reshape([-1])

            x = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), bias))

        else :

            weights, bias = item_tmp[0], item_tmp[1]

            bias = bias.reshape(-1)

            x = tf.nn.relu(_conv2d(x, weights, bias))

        cnn[layers_name[l_idx]] = x 

        l_idx += 1

    return cnn
    

if __name__ == '__main__':

    vgg_path = './imagenet-vgg-verydeep-19.mat' 

    vgg19_mat = scipy.io.loadmat(vgg_path)

    image_path = './tidy.jpg'

    image = scipy.misc.imread(image_path)

    image = scipy.misc.imresize(image, [224, 224, 3])

    mean = vgg19_mat['normalization'][0][0][0][0][0]

    layers = vgg19_mat['layers'][0]

    #the shape of x is change from [224, 224, 3] to [1, 224, 224, 3]
    normal_image = np.array([image - mean]).astype(np.float32)

    x = tf.placeholder(tf.float32, shape = normal_image.shape)

    cnn = cnn_vgg19(x, layers)
    
    with tf.Session() as sess:

        result_cnn = sess.run(cnn, feed_dict={x:normal_image})

        list_softmax = result_cnn['softmax'].reshape(-1).tolist()

        list_sort_idx = list(reversed(np.argsort(list_softmax)))

        for i in range(3):

            print "it's : ", class_names[list_sort_idx[i]]

            print "the probability is: ", list_softmax[list_sort_idx[i]]

    print "done"



